import random
from typing import List, Tuple

import torch

from vllm._custom_ops import build_sparse_block_table


def kv_token_to_hashable(k: torch.Tensor, v: torch.Tensor) -> Tuple:
    return (
        tuple(int(x) for x in k.tolist()),
        tuple(int(x) for x in v.tolist()),
    )


def write_token_to_block_storage(
    block_storage: torch.Tensor,  # [2, total_blocks, block_size, dim]
    bid: int,
    off: int,
    kvec: torch.Tensor,
    vvec: torch.Tensor,
):
    block_storage[0, bid, off].copy_(kvec)
    block_storage[1, bid, off].copy_(vvec)


def read_token_from_block_storage(
    block_storage: torch.Tensor,
    bid: int,
    off: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = block_storage[0, bid, off].clone()
    v = block_storage[1, bid, off].clone()
    return k, v


def build_reference_for_one_row(
    q: int,
    hq: int,
    top_clusters: torch.Tensor,               # [NQ,Hq,nprobe], CPU
    cluster_compact_block_ids: torch.Tensor,  # [Hkv,C,maxB], CPU
    cluster_temp_kv_pos: torch.Tensor,        # [Hkv,C,block_size,2], CPU
    cluster_total_kv_counts: torch.Tensor,    # [Hkv,C], CPU
    temp_block_ids: torch.Tensor,             # [max_temp_blocks], CPU
    block_storage_before: torch.Tensor,       # CPU
) -> Tuple[List[int], int, List[Tuple[Tuple, Tuple]], int]:
    """
    返回：
      ref_compact_prefix
      ref_seqused_k
      ref_temp_stream_hashable
      free_blocks_needed
    """
    NQ, Hq, nprobe = top_clusters.shape
    Hkv, C, maxB = cluster_compact_block_ids.shape
    _, total_blocks, block_size, dim = block_storage_before.shape

    assert Hq % Hkv == 0
    q_per_kv = Hq // Hkv
    hkv = hq // q_per_kv

    ref_compact_prefix: List[int] = []
    ref_temp_stream_hashable: List[Tuple[Tuple, Tuple]] = []

    seqused_k = 0

    for kp in range(nprobe):
        cid = int(top_clusters[q, hq, kp].item())
        assert 0 <= cid < C

        total_cnt = int(cluster_total_kv_counts[hkv, cid].item())
        if total_cnt <= 0:
            continue

        seqused_k += total_cnt
        full_nb = total_cnt // block_size
        tail = total_cnt % block_size

        # compact blocks
        for bi in range(full_nb):
            bid = int(cluster_compact_block_ids[hkv, cid, bi].item())
            assert bid >= 0
            ref_compact_prefix.append(bid)

        # temp token stream
        for slot in range(tail):
            tb_idx = int(cluster_temp_kv_pos[hkv, cid, slot, 0].item())
            tb_off = int(cluster_temp_kv_pos[hkv, cid, slot, 1].item())
            assert tb_idx >= 0
            assert 0 <= tb_off < block_size

            src_bid = int(temp_block_ids[tb_idx].item())
            assert src_bid >= 0

            kvec, vvec = read_token_from_block_storage(
                block_storage_before, src_bid, tb_off
            )
            ref_temp_stream_hashable.append(
                kv_token_to_hashable(
                    kvec.to(torch.int32),
                    vvec.to(torch.int32),
                )
            )

    free_blocks_needed = (len(ref_temp_stream_hashable) + block_size - 1) // block_size
    return ref_compact_prefix, seqused_k, ref_temp_stream_hashable, free_blocks_needed


def check_packed_temp_blocks_for_one_row(
    q: int,
    hq: int,
    out_block_table: torch.Tensor,       # [NQ,Hq,max_bt_len], CPU
    out_bt_len: torch.Tensor,            # [NQ,Hq], CPU
    compact_prefix_len: int,
    ref_temp_stream_hashable: List[Tuple[Tuple, Tuple]],
    free_blocks_needed: int,
    block_storage_after: torch.Tensor,   # CPU
    block_size: int,
    free_block_ids: torch.Tensor,        # CPU
):
    bt_len = int(out_bt_len[q, hq].item())
    row_ids = out_block_table[q, hq, :bt_len].tolist()

    # 末尾 free_blocks_needed 个 block 就是 packed temp free blocks
    packed_bids = row_ids[compact_prefix_len:]

    assert len(packed_bids) == free_blocks_needed, (
        f"packed block count mismatch at (q={q}, hq={hq})\n"
        f"expected={free_blocks_needed}\n"
        f"got={len(packed_bids)}"
    )

    # packed block id 不要求和 reference 一样，但必须来自 free_block_ids
    free_id_set = set(int(x) for x in free_block_ids.tolist())
    for bid in packed_bids:
        assert bid in free_id_set, (
            f"packed block id {bid} is not from free_block_ids at (q={q}, hq={hq})"
        )

    # 校验 packed 内容是否和 reference temp stream 一致
    for t, ref_item in enumerate(ref_temp_stream_hashable):
        dst_bid = packed_bids[t // block_size]
        dst_off = t % block_size
        k, v = read_token_from_block_storage(block_storage_after, dst_bid, dst_off)
        got_item = kv_token_to_hashable(
            k.to(torch.int32),
            v.to(torch.int32),
        )
        assert got_item == ref_item, (
            f"packed temp mismatch at (q={q}, hq={hq}, temp_idx={t})\n"
            f"expected={ref_item}\n"
            f"got={got_item}"
        )

    # 除最后一个 free block 外，其余都必须满
    if free_blocks_needed > 0:
        total_temp = len(ref_temp_stream_hashable)
        for i in range(free_blocks_needed):
            start = i * block_size
            end = min(start + block_size, total_temp)
            used = end - start
            if i < free_blocks_needed - 1:
                assert used == block_size, (
                    f"non-last packed free block is not full at "
                    f"(q={q}, hq={hq}, block_idx={i}), used={used}, block_size={block_size}"
                )
            else:
                assert 1 <= used <= block_size, (
                    f"last packed free block has invalid used count at "
                    f"(q={q}, hq={hq}, block_idx={i}), used={used}, block_size={block_size}"
                )


def make_unique_top_clusters(
    NQ: int,
    Hq: int,
    nprobe: int,
    C: int,
    device: torch.device,
) -> torch.Tensor:
    """
    每个 row 内 cluster id 唯一，长度固定为 nprobe。
    要求 nprobe <= C。
    """
    assert nprobe <= C, f"nprobe ({nprobe}) must be <= C ({C})"

    top_clusters = torch.empty(
        (NQ, Hq, nprobe), dtype=torch.int32, device=device
    )

    for q in range(NQ):
        for hq in range(Hq):
            picked = random.sample(range(C), nprobe)
            top_clusters[q, hq] = torch.tensor(
                picked, dtype=torch.int32, device=device
            )

    return top_clusters


def run_one_case(case_id: int, device: torch.device):
    random.seed(1000 + case_id)
    torch.manual_seed(1000 + case_id)

    NQ = random.choice([1, 2, 3, 4])
    Hkv = random.choice([1, 2, 4])
    q_per_kv = random.choice([1, 2, 4])
    Hq = Hkv * q_per_kv

    C = random.choice([2, 3, 4, 5, 6])
    nprobe = random.randint(1, C)

    block_size = random.choice([8, 16])
    dim = random.choice([16, 32, 64])

    maxB = 8
    max_temp_blocks = 64
    total_blocks = 512
    max_bt_len = nprobe * (maxB + block_size)

    dtype = random.choice([torch.float32, torch.float16, torch.bfloat16])

    # block_storage 初始化
    block_storage = torch.full(
        (2, total_blocks, block_size, dim),
        fill_value=-777,
        dtype=dtype,
        device=device,
    )

    cluster_compact_block_ids = torch.full(
        (Hkv, C, maxB), -1, dtype=torch.int32, device=device
    )
    cluster_temp_kv_pos = torch.full(
        (Hkv, C, block_size, 2), -1, dtype=torch.int32, device=device
    )
    cluster_total_kv_counts = torch.zeros(
        (Hkv, C), dtype=torch.int32, device=device
    )

    # temp_block_ids / free_block_ids 分离
    all_block_ids = list(range(total_blocks))
    temp_block_id_list = random.sample(all_block_ids, max_temp_blocks)
    remain = [x for x in all_block_ids if x not in temp_block_id_list]
    free_block_id_list = random.sample(remain, min(len(remain), 256))

    temp_block_ids = torch.tensor(
        temp_block_id_list, dtype=torch.int32, device=device
    )
    free_block_ids = torch.tensor(
        free_block_id_list, dtype=torch.int32, device=device
    )

    # 构造 cluster metadata
    used_compact_phys = set()
    temp_write_cursor = 0

    for h in range(Hkv):
        for cid in range(C):
            total_cnt = random.choice([0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23])
            full_nb = total_cnt // block_size
            tail = total_cnt % block_size

            assert full_nb <= maxB
            cluster_total_kv_counts[h, cid] = total_cnt

            # compact blocks
            compact_candidates = [x for x in remain if x not in used_compact_phys]
            assert len(compact_candidates) >= full_nb
            compact_bids = random.sample(compact_candidates, full_nb)

            for bi, bid in enumerate(compact_bids):
                cluster_compact_block_ids[h, cid, bi] = bid
                used_compact_phys.add(bid)

                for slot in range(block_size):
                    base = h * 100000 + cid * 10000 + bi * 100 + slot
                    kval = torch.full(
                        (dim,), base, dtype=torch.int32, device=device
                    ).to(dtype)
                    vval = torch.full(
                        (dim,), -base, dtype=torch.int32, device=device
                    ).to(dtype)
                    write_token_to_block_storage(block_storage, bid, slot, kval, vval)

            # temp slots
            for slot in range(tail):
                gpos = temp_write_cursor
                tb_idx = gpos // block_size
                tb_off = gpos % block_size
                assert tb_idx < max_temp_blocks

                cluster_temp_kv_pos[h, cid, slot, 0] = tb_idx
                cluster_temp_kv_pos[h, cid, slot, 1] = tb_off

                src_bid = int(temp_block_ids[tb_idx].item())
                base = h * 100000 + cid * 10000 + 9000 + slot
                kval = torch.full(
                    (dim,), base, dtype=torch.int32, device=device
                ).to(dtype)
                vval = torch.full(
                    (dim,), -base, dtype=torch.int32, device=device
                ).to(dtype)
                write_token_to_block_storage(block_storage, src_bid, tb_off, kval, vval)

                temp_write_cursor += 1

    top_clusters = make_unique_top_clusters(
        NQ=NQ,
        Hq=Hq,
        nprobe=nprobe,
        C=C,
        device=device,
    )

    block_storage_before = block_storage.detach().cpu().clone()

    out_block_table, out_bt_len, out_seqused_k, used_free_block_count = \
        build_sparse_block_table(
            top_clusters,
            cluster_compact_block_ids,
            cluster_temp_kv_pos,
            cluster_total_kv_counts,
            temp_block_ids,
            block_storage,
            free_block_ids,
            max_bt_len,
        )

    torch.cuda.synchronize(device)
    block_storage_after = block_storage.detach().cpu().clone()

    # 转 CPU 方便 reference
    top_clusters_cpu = top_clusters.cpu()
    cluster_compact_block_ids_cpu = cluster_compact_block_ids.cpu()
    cluster_temp_kv_pos_cpu = cluster_temp_kv_pos.cpu()
    cluster_total_kv_counts_cpu = cluster_total_kv_counts.cpu()
    temp_block_ids_cpu = temp_block_ids.cpu()
    free_block_ids_cpu = free_block_ids.cpu()
    out_block_table_cpu = out_block_table.cpu()
    out_bt_len_cpu = out_bt_len.cpu()
    out_seqused_k_cpu = out_seqused_k.cpu()

    total_ref_used_free = 0

    for q in range(NQ):
        for hq in range(Hq):
            ref_compact_prefix, ref_seqused_k, ref_temp_stream, free_blocks_needed = \
                build_reference_for_one_row(
                    q=q,
                    hq=hq,
                    top_clusters=top_clusters_cpu,
                    cluster_compact_block_ids=cluster_compact_block_ids_cpu,
                    cluster_temp_kv_pos=cluster_temp_kv_pos_cpu,
                    cluster_total_kv_counts=cluster_total_kv_counts_cpu,
                    temp_block_ids=temp_block_ids_cpu,
                    block_storage_before=block_storage_before,
                )

            got_bt_len = int(out_bt_len_cpu[q, hq].item())
            got_seqused_k = int(out_seqused_k_cpu[q, hq].item())
            got_row = out_block_table_cpu[q, hq, :got_bt_len].tolist()

            expected_bt_len = len(ref_compact_prefix) + free_blocks_needed

            assert got_bt_len == expected_bt_len, (
                f"[case {case_id}] bt_len mismatch at (q={q}, hq={hq})\n"
                f"expected={expected_bt_len}\n"
                f"got={got_bt_len}"
            )

            assert got_seqused_k == ref_seqused_k, (
                f"[case {case_id}] seqused_k mismatch at (q={q}, hq={hq})\n"
                f"expected={ref_seqused_k}\n"
                f"got={got_seqused_k}"
            )

            got_compact_prefix = got_row[:len(ref_compact_prefix)]
            assert got_compact_prefix == ref_compact_prefix, (
                f"[case {case_id}] compact prefix mismatch at (q={q}, hq={hq})\n"
                f"expected={ref_compact_prefix}\n"
                f"got={got_compact_prefix}"
            )

            check_packed_temp_blocks_for_one_row(
                q=q,
                hq=hq,
                out_block_table=out_block_table_cpu,
                out_bt_len=out_bt_len_cpu,
                compact_prefix_len=len(ref_compact_prefix),
                ref_temp_stream_hashable=ref_temp_stream,
                free_blocks_needed=free_blocks_needed,
                block_storage_after=block_storage_after,
                block_size=block_size,
                free_block_ids=free_block_ids_cpu,
            )

            total_ref_used_free += free_blocks_needed

    got_used_free = int(used_free_block_count.cpu()[0].item())
    assert got_used_free == total_ref_used_free, (
        f"[case {case_id}] used_free_block_count mismatch\n"
        f"expected={total_ref_used_free}\n"
        f"got={got_used_free}"
    )

    print(
        f"[case {case_id:04d}] PASS | "
        f"NQ={NQ}, Hkv={Hkv}, Hq={Hq}, C={C}, nprobe={nprobe}, "
        f"block_size={block_size}, dim={dim}, dtype={dtype}, "
        f"used_free={got_used_free}"
    )


def run_random_tests(
    num_cases: int = 100,
    seed: int = 1234,
    device: str = "cuda",
):
    assert torch.cuda.is_available(), "CUDA is required"
    dev = torch.device(device)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for i in range(num_cases):
        run_one_case(i, dev)

    print(f"\nALL {num_cases} build_sparse_block_table TESTS PASSED.")


if __name__ == "__main__":
    run_random_tests(
        num_cases=100,
        seed=1234,
        device="cuda",
    )