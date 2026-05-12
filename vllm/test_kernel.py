import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import torch


def append_kv_to_clusters(
    block_storage: torch.Tensor,
    cluster_compact_block_ids: torch.Tensor,
    cluster_temp_kv_pos: torch.Tensor,
    cluster_total_kv_counts: torch.Tensor,
    temp_block_ids: torch.Tensor,
    temp_block_kv_counts: torch.Tensor,
    temp_block_kv_owner: torch.Tensor,
    free_block_ids: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.append_kv_to_clusters(
        block_storage,
        cluster_compact_block_ids,
        cluster_temp_kv_pos,
        cluster_total_kv_counts,
        temp_block_ids,
        temp_block_kv_counts,
        temp_block_kv_owner,
        free_block_ids,
        key,
        value,
        label,
    )


def make_unique_free_block_ids(total_blocks: int, excluded: set, count: int, device: torch.device) -> torch.Tensor:
    candidates = [x for x in range(total_blocks) if x not in excluded]
    assert len(candidates) >= count, "not enough unique free blocks"
    picked = random.sample(candidates, count)
    return torch.tensor(picked, dtype=torch.int32, device=device)


def random_int_float_tensor(shape, low, high, device, dtype=torch.float32):
    # 先用 int 生成，再转 float，方便精确做集合校验
    x = torch.randint(low, high + 1, shape, device=device, dtype=torch.int32)
    return x.to(dtype)


def kv_token_to_hashable(k: torch.Tensor, v: torch.Tensor) -> Tuple:
    # 转成 python tuple，便于做多重集合比较
    return (
        tuple(int(x) for x in k.tolist()),
        tuple(int(x) for x in v.tolist()),
    )


def build_reference_multiset(
    key: torch.Tensor,
    value: torch.Tensor,
    label: torch.Tensor,
    Hkv: int,
    C: int,
) -> Dict[Tuple[int, int], Counter]:
    """
    Python reference:
    reference[(h, cid)] = Counter of (k_vec, v_vec)
    """
    ref: Dict[Tuple[int, int], Counter] = {
        (h, cid): Counter() for h in range(Hkv) for cid in range(C)
    }

    key_cpu = key.cpu().to(torch.int32)
    value_cpu = value.cpu().to(torch.int32)
    label_cpu = label.cpu()

    Nq = key.shape[0]
    for q in range(Nq):
        for h in range(Hkv):
            cid = int(label_cpu[q, h].item())
            item = kv_token_to_hashable(key_cpu[q, h], value_cpu[q, h])
            ref[(h, cid)][item] += 1

    return ref


def decode_kernel_output_multiset(
    block_storage: torch.Tensor,              # [2, total_blocks, block_size, dim]
    cluster_compact_block_ids: torch.Tensor,  # [Hkv, C, maxB]
    cluster_temp_kv_pos: torch.Tensor,        # [Hkv, C, block_size, 2]
    cluster_total_kv_counts: torch.Tensor,    # [Hkv, C]
) -> Dict[Tuple[int, int], Counter]:
    """
    从 kernel 执行后的状态中解码每个 cluster 的 KV 多重集合。
    不关心 cluster 内部顺序。
    """
    bs = block_storage.cpu().to(torch.int32)
    compact_ids = cluster_compact_block_ids.cpu()
    temp_pos = cluster_temp_kv_pos.cpu()
    total_counts = cluster_total_kv_counts.cpu()

    _, total_blocks, block_size, dim = block_storage.shape
    Hkv, C, maxB = cluster_compact_block_ids.shape

    out: Dict[Tuple[int, int], Counter] = {
        (h, cid): Counter() for h in range(Hkv) for cid in range(C)
    }

    for h in range(Hkv):
        for cid in range(C):
            total_cnt = int(total_counts[h, cid].item())
            full_blocks = total_cnt // block_size
            temp_cnt = total_cnt % block_size

            # 1) compact blocks
            for b in range(full_blocks):
                bid = int(compact_ids[h, cid, b].item())
                assert bid >= 0, f"missing compact block id for h={h}, cid={cid}, block={b}"
                for slot in range(block_size):
                    k = bs[0, bid, slot]
                    v = bs[1, bid, slot]
                    item = kv_token_to_hashable(k, v)
                    out[(h, cid)][item] += 1

            # 2) temp part
            for slot in range(temp_cnt):
                tb_idx = int(temp_pos[h, cid, slot, 0].item())
                tb_off = int(temp_pos[h, cid, slot, 1].item())
                assert tb_idx >= 0, f"missing temp pos for h={h}, cid={cid}, slot={slot}"
                bid = tb_idx  # 注意：这里只是先占位，下面会在调用处用 temp_block_ids 解码

    return out


def decode_kernel_output_multiset_with_temp_ids(
    block_storage: torch.Tensor,
    cluster_compact_block_ids: torch.Tensor,
    cluster_temp_kv_pos: torch.Tensor,
    cluster_total_kv_counts: torch.Tensor,
    temp_block_ids: torch.Tensor,
) -> Dict[Tuple[int, int], Counter]:
    bs = block_storage.cpu().to(torch.int32)
    compact_ids = cluster_compact_block_ids.cpu()
    temp_pos = cluster_temp_kv_pos.cpu()
    total_counts = cluster_total_kv_counts.cpu()
    temp_ids = temp_block_ids.cpu()

    _, total_blocks, block_size, dim = block_storage.shape
    Hkv, C, maxB = cluster_compact_block_ids.shape

    out: Dict[Tuple[int, int], Counter] = {
        (h, cid): Counter() for h in range(Hkv) for cid in range(C)
    }

    for h in range(Hkv):
        for cid in range(C):
            total_cnt = int(total_counts[h, cid].item())
            full_blocks = total_cnt // block_size
            temp_cnt = total_cnt % block_size

            # compact
            for b in range(full_blocks):
                bid = int(compact_ids[h, cid, b].item())
                assert bid >= 0, f"missing compact block id for h={h}, cid={cid}, block={b}"
                for slot in range(block_size):
                    k = bs[0, bid, slot]
                    v = bs[1, bid, slot]
                    item = kv_token_to_hashable(k, v)
                    out[(h, cid)][item] += 1

            # temp
            for slot in range(temp_cnt):
                tb_idx = int(temp_pos[h, cid, slot, 0].item())
                tb_off = int(temp_pos[h, cid, slot, 1].item())
                assert tb_idx >= 0, f"missing temp pos for h={h}, cid={cid}, slot={slot}"
                assert 0 <= tb_idx < temp_ids.numel()
                bid = int(temp_ids[tb_idx].item())
                assert bid >= 0
                assert 0 <= tb_off < block_size
                k = bs[0, bid, tb_off]
                v = bs[1, bid, tb_off]
                item = kv_token_to_hashable(k, v)
                out[(h, cid)][item] += 1

    return out


def check_no_full_temp_cluster(cluster_total_kv_counts: torch.Tensor, block_size: int):
    counts = cluster_total_kv_counts.cpu()
    full_mask = (counts > 0) & ((counts % block_size) == 0)
    # 允许 total_cnt == 0；但不允许返回时有 temp 满块未 compact
    # 在当前语义下，只要 count % block_size == 0 且 count > 0，说明该 cluster 应该没有 temp 残留满块
    # 这里单靠 total_counts 不能判断 compact blocks 是否存在，因此不能直接 assert false。
    # 真正判断是：temp 部分长度必须是 total_cnt % block_size == 0 => temp 为空
    # 所以这里只作为信息输出，不做 hard fail。
    return full_mask


def check_temp_owner_consistency(
    cluster_temp_kv_pos: torch.Tensor,
    cluster_total_kv_counts: torch.Tensor,
    temp_block_kv_owner: torch.Tensor,
    Hkv: int,
    C: int,
    block_size: int,
):
    """
    对所有 temp 残留项检查：
    cluster_temp_kv_pos[h,cid,slot] 指向的 gpos owner 是否是 (h*C+cid, slot)
    """
    temp_pos = cluster_temp_kv_pos.cpu()
    total_counts = cluster_total_kv_counts.cpu()
    owner = temp_block_kv_owner.cpu()

    for h in range(Hkv):
        for cid in range(C):
            temp_cnt = int(total_counts[h, cid].item()) % block_size
            for slot in range(temp_cnt):
                tb_idx = int(temp_pos[h, cid, slot, 0].item())
                tb_off = int(temp_pos[h, cid, slot, 1].item())
                assert tb_idx >= 0
                assert 0 <= tb_off < block_size
                gpos = tb_idx * block_size + tb_off
                owner_cluster = int(owner[gpos, 0].item())
                owner_slot = int(owner[gpos, 1].item())
                assert owner_cluster == h * C + cid, (
                    f"owner_cluster mismatch: expected {(h, cid)}, got {owner_cluster}"
                )
                assert owner_slot == slot, (
                    f"owner_slot mismatch: expected slot={slot}, got {owner_slot}"
                )


def run_one_random_case(case_id: int, device: torch.device):
    # 你说的主场景 Hkv 很小，这里重点测小 Hkv，但也覆盖一些别的
    Hkv = random.choice([1, 2, 2, 2, 4])
    C = random.choice([2, 3, 4, 5])
    block_size = random.choice([8, 16, 32, 64])
    dim = random.choice([16, 32, 64, 128])
    Nq = random.choice([1, 2, 3, 4, 8, 16, 24, 32])

    max_cluster_blocks = random.choice([4, 8, 12, 16])
    max_temp_blocks = random.choice([8, 12, 16, 24, 32])

    # 估计需要的总 block 数
    total_possible_compact = Hkv * C * max_cluster_blocks
    total_blocks = total_possible_compact + max_temp_blocks + 64

    dtype = random.choice([torch.float32, torch.float16, torch.bfloat16])

    # 初始化 storage / metadata
    block_storage = torch.full(
        (2, total_blocks, block_size, dim),
        fill_value=-777,
        dtype=dtype,
        device=device,
    )

    cluster_compact_block_ids = torch.full(
        (Hkv, C, max_cluster_blocks),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    cluster_temp_kv_pos = torch.full(
        (Hkv, C, block_size, 2),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    cluster_total_kv_counts = torch.zeros(
        (Hkv, C),
        dtype=torch.int32,
        device=device,
    )

    # temp blocks 固定分配一批唯一 block id
    all_block_ids = list(range(total_blocks))
    temp_block_id_list = random.sample(all_block_ids, max_temp_blocks)
    temp_block_ids = torch.tensor(temp_block_id_list, dtype=torch.int32, device=device)

    temp_block_kv_counts = torch.zeros((1,), dtype=torch.int32, device=device)

    temp_block_kv_owner = torch.full(
        (max_temp_blocks * block_size, 2),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    # free_block_ids 要 unique，且不能与 temp_block_ids 重叠
    excluded = set(temp_block_id_list)
    max_free_blocks = min(
        len([x for x in all_block_ids if x not in excluded]),
        Hkv * C * max_cluster_blocks,
    )
    assert max_free_blocks > 0
    free_block_ids = make_unique_free_block_ids(
        total_blocks=total_blocks,
        excluded=excluded,
        count=max_free_blocks,
        device=device,
    )

    # 随机 key/value/label
    key = random_int_float_tensor((Nq, Hkv, dim), low=-50, high=50, device=device, dtype=dtype)
    value = random_int_float_tensor((Nq, Hkv, dim), low=-50, high=50, device=device, dtype=dtype)
    label = torch.randint(0, C, (Nq, Hkv), dtype=torch.int32, device=device)

    # Python reference
    ref = build_reference_multiset(key, value, label, Hkv, C)

    # 调 kernel
    used_free_blocks = append_kv_to_clusters(
        block_storage,
        cluster_compact_block_ids,
        cluster_temp_kv_pos,
        cluster_total_kv_counts,
        temp_block_ids,
        temp_block_kv_counts,
        temp_block_kv_owner,
        free_block_ids,
        key,
        value,
        label,
    )
    torch.cuda.synchronize(device)

    # decode kernel output
    got = decode_kernel_output_multiset_with_temp_ids(
        block_storage=block_storage,
        cluster_compact_block_ids=cluster_compact_block_ids,
        cluster_temp_kv_pos=cluster_temp_kv_pos,
        cluster_total_kv_counts=cluster_total_kv_counts,
        temp_block_ids=temp_block_ids,
    )

    # 1) 集合校验
    for h in range(Hkv):
        for cid in range(C):
            assert got[(h, cid)] == ref[(h, cid)], (
                f"[case {case_id}] cluster mismatch at (h={h}, cid={cid})\n"
                f"expected={ref[(h, cid)]}\n"
                f"got={got[(h, cid)]}"
            )

    # 2) total counts 校验
    expected_counts = torch.zeros((Hkv, C), dtype=torch.int32)
    label_cpu = label.cpu()
    for q in range(Nq):
        for h in range(Hkv):
            expected_counts[h, int(label_cpu[q, h].item())] += 1
    assert torch.equal(cluster_total_kv_counts.cpu(), expected_counts), (
        f"[case {case_id}] cluster_total_kv_counts mismatch\n"
        f"expected={expected_counts}\n"
        f"got={cluster_total_kv_counts.cpu()}"
    )

    # 3) temp kv 总数校验
    expected_temp = int(sum(int(expected_counts[h, cid]) % block_size for h in range(Hkv) for cid in range(C)))
    got_temp = int(temp_block_kv_counts.cpu()[0].item())
    assert got_temp == expected_temp, (
        f"[case {case_id}] temp_block_kv_counts mismatch: expected {expected_temp}, got {got_temp}"
    )

    # 4) used_free_blocks 校验
    expected_used_free = int(sum(int(expected_counts[h, cid]) // block_size for h in range(Hkv) for cid in range(C)))
    got_used_free = int(used_free_blocks.cpu()[0].item())
    assert got_used_free == expected_used_free, (
        f"[case {case_id}] used_free_blocks mismatch: expected {expected_used_free}, got {got_used_free}"
    )

    # 5) compact block ids 合法性校验
    compact_ids_cpu = cluster_compact_block_ids.cpu()
    used_compact_ids = []
    for h in range(Hkv):
        for cid in range(C):
            need_blocks = int(expected_counts[h, cid]) // block_size
            for b in range(need_blocks):
                bid = int(compact_ids_cpu[h, cid, b].item())
                assert bid >= 0, f"[case {case_id}] missing compact block id at {(h, cid, b)}"
                used_compact_ids.append(bid)

            for b in range(need_blocks, max_cluster_blocks):
                # 后面的 block id 可以还是 -1；如果实现复用已存在 block，也不强制清掉
                pass

    # compact block ids 应该 unique（这版语义下不同 compact block 不应共用同一个物理 block）
    assert len(used_compact_ids) == len(set(used_compact_ids)), (
        f"[case {case_id}] compact block ids are not unique: {used_compact_ids}"
    )

    # 6) free_block_ids 输入自身 unique 校验
    free_ids_cpu = free_block_ids.cpu().tolist()
    assert len(free_ids_cpu) == len(set(free_ids_cpu)), f"[case {case_id}] free_block_ids not unique"

    # 7) temp owner consistency
    check_temp_owner_consistency(
        cluster_temp_kv_pos=cluster_temp_kv_pos,
        cluster_total_kv_counts=cluster_total_kv_counts,
        temp_block_kv_owner=temp_block_kv_owner,
        Hkv=Hkv,
        C=C,
        block_size=block_size,
    )

    print(
        f"[case {case_id:04d}] PASS | "
        f"Nq={Nq}, Hkv={Hkv}, C={C}, block_size={block_size}, dim={dim}, "
        f"used_free={got_used_free}, temp_kv={got_temp}, dtype={dtype}"
    )


def run_random_tests(
    num_cases: int = 200,
    seed: int = 1234,
    device: str = "cuda",
):
    assert torch.cuda.is_available(), "CUDA is required"
    dev = torch.device(device)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for i in range(num_cases):
        run_one_random_case(i, dev)

    print(f"\nALL {num_cases} RANDOM TESTS PASSED.")


if __name__ == "__main__":
    run_random_tests(
        num_cases=200,
        seed=1234,
        device="cuda",
    )