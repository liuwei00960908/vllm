import random
from collections import Counter
from typing import Dict, Tuple

import torch
from vllm._custom_ops import append_kv_to_clusters


def make_unique_free_block_ids(
    total_blocks: int,
    excluded: set,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    candidates = [x for x in range(total_blocks) if x not in excluded]
    assert len(candidates) >= count, "not enough unique free blocks"
    picked = random.sample(candidates, count)
    return torch.tensor(picked, dtype=torch.int32, device=device)


def random_int_float_tensor(shape, low, high, device, dtype=torch.float32):
    # 先用 int 生成，再转 float，方便精确做集合校验
    x = torch.randint(low, high + 1, shape, device=device, dtype=torch.int32)
    return x.to(dtype)


def kv_token_to_hashable(k: torch.Tensor, v: torch.Tensor) -> Tuple:
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
    reference[(h, cid)] = Counter of ((k_vec), (v_vec))
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


def add_reference_multiset_inplace(
    ref_acc: Dict[Tuple[int, int], Counter],
    key: torch.Tensor,
    value: torch.Tensor,
    label: torch.Tensor,
    Hkv: int,
    C: int,
) -> None:
    delta = build_reference_multiset(key, value, label, Hkv, C)
    for k, ctr in delta.items():
        ref_acc[k].update(ctr)


def compute_expected_counts_from_ref(
    ref_acc: Dict[Tuple[int, int], Counter],
    Hkv: int,
    C: int,
) -> torch.Tensor:
    out = torch.zeros((Hkv, C), dtype=torch.int32)
    for h in range(Hkv):
        for cid in range(C):
            out[h, cid] = sum(ref_acc[(h, cid)].values())
    return out


def decode_kernel_output_multiset_with_temp_ids(
    block_storage: torch.Tensor,              # [2, total_blocks, block_size, dim]
    cluster_compact_block_ids: torch.Tensor,  # [Hkv, C, maxB]
    cluster_temp_kv_pos: torch.Tensor,        # [Hkv, C, block_size, 2]
    cluster_total_kv_counts: torch.Tensor,    # [Hkv, C]
    temp_block_ids: torch.Tensor,             # [max_temp_blocks]
) -> Dict[Tuple[int, int], Counter]:
    """
    从 kernel 输出状态中解码每个 cluster 的 KV 多重集合。
    不关心 cluster 内部顺序。
    """
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

            assert full_blocks <= maxB, (
                f"full_blocks={full_blocks} exceeds max_cluster_blocks={maxB} "
                f"at h={h}, cid={cid}, total_cnt={total_cnt}, block_size={block_size}"
            )

            # compact 部分
            for b in range(full_blocks):
                bid = int(compact_ids[h, cid, b].item())
                assert bid >= 0, (
                    f"missing compact block id for h={h}, cid={cid}, block={b}"
                )
                for slot in range(block_size):
                    k = bs[0, bid, slot]
                    v = bs[1, bid, slot]
                    item = kv_token_to_hashable(k, v)
                    out[(h, cid)][item] += 1

            # temp 部分
            for slot in range(temp_cnt):
                tb_idx = int(temp_pos[h, cid, slot, 0].item())
                tb_off = int(temp_pos[h, cid, slot, 1].item())
                assert tb_idx >= 0, (
                    f"missing temp pos for h={h}, cid={cid}, slot={slot}"
                )
                assert 0 <= tb_idx < temp_ids.numel()
                bid = int(temp_ids[tb_idx].item())
                assert bid >= 0
                assert 0 <= tb_off < block_size
                k = bs[0, bid, tb_off]
                v = bs[1, bid, tb_off]
                item = kv_token_to_hashable(k, v)
                out[(h, cid)][item] += 1

    return out


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


def check_each_cluster_temp_lt_block_size(
    cluster_temp_kv_pos: torch.Tensor,
    cluster_total_kv_counts: torch.Tensor,
    Hkv: int,
    C: int,
    block_size: int,
):
    """
    校验每个 cluster 在 temp 中当前有效残留数严格小于 block_size。
    同时要求 [0, temp_cnt) 这些 slot 都必须是有效的 temp 映射。
    """
    temp_pos = cluster_temp_kv_pos.cpu()
    total_counts = cluster_total_kv_counts.cpu()

    for h in range(Hkv):
        for cid in range(C):
            temp_cnt = int(total_counts[h, cid].item()) % block_size
            assert 0 <= temp_cnt < block_size, (
                f"temp_cnt out of range at h={h}, cid={cid}, temp_cnt={temp_cnt}, block_size={block_size}"
            )

            for slot in range(temp_cnt):
                tb_idx = int(temp_pos[h, cid, slot, 0].item())
                tb_off = int(temp_pos[h, cid, slot, 1].item())
                assert tb_idx >= 0, (
                    f"expected valid temp slot but got invalid at h={h}, cid={cid}, slot={slot}"
                )
                assert 0 <= tb_off < block_size, (
                    f"temp tb_off out of range at h={h}, cid={cid}, slot={slot}, tb_off={tb_off}"
                )


def check_compact_block_count_consistency(
    cluster_compact_block_ids: torch.Tensor,
    cluster_total_kv_counts: torch.Tensor,
    Hkv: int,
    C: int,
    block_size: int,
    *,
    require_tail_minus_one: bool = False,
):
    """
    校验每个 cluster 的 compact block 数应等于 total_cnt // block_size。

    检查：
    1. 前 expected_full_blocks 个 compact block id 必须有效
    2. 同一 cluster 内这些 id 不重复
    3. 如果 require_tail_minus_one=True，则后面的项必须全是 -1
    """
    compact_ids = cluster_compact_block_ids.cpu()
    total_counts = cluster_total_kv_counts.cpu()

    _, _, maxB = compact_ids.shape

    for h in range(Hkv):
        for cid in range(C):
            total_cnt = int(total_counts[h, cid].item())
            expected_full_blocks = total_cnt // block_size

            assert 0 <= expected_full_blocks <= maxB, (
                f"expected_full_blocks out of range at h={h}, cid={cid}, "
                f"expected_full_blocks={expected_full_blocks}, maxB={maxB}, total_cnt={total_cnt}"
            )

            local_ids = []
            for b in range(expected_full_blocks):
                bid = int(compact_ids[h, cid, b].item())
                assert bid >= 0, (
                    f"missing compact block id at h={h}, cid={cid}, block={b}, "
                    f"expected_full_blocks={expected_full_blocks}, total_cnt={total_cnt}"
                )
                local_ids.append(bid)

            assert len(local_ids) == expected_full_blocks, (
                f"compact block count mismatch at h={h}, cid={cid}: "
                f"len(local_ids)={len(local_ids)}, expected_full_blocks={expected_full_blocks}"
            )

            assert len(local_ids) == len(set(local_ids)), (
                f"duplicate compact block ids within cluster at h={h}, cid={cid}: {local_ids}"
            )

            if require_tail_minus_one:
                for b in range(expected_full_blocks, maxB):
                    bid = int(compact_ids[h, cid, b].item())
                    assert bid == -1, (
                        f"expected trailing compact block ids to be -1 at "
                        f"h={h}, cid={cid}, block={b}, got {bid}"
                    )


def check_global_compact_block_uniqueness(
    cluster_compact_block_ids: torch.Tensor,
    cluster_total_kv_counts: torch.Tensor,
    Hkv: int,
    C: int,
    block_size: int,
):
    """
    校验所有当前有效 compact block id 在全局范围内不重复。
    """
    compact_ids = cluster_compact_block_ids.cpu()
    total_counts = cluster_total_kv_counts.cpu()

    used_compact_ids = []
    for h in range(Hkv):
        for cid in range(C):
            need_blocks = int(total_counts[h, cid].item()) // block_size
            for b in range(need_blocks):
                bid = int(compact_ids[h, cid, b].item())
                assert bid >= 0
                used_compact_ids.append(bid)

    assert len(used_compact_ids) == len(set(used_compact_ids)), (
        f"compact block ids are not globally unique: {used_compact_ids}"
    )


def run_one_multi_round_case(case_id: int, device: torch.device):
    # 主场景
    Hkv = 2
    C = random.choice([2, 3, 4, 5])
    block_size = 8
    dim = 64

    num_rounds = random.choice([3, 4, 5, 6, 8])
    round_nqs = [random.choice([1, 3, 7, 8, 15, 16, 31, 32, 63, 64]) for _ in range(num_rounds)]

    # 保证单个 cluster 即使极端集中也不会爆 max_cluster_blocks
    max_tokens_one_head = sum(round_nqs)
    max_cluster_blocks = (max_tokens_one_head + block_size - 1) // block_size + 4

    max_temp_blocks = 256
    total_possible_compact = Hkv * C * max_cluster_blocks
    total_blocks = total_possible_compact + max_temp_blocks + 64

    dtype = random.choice([torch.float32, torch.float16, torch.bfloat16])

    # 初始化一次，整轮 case 复用状态
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

    # free_block_ids unique 且不和 temp_block_ids 重叠
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

    # Python reference 累积状态
    ref_acc: Dict[Tuple[int, int], Counter] = {
        (h, cid): Counter() for h in range(Hkv) for cid in range(C)
    }

    consumed_free_blocks_so_far = 0

    for r, Nq in enumerate(round_nqs):
        key = random_int_float_tensor((Nq, Hkv, dim), low=-50, high=50, device=device, dtype=dtype)
        value = random_int_float_tensor((Nq, Hkv, dim), low=-50, high=50, device=device, dtype=dtype)
        label = torch.randint(0, C, (Nq, Hkv), dtype=torch.int32, device=device)

        # 累加 reference
        add_reference_multiset_inplace(ref_acc, key, value, label, Hkv, C)

        # 记录本轮前的 counts，用于校验本轮新增 compact blocks 数
        before_counts = cluster_total_kv_counts.cpu().clone()

        # 多轮 append：free_block_ids 从未消费部分开始传入
        round_used_free_blocks = append_kv_to_clusters(
            block_storage,
            cluster_compact_block_ids,
            cluster_temp_kv_pos,
            cluster_total_kv_counts,
            temp_block_ids,
            temp_block_kv_counts,
            temp_block_kv_owner,
            free_block_ids[consumed_free_blocks_so_far:],
            key,
            value,
            label,
        )
        torch.cuda.synchronize(device)

        got_round_used = int(round_used_free_blocks.cpu()[0].item())
        consumed_free_blocks_so_far += got_round_used

        # decode 当前累计状态
        got = decode_kernel_output_multiset_with_temp_ids(
            block_storage=block_storage,
            cluster_compact_block_ids=cluster_compact_block_ids,
            cluster_temp_kv_pos=cluster_temp_kv_pos,
            cluster_total_kv_counts=cluster_total_kv_counts,
            temp_block_ids=temp_block_ids,
        )

        # 1) multiset 校验：累计 cluster 内容正确
        for h in range(Hkv):
            for cid in range(C):
                assert got[(h, cid)] == ref_acc[(h, cid)], (
                    f"[case {case_id}, round {r}] cluster mismatch at (h={h}, cid={cid})\n"
                    f"expected={ref_acc[(h, cid)]}\n"
                    f"got={got[(h, cid)]}"
                )

        # 2) cluster_total_kv_counts 校验
        expected_counts = compute_expected_counts_from_ref(ref_acc, Hkv, C)
        assert torch.equal(cluster_total_kv_counts.cpu(), expected_counts), (
            f"[case {case_id}, round {r}] cluster_total_kv_counts mismatch\n"
            f"expected={expected_counts}\n"
            f"got={cluster_total_kv_counts.cpu()}"
        )

        # 3) temp_block_kv_counts 校验
        expected_temp = int(
            sum(int(expected_counts[h, cid]) % block_size for h in range(Hkv) for cid in range(C))
        )
        got_temp = int(temp_block_kv_counts.cpu()[0].item())
        assert got_temp == expected_temp, (
            f"[case {case_id}, round {r}] temp_block_kv_counts mismatch: "
            f"expected {expected_temp}, got {got_temp}"
        )

        # 4) 本轮 used_free_blocks 校验
        after_counts = cluster_total_kv_counts.cpu().clone()
        expected_round_used = 0
        for h in range(Hkv):
            for cid in range(C):
                before_full = int(before_counts[h, cid].item()) // block_size
                after_full = int(after_counts[h, cid].item()) // block_size
                expected_round_used += (after_full - before_full)

        assert got_round_used == expected_round_used, (
            f"[case {case_id}, round {r}] used_free_blocks mismatch: "
            f"expected {expected_round_used}, got {got_round_used}"
        )

        # 5) 每个 cluster 的 compact block 数 == total_cnt // block_size
        check_compact_block_count_consistency(
            cluster_compact_block_ids=cluster_compact_block_ids,
            cluster_total_kv_counts=cluster_total_kv_counts,
            Hkv=Hkv,
            C=C,
            block_size=block_size,
            require_tail_minus_one=False,  # 如你确定尾部必须始终为 -1，可改 True
        )

        # 6) 全局 compact block id 唯一
        check_global_compact_block_uniqueness(
            cluster_compact_block_ids=cluster_compact_block_ids,
            cluster_total_kv_counts=cluster_total_kv_counts,
            Hkv=Hkv,
            C=C,
            block_size=block_size,
        )

        # 7) free_block_ids 输入自身 unique
        free_ids_cpu = free_block_ids.cpu().tolist()
        assert len(free_ids_cpu) == len(set(free_ids_cpu)), (
            f"[case {case_id}, round {r}] free_block_ids not unique"
        )

        # 8) temp owner consistency
        check_temp_owner_consistency(
            cluster_temp_kv_pos=cluster_temp_kv_pos,
            cluster_total_kv_counts=cluster_total_kv_counts,
            temp_block_kv_owner=temp_block_kv_owner,
            Hkv=Hkv,
            C=C,
            block_size=block_size,
        )

        # 9) 每个 cluster 在 temp 中的残留数严格小于 block_size
        check_each_cluster_temp_lt_block_size(
            cluster_temp_kv_pos=cluster_temp_kv_pos,
            cluster_total_kv_counts=cluster_total_kv_counts,
            Hkv=Hkv,
            C=C,
            block_size=block_size,
        )

        print(
            f"[case {case_id:04d}, round {r:02d}] PASS | "
            f"Nq={Nq}, Hkv={Hkv}, C={C}, block_size={block_size}, dim={dim}, "
            f"round_used_free={got_round_used}, total_temp_kv={got_temp}, dtype={dtype}"
        )


def run_random_tests(
    num_cases: int = 50,
    seed: int = 1234,
    device: str = "cuda",
):
    assert torch.cuda.is_available(), "CUDA is required"
    dev = torch.device(device)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for i in range(num_cases):
        run_one_multi_round_case(i, dev)

    print(f"\nALL {num_cases} MULTI-ROUND RANDOM TESTS PASSED.")


if __name__ == "__main__":
    run_random_tests(
        num_cases=50,
        seed=1234,
        device="cuda",
    )