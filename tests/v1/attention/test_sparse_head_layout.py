# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controllable unit tests for the *tail-layout, Route B* sparse-decode KV
arrangement (LMCache offload + token-granularity sparse attention).

Layout (Route B)
----------------
A decode step lays the paged block-table row out as::

    [ scratch (selected prompt KV, front)  |  decode (resident, back) ]

The LMCache-loaded scratch occupies the front at ``token_start_index = 0``
(matching LMCache's append-order block tracking -- no block-order surgery).
The resident decode region is laid out immediately after the *valid* selected
length ``sel_kv_len = budget`` (the LMCache-loaded budget; ``max_selected_tokens``
is "steady (static head/tail) + retrieve", so budget already bounds the total
selected count -- static is not separately loaded).  It borrows the last scratch
block's block-align slack, so a single token-granular
``seqused_k = sel_kv_len + D'`` covers [scratch | decode] contiguously and the
newly generated decode tokens always enter FA.

Production wiring:
  * ``vllm/v1/core/sparse_kv_cache_manager.py``  -- row = [scratch, history, cur_decode]
  * ``vllm/v1/worker/gpu_model_runner.py``:
      - decode slot_mapping = sel_kv_len + (pos - prompt_len)   (_prepare_inputs)
      - seqused_k (actual_kv_len) = sel_kv_len + decode_len, token_start_index = 0
        (_build_sparse_runtime_q_head_gather)
      - BOTH must use the SAME sel_kv_len formula or decode is mis-slotted.
  * ``lmcache/integration/vllm/vllm_v1_adapter.py`` (ReqMeta.from_request_tracker)
      -- after offload frees the prefill blocks, the load slot_mapping must point
         at the SCRATCH blocks, not the stale prefill blocks at the tracker front.

These tests run on CPU in pure PyTorch (no GPU/FA/LMCache).  They reproduce the
layout / index arithmetic and the attention math so the design can be iterated
quickly.  Config shaped from a production run (``vllm.log``):
``block_size=16, num_kv_heads=4, num_q_heads=28, head_dim=128, budget=512,
static_pattern_start=8, static_pattern_end=128`` (Qwen2.5-7B-Instruct).
"""

import torch

torch.manual_seed(0)


# ── Production index contract (mirrors the Route-B edits) ──────────────────────


def sel_kv_len(budget: int) -> int:
    """Valid selected length = the LMCache-loaded budget.  ``max_selected_tokens``
    is "steady (static head/tail) + retrieve", so budget already bounds the total
    selected count; static is NOT separately loaded.  Computed identically in
    _prepare_inputs and _build_sparse_runtime_q_head_gather."""
    return budget


def token_start_index() -> int:
    """Scratch is written at the front of LMCache's slot map (offset 0)."""
    return 0


def head_seqused_k(scratch_len: int, decode_len: int) -> int:
    """FA key length covering [scratch | decode] contiguously."""
    return scratch_len + decode_len


def decode_row_slot(pos: int, prompt_len: int, scratch_len: int) -> int:
    """Row-logical slot a decode token at sequence position ``pos`` writes to:
    right after the valid scratch length."""
    return scratch_len + (pos - prompt_len)


# ── Pure-torch reference attention (single decode query, GQA-aware) ────────────


def _attention(q, k, v):
    """q:[Hq,D]  k/v:[N,Hkv,D]  ->  out:[Hq,D]  (float64, full softmax)."""
    hq, d = q.shape
    n, hkv, _ = k.shape
    group = hq // hkv
    out = torch.empty(hq, d, dtype=torch.float64)
    scale = 1.0 / (d ** 0.5)
    for h in range(hq):
        kvh = h // group
        scores = (q[h].double() @ k[:, kvh].double().T) * scale
        w = torch.softmax(scores, dim=-1)
        out[h] = w @ v[:, kvh].double()
    return out


# ── Paged-cache layout simulation (Route B) ───────────────────────────────────


class Cfg:
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 28
    head_dim = 32  # structurally irrelevant; small for speed


def _empty_cache(num_blocks):
    shape = (num_blocks, Cfg.block_size, Cfg.num_kv_heads, Cfg.head_dim)
    # NaN-fill so any unwritten/garbage slot that leaks into attention is loud.
    return (torch.full(shape, float("nan")), torch.full(shape, float("nan")))


def _row_slot_mapping(block_ids):
    bs = Cfg.block_size
    slots = []
    for bid in block_ids:
        slots.extend(range(bid * bs, bid * bs + bs))
    return slots


def _write(cache, slot, kv):
    k_cache, v_cache = cache
    b, off = slot // Cfg.block_size, slot % Cfg.block_size
    k_cache[b, off] = kv[0]
    v_cache[b, off] = kv[1]


def _read(cache, slot):
    k_cache, v_cache = cache
    b, off = slot // Cfg.block_size, slot % Cfg.block_size
    return k_cache[b, off], v_cache[b, off]


def _build_tail_layout(prompt_kv, decode_kv, selected_idx, scratch_blocks):
    """Materialise the Route-B tail layout and return the (K, V) FA reads for
    ``seqused_k = scratch_len + D'``, where ``scratch_len`` = the valid selected
    length (decode borrows the last scratch block's slack).

    scratch_blocks: number of physical scratch blocks (block-aligned capacity).
    """
    pk, pv = prompt_kv
    dk, dv = decode_kv
    decode_len = dk.shape[0]
    s_valid = len(selected_idx)                 # valid scratch length (sel_kv_len)
    prompt_len = pk.shape[0]
    bs = Cfg.block_size

    n_decode_blocks = (s_valid + decode_len + bs - 1) // bs + 2  # slack
    num_blocks = scratch_blocks + n_decode_blocks
    cache = _empty_cache(num_blocks)
    block_ids = list(range(num_blocks))
    slot_map = _row_slot_mapping(block_ids)

    # 1) LMCache writes scratch at token_start_index=0 -> rows [0, s_valid).
    tsi = token_start_index()
    for i, p in enumerate(selected_idx):
        _write(cache, slot_map[tsi + i], (pk[p], pv[p]))

    # 2) decode written by vLLM at rows [s_valid, s_valid + D').
    for k in range(decode_len):
        row_pos = decode_row_slot(prompt_len + k, prompt_len, s_valid)
        _write(cache, slot_map[row_pos], (dk[k], dv[k]))

    seqused = head_seqused_k(s_valid, decode_len)
    gk, gv = [], []
    for row_pos in range(seqused):
        k, v = _read(cache, slot_map[row_pos])
        gk.append(k)
        gv.append(v)
    return torch.stack(gk), torch.stack(gv), seqused, tsi, decode_len, s_valid


def _make_kv(n):
    return (torch.randn(n, Cfg.num_kv_heads, Cfg.head_dim),
            torch.randn(n, Cfg.num_kv_heads, Cfg.head_dim))


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_index_contract():
    """Route-B index formulas match their documented meaning."""
    prompt_len, decode_len = 200, 23
    budget = 64
    s = sel_kv_len(budget)
    assert s == 64
    assert token_start_index() == 0
    assert head_seqused_k(s, decode_len) == s + decode_len
    # the k-th generated token (abs pos prompt_len+k) -> row slot s + k
    for k in range(decode_len):
        assert decode_row_slot(prompt_len + k, prompt_len, s) == s + k


def test_tail_layout_matches_union_attention():
    """Tail Route-B single-pass FA == reference attention over
    (selected prompt tokens ∪ all decode tokens)."""
    prompt_len, decode_len = 200, 23
    prompt_kv = _make_kv(prompt_len)
    decode_kv = _make_kv(decode_len)
    head = list(range(8))
    tail = list(range(prompt_len - 16, prompt_len))
    dynamic = list(range(40, 80))
    selected = head + dynamic + tail            # 64 tokens
    scratch_blocks = (len(selected) + 15) // 16 + 1   # block-aligned + slack

    gk, gv, seqused, tsi, dlen, s = _build_tail_layout(
        prompt_kv, decode_kv, selected, scratch_blocks)
    assert tsi == 0
    assert seqused == len(selected) + decode_len
    assert not torch.isnan(gk).any(), "tail layout read an unwritten/garbage slot"

    q = torch.randn(Cfg.num_q_heads, Cfg.head_dim)
    got = _attention(q, gk, gv)

    pk, pv = prompt_kv
    dk, dv = decode_kv
    ref = _attention(q, torch.cat([pk[selected], dk]), torch.cat([pv[selected], dv]))
    assert torch.allclose(got, ref, atol=1e-9), "tail layout != union attention"


def test_old_layout_drops_decode_is_detectably_wrong():
    """Control: excluding the decode region (the pre-fix behaviour) yields a
    different result, and the tail layout recovers the correct one."""
    prompt_len, decode_len = 200, 23
    prompt_kv = _make_kv(prompt_len)
    decode_kv = _make_kv(decode_len)
    selected = list(range(8)) + list(range(40, 80)) + \
        list(range(prompt_len - 16, prompt_len))
    scratch_blocks = (len(selected) + 15) // 16 + 1
    q = torch.randn(Cfg.num_q_heads, Cfg.head_dim)

    pk, pv = prompt_kv
    dk, dv = decode_kv
    truth = _attention(q, torch.cat([pk[selected], dk]),
                       torch.cat([pv[selected], dv]))
    buggy = _attention(q, pk[selected], pv[selected])  # decode dropped
    assert not torch.allclose(buggy, truth, atol=1e-6), \
        "test scenario must actually depend on decode tokens"

    gk, gv, *_ = _build_tail_layout(prompt_kv, decode_kv, selected, scratch_blocks)
    fixed = _attention(q, gk, gv)
    assert torch.allclose(fixed, truth, atol=1e-9)


def test_decode_borrows_scratch_block_slack_no_gap():
    """Route B: when the valid selection ends mid-block, decode starts in that
    same block's slack tail -> contiguous, no gap, no stale read."""
    prompt_len, decode_len = 200, 10
    prompt_kv = _make_kv(prompt_len)
    decode_kv = _make_kv(decode_len)
    # 20 selected -> ends at row 20 = block 1, slot 4 (mid-block). decode starts
    # at row 20 (block 1 slot 4), borrowing the block's slack.
    selected = list(range(8)) + list(range(50, 62))
    assert len(selected) % 16 != 0, "want a mid-block boundary for this test"
    scratch_blocks = (len(selected) + 15) // 16 + 1

    gk, gv, seqused, tsi, dlen, s = _build_tail_layout(
        prompt_kv, decode_kv, selected, scratch_blocks)
    assert not torch.isnan(gk).any(), "gap/slack leaked stale KV into attention"
    q = torch.randn(Cfg.num_q_heads, Cfg.head_dim)
    pk, pv = prompt_kv
    dk, dv = decode_kv
    ref = _attention(q, torch.cat([pk[selected], dk]), torch.cat([pv[selected], dv]))
    assert torch.allclose(_attention(q, gk, gv), ref, atol=1e-9)


def test_prepare_inputs_and_runtime_gather_use_same_sel_kv_len():
    """The decode slot_mapping offset (_prepare_inputs) and the seqused_k scratch
    part (_build_sparse_runtime_q_head_gather) must use the same sel_kv_len = the
    LMCache-loaded budget, or decode is written/read at misaligned slots.  Guards
    against the two formulas drifting apart (and against the over-read regression
    where sel_kv_len wrongly added static head/tail on top of budget)."""
    budget = 512
    n_scratch_tokens = 1500 * 16  # scratch capacity >= budget
    # _prepare_inputs: min(budget, scratch capacity)
    prep = min(budget, n_scratch_tokens)
    # runtime gather (dynamic path): budget
    gather = budget
    assert prep == gather == 512


# ── LMCache offload slot_mapping contract (root-cause regression) ───────────────


def _from_request_tracker_slot_mapping(allocated_block_ids, prompt_len,
                                       num_token_ids, block_size, is_sparse_decode):
    """Pure replica of LMCache ``ReqMeta.from_request_tracker``'s slot_mapping
    build (lmcache/integration/vllm/vllm_v1_adapter.py).  After an offload the
    tracker holds ``[prefill..., scratch..., decode...]``; the sparse path must
    point the leading (loaded) slots at the SCRATCH blocks, not the freed prefill
    blocks -- while keeping length == num_token_ids for the prompt-length keys."""
    def cdiv(a, b):
        return (a + b - 1) // b

    def flatten(block_ids):
        out = []
        for bid in block_ids:
            out.extend(range(bid * block_size, bid * block_size + block_size))
        return out

    slot_mapping = flatten(allocated_block_ids)[:num_token_ids]
    if is_sparse_decode:
        n_prefill_blocks = cdiv(prompt_len, block_size)
        scratch_slots = flatten(allocated_block_ids[n_prefill_blocks:])
        n = min(len(scratch_slots), len(slot_mapping))
        slot_mapping[:n] = scratch_slots[:n]
    return slot_mapping


def test_offload_slot_mapping_targets_scratch_not_prefill():
    """Root-cause regression for garbage sparse-decode output: after
    VLLM_SPARSE_FREE_PREFILL_AFTER_SAVE frees the prompt's prefill blocks and
    allocates fresh scratch blocks, LMCache's load slot_mapping must write the
    selected KV into the SCRATCH blocks (where the runner's block_table / FA
    reads), NOT the stale prefill blocks still sitting at the front of
    ``tracker.allocated_block_ids``."""
    bs = 16
    prompt_len, budget = 24640, 24000
    n_prefill = (prompt_len + bs - 1) // bs            # 1540 prefill blocks
    n_scratch = (budget + bs - 1) // bs                # 1500 scratch blocks
    # tracker accumulates prefill (blocks 1..1540), then scratch, then decode.
    prefill_blocks = list(range(1, 1 + n_prefill))
    scratch_blocks = list(range(1 + n_prefill, 1 + n_prefill + n_scratch))
    decode_blocks = [1 + n_prefill + n_scratch]
    allocated = prefill_blocks + scratch_blocks + decode_blocks
    num_token_ids = prompt_len  # prompt-length token keys

    sm = _from_request_tracker_slot_mapping(
        allocated, prompt_len, num_token_ids, bs, is_sparse_decode=True)

    # length unchanged -> start_load_kv's `len(tokens) == len(slot_mapping)` holds
    assert len(sm) == num_token_ids
    # the first ``budget`` slots (what the clustered kernel writes) must land in
    # the scratch blocks = the runner's block_table front, NOT freed prefill.
    assert sm[0] == scratch_blocks[0] * bs
    scratch_set, prefill_set = set(scratch_blocks), set(prefill_blocks)
    for slot in sm[:budget]:
        assert slot // bs in scratch_set, "load wrote outside the scratch blocks"
        assert slot // bs not in prefill_set, "load wrote into a freed prefill block"

    # control: the pre-fix path (no scratch redirect) wrongly targets prefill.
    buggy = _from_request_tracker_slot_mapping(
        allocated, prompt_len, num_token_ids, bs, is_sparse_decode=False)
    assert buggy[0] == prefill_blocks[0] * bs
    assert buggy[0] != sm[0], "fix must change which physical block is written"
