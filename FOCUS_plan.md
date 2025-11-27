# FOCUS Integration Plan

## Objective

- Port the attention-importance FOCUS pipeline implemented in `focus_generate.py` and Hugging Face’s `modeling_sdar.py` into LMDeploy’s PyTorch DLLM stack.
- Keep the existing delayed-cache flow intact and reuse its ragged-kernel plumbing; FOCUS becomes an optional refinement that prunes masked tokens between the first two SDAR decoder layers.
- Ensure tensor shapes (`hidden_states`, ragged metadata) and runtime metadata (`processing_indices`, `position_ids`, `attn_metadata`...) stay consistent after the second layer starts operating on the trimmed token set.

## Key References

- HF artifacts: `focus_generate.py`, `modeling_sdar.py`, `context.py`.
- LMDeploy components:
  - Scheduler & message state: `lmdeploy/pytorch/strategies/dllm/sequence.py`
  - DLLM model agent: `lmdeploy/pytorch/strategies/dllm/model_agent.py`
  - Engine input builder: `lmdeploy/pytorch/engine/engine.py`
  - Model inputs/context: `lmdeploy/pytorch/model_inputs.py`
  - SDAR model wrapper: `lmdeploy/pytorch/models/sdar.py`
  - Triton attention metadata & kernels: `lmdeploy/pytorch/backends/cuda/{op_backend.py,attention.py}`, `lmdeploy/pytorch/kernels/cuda/pagedattention.py`

## Constraints & Assumptions

- FOCUS only triggers during DLLM decoding when both delayed cache and FOCUS flags are ON; prefill and standard auto-regressive decoding remain unchanged.
- The FOCUS decision needs all masked tokens in the current block (to compute attention deltas). Only after layer 1 (0-indexed)'s QK projections (and Normalizations) finishes can we safely shrink the token dimension that feeds layer 1's attention and so on.
- Triton ragged kernels already handle ragged `q_start_loc/q_seqlens`; FOCUS-pruned tokens must update those tensors and the associated `processing_indices`/`processing_q_lens` so later layers, KV fills, logits scatter, and scheduler bookkeeping stay coherent.
- `dllm_mask` semantics stay identical (`MASKED`, `UNMASKED`, `CACHED`). FOCUS only prunes which MASKED rows get processed this denoising step; it does **not** immediately change the mask stored in the scheduler.
- Configuration validation must ensure FOCUS can only be enabled when delayed cache is also enabled; otherwise, the engine must error or silently disable FOCUS.

## Implementation Steps

1. **Expose FOCUS Controls**
   - Extend `DLLMConfig` with knobs such as `enable_focus` plus a ratio-based `focus_alpha`.
   - Parse CLI / engine config to populate these fields and surface them through `DLLMStrategyFactory` so all DLLM strategies can query a unified configuration object.
   - Guard all new code paths with `dllm_config.enable_focus` to keep default behaviour unchanged.
   - Enforce `enable_focus ⇒ enable_delayed_cache` inside `PytorchEngineConfig.__post_init__` so there is a single validation source and downstream layers can assume the configuration is sane without duplicating guard rails.
2. **Augment Sequence State With Mask Metadata**
    - In `SchedulerSequenceDLLM`, add helpers that build PyTorch tensors for:
      - `mask_indices`: boolean mask over the current block (derived from `dllm_mask == DLLM_MASKED`). Treat this block-length mask as the source of truth and always build ragged masks via `processing_mask = mask_indices[..., processing_indices]` so anything flattened later stays aligned with the scheduler’s ordering.
      - `unprocessed_positions`: mirrors HF’s `unprocessed_positions` to remember which masked tokens still need at least one forward pass.
   - Introduce a standalone `FocusState` to own `unprocessed_positions`, the latest mask snapshot, and any focus-only counters. This keeps the original delayed-cache logic unchanged while still letting FOCUS track which tokens require another pass. Whenever `_update_focus_state` runs (before each decode step), refresh `unprocessed_positions` to all “True” if the sequence moves to a new block.
   - Add a method (e.g., `SchedulerSequenceDLLM.get_focus_info()`) that packages the block-level `mask_indices`, `unprocessed_positions`, and rolling statistics (e.g., `avg_decoded_tokens`) for the engine.
3. **Thread FOCUS Info Through Engine Inputs**
   - In `Engine.create_model_inputs`, when delayed cache is enabled, gather all per-sequence FOCUS tensors:
     - Concatenate `processing_indices` as today.
     - Store `processing_q_lens` (counts per message).
     - Keep `unprocessed_positions` as a block-length tensor per sequence and pass it through unchanged; the model will scatter updates back into this view after pruning.
   - Extend `ModelInputs` with optional FOCUS fields (`focus_block_unprocessed`, ragged mask indices/indptr, rolling averages) plus convenience accessors to slice them per sequence.
   - Avoid duplicating derived ragged metadata such as local offsets or dense boolean masks; a CSR indptr together with flattened global mask indices and the existing `processing_indices` already capture the full ragged view.
   - Ensure these tensors are CUDA-pinned / on CPU when the context builder gathers them (mirroring current `processing_indices` handling).
4. **Extend StepContext & Attention Metadata**
   - Update `StepContext.new` to:
     - Store a back-reference to the originating `ModelInputs` (e.g., `source_inputs`) so the model can mutate `processing_indices` mid-forward; this is crucial for logits reshaping later.
     - Carry the flattened `processing_indices` tensor (`processing_tensor` already exists) and the new mask tensors on-device.
   - When `use_delayed_cache` is true, attach a lean per-sequence `FocusRuntimeView` inside `StepContext` that only tracks the masks needed at runtime (`block_unprocessed`, ragged mask indices/CSR pointers, `avg_decoded_tokens`). Avoid duplicating the full ragged metadata snapshot—the context itself already owns the canonical view and `update_processing_view` handles mutations.
     - Keep focus bookkeeping tensors on the same device as the step context; avoid bouncing to CPU except when exporting scheduler metadata back through `commit_processing_view`.
   - Extend `TritonAttentionMetadata` to include `processing_indices` & a `use_delayed_cache` flag (already there) plus a pointer or indices describing the current ragged view. Allow in-place updates so the second layer can change `q_seqlens`, `q_start_loc`, and `processing_indices` without rebuilding the whole metadata object.
   - Split Triton attention invocations so layer 1 first calls `forward_only_fill_kv` with the *original* ragged metadata, then updates the metadata and launches `forward_only_attention` on the trimmed view. Later layers can keep using the combined `forward` path.
5. **Model-Agent Plumbing**
   - Before `model_forward`, convert scheduler-provided numpy arrays into torch tensors and stash them into `ModelInputs` (mirrors existing delayed-cache path).
   - Populate the newly-extended `StepContext` fields (mask indices, `FOCUSView`, focus thresholds, etc.) right after `ctx_mgr.build_context` so the model can consult them directly through `ctx_mgr.current_context()` without relying on external globals.
   - After the forward pass, read back the possibly updated `processing_indices`/`processing_q_lens` from the same `StepContext` instance and overwrite `origin_inputs.processing_indices` & `processing_q_lens` before `reshape_logits` runs. This guarantees the logits scatter map matches the actually-processed tokens.
6. **FOCUS Logic Inside `lmdeploy/pytorch/models/sdar.py`**
   - **Layer 0 (importance gathering)**:
    - Inside `SDARAttention.forward`, detect `focus_enabled` when `layer_idx in {0, 1}`, `context.is_decoding`, and the runtime mask contains more masked tokens than the configured ratio would retain (all pulled from the active `StepContext`).
     - Slice `query_states`/`key_states` to the masked positions (sorted via `processing_indices`), repeat KV heads for GQA, compute scaled attention logits, run the 1D max-pool + softmax trick, and store per-token importance sums.
     - Save these scores in the runtime context (`first_layer_importance`) without changing tensor shapes yet. The actual attention block for layer 0 still consumes the full block (ensuring KV/cache writes stay dense) while the importance path operates on the masked-only slice.
   - **Transition between layer 0 and layer 1 (shape/mask rewrite)**:
     - After layer 0 finishes, use the stored importance delta logic from HF (`mean/std` dynamic selection, adjacency handling, unprocessed fallbacks) to produce `retain_mask` over the masked positions, deriving lengths directly from the ragged CSR metadata.
     - Mirror HF’s `adjacent_and_right_selected` behaviour so a masked token is retained when its right neighbour survives but it would otherwise be evicted.
     - Collapse that mask directly onto `context.processing_indices` to compute `retained_processing_indices` for the surviving rows.
     - Apply this trim to the tensors feeding layer 1 (e.g., `query_states`, `key_states`, `value_states`, residual streams) so only the retained tokens continue through the attention stack, and recompute `input_shape` to reflect the shortened sequence length.
     - Update rotary position inputs and attention masks by re-indexing `position_ids`/`attention_mask` with `retained_processing_indices` so that everything downstream already reasons in the trimmed token space.
   - **Update Ragged Metadata Before Layer 1 Attn**:
     - Use the new helper in StepContext to:
       - Replace `context.processing_indices` with the flattened list of retained block positions (i.e., indices into the DLLM block).
       - Recompute `processing_q_lens` per sequence (counts of retained tokens).
        - Mutate `attn_metadata.q_seqlens`, `q_start_loc`, and `kv_seqlens` to match the resized input so Triton ragged kernels operate on the trimmed batch.
        - Update `attn_metadata.processing_indices` so only the attention kernels touch the surviving token rows after the metadata refresh; layer 1 should issue a standalone KV-fill call *before* pruning so cache writes still cover the full mask.
     - The `FocusRuntimeView` only supplies the ragged mask metadata and per-sequence bookkeeping; rely on `StepContext`'s existing methods to snapshot/update the canonical ragged tensors instead of mirroring them inside the focus helper.
   - **Layer 1 (second attention) forward**:
     - Use the trimmed tensors and updated metadata; `SDARAttention.forward` should now see the ragged subset automatically through `attn_metadata`.
   - **Layers >= 2**:
     - Maintain the ragged view created after layer 1. No further FOCUS pruning occurs this block; use the updated metadata for the rest of the stack.
   - **Exiting the model**:
     - When returning `hidden_states`, keep them in ragged order (matching the updated `processing_indices`). The downstream `reshape_logits` path will scatter them back using the modified `ModelInputs.processing_indices`.
7. **Engine / Scheduler Feedback Loop**
   - Before unmasking, scatter the trimmed `processing_indices` back into block space so the scheduler knows exactly which positions were fed through the model this round. During unmasking, only touch those rows—mirror `focus_generate.py` by leaving untouched positions intact so partially processed masks are preserved accurately.
   - To keep the DLLM unmasker from looking at stale logits, temporarily flip every masked-but-unprocessed slot to the `DLLM_CACHED` state before calling the unmasker, then restore them back to `DLLM_MASKED` afterward. This protects untouched rows without forcing a costly logits buffer memset.
   - After unmasking, still update `SchedulerSequenceDLLM._delayed_cache_state` using the `dllm_mask` so FOCUS-pruned tokens remain masked for the rest of the denoising steps.
   - Keep `unprocessed_positions` in sync: once a token has been processed, flip its slot to `False` so future iterations know it has already consumed one forward pass.
   - Persist the per-sequence processed indices in `model_metas` so the scheduler can update statistics and delayed-cache bookkeeping on the next step without reconstructing dense masks.
8. **Testing & Validation**
   - Unit-level: add tests for the focus pruning path to ensure the lean runtime view still propagates masks correctly (masked tokens retained across layers, processed-mask scatter/writeback).
   - Kernel-level: verify ragged attention outputs match dense reference when no tokens are FOCUS-pruned (retain mask == 1) and when only a subset is processed.

## Notes

- Keep the HF `context.py` logic as a behavioural reference but re-implement it within LMDeploy’s runtime (prefer a dataclass attached to `StepContext`) to avoid cross-module globals.
- Defer FOCUS entirely when the number of masked tokens is already ≤ the ratio-derived target, when delayed cache is disabled, or when ragged kernels are unavailable (fallback to dense path).
- Ensure all tensors manipulated for indexing remain sorted to satisfy `ragged_paged_attention` kernel assumptions.
