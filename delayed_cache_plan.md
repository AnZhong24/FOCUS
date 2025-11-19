# Delayed Cache Integration Plan

## Objective
- Port the delayed KV-cache mechanism used by Hugging Face’s SDAR backend (`modeling_sdar.py`, `focus_generate.py`) into LMDeploy’s PyTorch DLLM engine (`lmdeploy/pytorch/strategies/dllm/__init__.py`) without enabling token eviction.
- Preserve existing decoding behaviour when the feature is disabled and guard the new pathway behind explicit configuration switches.

## Key References
- `modeling_sdar.py`: encoder/decoder changes for delayed cache flow (`use_block_cache`, selective KV updates).
- `focus_generate.py`: orchestration of delayed cache passes and context updates (`uncached_positions`, span derivation, etc.).
- LMDeploy components that must cooperate:
  - Sequence management: `lmdeploy/pytorch/strategies/dllm/sequence.py`
  - Strategy factory glue: `lmdeploy/pytorch/strategies/dllm/__init__.py`
  - Model agent loop: `lmdeploy/pytorch/strategies/dllm/model_agent.py`
  - Engine/runtime wiring: `lmdeploy/pytorch/engine/*`, `lmdeploy/pytorch/models/sdar.py`

## Constraints & Assumptions
- Token eviction logic from the reference implementation is **out of scope**; retain only delayed cache mechanics.
- Sparse decoding only prunes which query rows are processed; every launched token still attends over the full `[0, kv_seqlen)` prefix exposed by the block cache.
- Plan for feature to be opt-in through DLLM configuration to avoid regressions.

## Implementation Steps

1. **Expose Feature Flags**
   - Extend `DLLMConfig` and the `DLLMStrategyFactory` (`lmdeploy/pytorch/strategies/dllm/__init__.py`) to recognise knobs such as `enable_delayed_cache`.
   - Thread these values through scheduler / engine configs so downstream components can check the flag without ad-hoc environment reads.

2. **Sequence State Enhancements**
- Update `SchedulerSequenceDLLM` in `sequence.py` to track per-block delayed-cache metadata (`uncached_positions`, contiguous span descriptors) alongside existing DLLM masks.
- When advancing decode steps (`DLLMSequenceStrategy.update_running`), snapshot the delayed-cache metadata **before** applying mask updates so any freshly unmasked tokens are queued for at least one more pass; this mirrors HF’s behaviour where blocks must process the newly exposed logits prior to being considered cached.
- Add helpers to reset/apply these masks each denoising step and ensure they are serialised/deserialised consistently (numpy ↔ torch).
   - Mirror the HF rule: mark a position as “cached” (stable) once both the token and its immediate right neighbour are unmasked. It's checked after each model forward (see `uncached_positions` update in `focus_generate.py`).
   - When delayed cache is enabled, compute per-sequence lengths (the count of unstable tokens) that double as the ragged `q_seqlens`, and derive `q_start_loc` via prefix sums. Because delayed cache only reveals contiguous suffixes per block, start/length pairs are sufficient for the GPU kernels—no flattened index buffer is required.
   - When delayed cache is disabled, skip these ragged helpers entirely and re-use the existing `q_seqlens = block_len` contract so we can call the legacy kernels unchanged.

3. **Attention Metadata Extensions**
   - Avoid the external `context.py`; instead, extend the attention metadata carried in `StepContext.attn_metadata` to transport delayed-cache fields (ragged `q_start_loc`, ragged `q_seqlens`, `use_block_cache`, etc.).
   - Guarantee that the flattened query tensor respects `[q_start_loc[i], q_start_loc[i] + q_seqlens[i])` so CTAs can compute pointer math without extra bookkeeping.
   - Ensure metadata defaults collapse to “full block” when delayed cache is disabled so existing kernels remain unaffected.

4. **Model Inputs & Position Handling**
   - Update the sequence scheduler / `ModelInputs` builder to pre-trim the attention masks and position ids using the delayed-cache metadata (so by the time `StepContext.get_mask_and_position_ids` runs, the tensors already reflect varlen processing).
   - During decode, treat the sequence lengths emitted by the scheduler as `q_seqlens` and derive ragged `q_start_loc` via prefix sums; these tensors drive both the sparse kernels and the later scatter-back step.
   - Ensure attention metadata (`q_seqlens`, `kv_seqlens`, block offsets) reflects the trimmed lengths when partial blocks are processed.

5. **Engine Scheduling Adjustments**
   - Disable the decode loop when delayed cache is enabled. It's hard to fix the forward_event() management.

6. **Model Agent Orchestration**
   - Augment `DLLMModelAgentStrategy` (`model_agent.py`) to:
     1. Derive `uncached_positions` and collapse them into contiguous `[start, len]` spans per sequence for denoising steps.
     2. If the block is fully unmaksed, emit the single span `[start=0, len=block_len]` so the kernel flushes the entire block before returning to dense mode.
     3. Invoke the model with delayed-cache parameters (`use_block_cache=True`) while relying on LMDeploy’s in-place KV storage, merge partial logits, and update `uncached_positions` using the stride-wise check prior to unmasking.
     4. At the tail of each decode pass, call into the sequence helpers added in Step 2 to (a) mark any tokens that became stable during this pass and (b) regenerate the ragged metadata for the next pass. This ensures the second through Nth passes only touch the newly unstable spans instead of reprocessing the entire block.

7. **Model Runtime Support & Kernel Selection**
   - Update `lmdeploy/pytorch/models/sdar.py` (and any other DLLM-capable model wrappers) so `forward` can accept delayed-cache flags and propagate them to attention.
   - Introduce sparse-aware copies of the Triton `fill_kv_cache` and `paged_attention` kernels that accept ragged `q_start_loc`/`q_seqlens` spans (without extra index buffers). Keep the existing dense kernels untouched and use them whenever delayed cache is disabled.
   - Teach `TritonAttentionImpl` to dispatch between the dense kernels and the new sparse variants based on the metadata flag, and to reshape inputs/outputs as needed before/after kernel calls so diffusion-only runs can opt in without impacting other models.

8. **KV Storage Semantics**
   - Add assertions or adapter code to guard against unsupported patterns before enabling delayed cache. When sparse mode is active, ensure block cache writes happen only for the requested indices before we call the sparse kernels.
   - Teach the attention metadata builder to set `kv_seqlens` to `history_len + span_end` (where `span_end = q_start_loc + q_seqlens - 1`) so each sparse launch still attends across the entire prefix that is valid in cache without trying to apply per-token cutoffs.

9.  **Strategy/Factory Integration**
   - Wire the configuration into `DLLMStrategyFactory.build_engine_strategy`, `.build_sequence_strategy`, and `.build_model_agent_strategy` so each component can opt into delayed cache based on configuration.
   - Validate that existing constructor signatures remain backward compatible.

10. **Sparse Kernel Semantics & Output Scatter**
   - `fill_kv_cache_sparse`: consume ragged `q_start_loc`/`q_seqlens`, compute the target offset inside each block (history offset + span-relative index), and scatter the key/value rows into the correct cache slots. The kernel only touches the contiguous subset so other positions remain untouched.
     - Inputs: same tensor layout as dense kernel plus `q_start_loc` and `q_seqlens`. Each CTA derives token offsets from `(start, len)` pairs, avoiding the need for an intermediate index list.
     - Launch shape: derive CTA scheduling from the ragged `q_seqlens` (e.g., CTA-per-chunk) instead of the dense block grid to keep sparse workloads balanced.
     - Writes: update only the selected cache rows; mask all others to keep previous contents intact.
   - `paged_attention_sparse`: read queries in ragged form (similar to the varlen flow in `lmdeploy/pytorch/backends/cuda/flash_attention.py`). Use the same ragged metadata to decide how many queries belong to each request without assuming contiguous blocks beyond the `(start, len)` contract.
     - Queries: load via `q_start_loc`/`q_seqlens` spans and write attention outputs back through the same varlen metadata so the tensor handed to the MLP already aligns with the ragged ordering.
     - Launch shape: derive CTA scheduling from the ragged `q_seqlens` (e.g., CTA-per-chunk) instead of the dense block grid to keep sparse workloads balanced (similar to the one in `lmdeploy/pytorch/backends/cuda/flash_attention.py`)
     - Implementation detail: build compact `tile_to_batch` / `tile_to_subtile` remap tables on the host so each launched CTA corresponds to real work, avoiding the old worst-case grid that immediately exited for short sequences.
     - KV readback: continue using `block_offsets`/`kv_seqlens` (decode still only touches the newest block per sequence) and just iterate over the ragged set of query rows within that block. Every sparse query should still attend over the entire `[0, kv_seqlen)` prefix implied by the capped value so causal semantics match the dense kernel.
     - Output: only the model logits (after the LM head) require scattering into `[batch, block_len, vocab]`; attention outputs stay in ragged form (ordered by the spans) until the end of the layer stack.
     - Simpilification: Quantization, learnable-sinks, sliding-window are not needed. Add assertions in the CLI to make sure they are disabled.
  
11. **After forward (Logit Assembly)**
   - `SDARForCausalLM.get_logits` expects contiguous `[tokens, hidden]` inputs, and the DLLM `UnmaskingProcessor` (`unmasking.py`) indexes logits by block layout (`block_len` stride). When delayed cache is enabled, run the LM head over the ragged hidden states ordered by `q_start_loc`/`q_seqlens`, and scatter the resulting logits into a dense `[batch, block_len, vocab]` tensor before calling the unmasker.
   - When delayed cache is disabled, reuse the existing dense pathway. The scatter/gather branch should be transparent to downstream components so unmasking, remasking, and `dllm_mask` updates continue to operate on full blocks regardless of how many tokens were processed in the current denoising step.

## Deliverables
- Code updates across the modules above.

## Open Questions / Follow-ups
- Validate sparse KV write support across all supported attention backends; if missing, scope the necessary backend work.
