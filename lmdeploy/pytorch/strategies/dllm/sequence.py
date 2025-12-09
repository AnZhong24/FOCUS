# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (HistoryTokenIds, InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSession, UpdateTokenMode, _to_ndarray)
from lmdeploy.pytorch.config import DLLMConfig

from ..ar.sequence import SchedulerSequenceDefault
from ..base.sequence import SequenceStrategy

SeqList = List['SchedulerSequenceDLLM']

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED
DLLM_MASK_DTYPE = np.uint8


class HistoryDLLMMask(HistoryTokenIds):

    def __init__(self, token_ids: np.ndarray = None, dtype: np.dtype = DLLM_MASK_DTYPE):
        super().__init__(token_ids=token_ids, dtype=dtype)


@dataclass
class FocusInfo:
    mask_indices: torch.Tensor
    rightmost_processed: int
    avg_decoded_tokens: float


@dataclass
class DelayedCacheState:
    block_length: int
    uncached_positions: torch.Tensor
    needs_warmup: bool = True

    @classmethod
    def new(cls, block_length: int):
        uncached = torch.ones((block_length, ), dtype=torch.bool, pin_memory=True)
        return cls(block_length=block_length, uncached_positions=uncached)

    def reset(self):
        self.uncached_positions.fill_(True)
        self.needs_warmup = True

    def mark_cached(self, ready_mask: torch.Tensor):
        ready_mask = ready_mask.to(device=self.uncached_positions.device, dtype=torch.bool)
        self.uncached_positions &= ~ready_mask

    def update_from_mask(self, dllm_mask: np.ndarray):
        non_mask = torch.from_numpy(dllm_mask != DLLM_MASKED)
        right_neighbor = torch.roll(non_mask, shifts=-1, dims=0)
        right_neighbor[-1] = True
        ready = non_mask & right_neighbor
        self.mark_cached(ready)

    def get_processing_indices(self) -> torch.Tensor:
        device = self.uncached_positions.device
        if self.needs_warmup:
            indices = torch.arange(self.block_length, dtype=torch.long, device=device)
        else:
            indices = self.uncached_positions.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                indices = torch.arange(self.block_length, dtype=torch.long, device=device)
        if not indices.is_pinned():
            indices = indices.pin_memory()
        self.needs_warmup = False
        return indices


@dataclass
class FocusState:
    block_length: int
    rightmost_processed: int

    @classmethod
    def new(cls, block_length: int):
        return cls(block_length=block_length, rightmost_processed=-1)

    def reset(self):
        self.rightmost_processed = -1

    def mark_processed(self, processed: int):
        assert processed >= 0 and processed < self.block_length, f'processed={processed}'
        if processed > self.rightmost_processed:
            self.rightmost_processed = processed

    def get_rightmost(self) -> int:
        return self.rightmost_processed


@dataclass
class SchedulerSequenceDLLM(SchedulerSequenceDefault):

    # For dllm
    history_dllm_mask: HistoryDLLMMask = field(default_factory=HistoryDLLMMask)
    dllm_config: DLLMConfig | None = None

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        self._num_valid_ids: int = len(self.history_cache)
        self._strategy: DLLMSequenceStrategy = self._seq_meta.strategy
        self._delayed_cache_state: DelayedCacheState | None = None
        if self._strategy.enable_delayed_cache:
            self._delayed_cache_state = DelayedCacheState.new(self.dllm_block_length)
        cfg = self.dllm_config
        focus_enabled = bool(cfg and getattr(cfg, 'enable_focus', False))
        self._focus_enabled = focus_enabled and self.delayed_cache_enabled
        self._focus_state: FocusState | None = FocusState.new(self.dllm_block_length) if self._focus_enabled else None
        self._focus_token_sum: float = 0.0
        self._focus_steps: int = 0
        self._dllm_processed_tokens: int = 0
        self._dllm_decode_steps: int = 0

    @property
    def dllm_mask(self):
        start = self.num_history_ids
        end = start + self._num_token_ids
        return self.history_dllm_mask._token_ids[start:end]

    @property
    def num_valid_ids(self):
        return self._num_valid_ids

    @property
    def generated_ids(self) -> np.ndarray:
        end = self.num_valid_ids
        start = end - self.num_new_tokens
        return self.history_cache._token_ids[start:end]

    @property
    def all_dllm_mask(self):
        return self.history_dllm_mask._token_ids[:self.num_all_ids]

    @property
    def dllm_block_length(self):
        return self._strategy.block_size

    @property
    def dllm_mask_token(self):
        return self._strategy.dllm_mask_token

    @property
    def delayed_cache_enabled(self) -> bool:
        return self._delayed_cache_state is not None

    def _reset_delayed_cache_state(self):
        if self._delayed_cache_state is not None:
            self._delayed_cache_state.reset()
        if self._focus_state is not None:
            self._focus_state.reset()
        self._focus_token_sum = 0.0
        self._focus_steps = 0

    def _update_delayed_cache_state(self):
        mask = self.dllm_mask
        self._delayed_cache_state.update_from_mask(mask)

    def get_processing_indices(self) -> torch.Tensor:
        indices = self._delayed_cache_state.get_processing_indices()
        # NOTE: sparse kernels assume each per-sequence slice is sorted
        # ascending, so keep the natural torch.nonzero ordering.
        return indices

    def get_focus_info(self) -> FocusInfo:
        mask_np = self.dllm_mask
        mask_tensor = torch.from_numpy(mask_np == DLLM_MASKED)
        mask_tensor = mask_tensor.to(dtype=torch.bool)
        rightmost = self._focus_state.get_rightmost() if self._focus_state is not None else -1
        avg_tokens = 1.0
        if self._focus_steps > 0:
            avg_tokens = self._focus_token_sum / self._focus_steps
        return FocusInfo(mask_indices=mask_tensor,
                         rightmost_processed=rightmost,
                         avg_decoded_tokens=avg_tokens)

    def mark_focus_processed(self, rightmost_processed: int):
        self._focus_state.mark_processed(rightmost_processed)

    def update_focus_stats(self, decoded_tokens: int):
        if decoded_tokens <= 0:
            return
        self._focus_token_sum += int(decoded_tokens)
        self._focus_steps += 1

    def record_decode_stats(self, processed_tokens: int):
        self._dllm_decode_steps += 1
        self._dllm_processed_tokens += processed_tokens

    def get_dllm_request_stats(self) -> Dict[str, int]:
        return dict(processed_tokens=self._dllm_processed_tokens,
                    decode_steps=self._dllm_decode_steps)

    def set_stop_pos(self, pos: int):
        dllm_block_length = self.dllm_block_length
        val = dllm_block_length - pos - 1
        self._num_valid_ids -= val
        self.num_new_tokens -= val

    def _update_token_ids_inputs(self, token_ids: np.ndarray, dllm_mask: np.ndarray):
        """Append tokens."""
        num_tokens = len(token_ids)
        dllm_block_length = self.dllm_block_length
        dllm_mask_token = self.dllm_mask_token
        new_token_ids = [token_ids]
        new_dllm_mask = [dllm_mask]

        # add uncached tokens in token_ids
        # for example, [cccc cccc uumm], the [uu] in last block is remain valid.
        num_remain_valid = self.num_valid_ids - self.num_history_ids
        if num_remain_valid != 0:
            prev_token_ids = self.valid_ids[-num_remain_valid:]
            prev_dllm_mask = np.full_like(prev_token_ids, DLLM_UNMASKED, dtype=DLLM_MASK_DTYPE)
            new_token_ids = [prev_token_ids] + new_token_ids
            new_dllm_mask = [prev_dllm_mask] + new_dllm_mask
            self.history_cache.resize(self.num_history_ids)
            self.history_dllm_mask.resize(self.num_history_ids)
            num_tokens += num_remain_valid

        # pad to align with dllm_block_length
        num_pad = (-num_tokens) % dllm_block_length
        if num_pad > 0:
            pad_ids = np.full_like(token_ids, dllm_mask_token, shape=(num_pad, ))
            pad_mask = np.full_like(dllm_mask, DLLM_MASKED, shape=(num_pad, ))
            new_token_ids += [pad_ids]
            new_dllm_mask += [pad_mask]

        token_ids = np.concatenate(new_token_ids)
        dllm_mask = np.concatenate(new_dllm_mask)

        assert len(token_ids) % dllm_block_length == 0

        self.history_cache.append(token_ids)
        self.history_dllm_mask.append(dllm_mask)
        self.output_start_pos = self._num_valid_ids + len(token_ids)
        self._num_valid_ids = self.num_history_ids + num_tokens
        self._num_token_ids = len(token_ids)
        self.num_new_tokens = 0

    def _update_token_ids_decode(self, token_ids: np.ndarray, dllm_mask: np.ndarray):
        """Update token ids for decode."""
        num_tokens = len(token_ids)
        dllm_block_length = self.dllm_block_length
        dllm_mask_token = self.dllm_mask_token
        assert num_tokens % dllm_block_length == 0
        num_history_ids = self.num_history_ids

        token_ids[dllm_mask == DLLM_MASKED] = dllm_mask_token
        self.history_cache[num_history_ids:] = token_ids
        self.history_dllm_mask[num_history_ids:] = dllm_mask

        # check if all blocks are cached
        last_mask = dllm_mask[-dllm_block_length:]
        is_unmasked = np.all(last_mask == DLLM_UNMASKED)
        is_cached = np.all(last_mask == DLLM_CACHED)

        if is_unmasked:
            num_new = dllm_block_length - self._num_valid_ids % dllm_block_length
            self._num_valid_ids += num_new
            self.num_new_tokens += num_new

        if is_cached:
            # add new block
            new_token_ids = np.full_like(token_ids, dllm_mask_token, shape=(dllm_block_length, ))
            new_dllm_mask = np.full_like(dllm_mask, DLLM_MASKED, shape=(dllm_block_length, ))
            self.history_cache.append(new_token_ids)
            self.history_dllm_mask.append(new_dllm_mask)
            self._num_history_ids += self._num_token_ids
            self._num_token_ids = dllm_block_length
            self._reset_delayed_cache_state()

    def _update_token_ids_prefill(self, token_ids: np.ndarray, dllm_mask: np.ndarray):
        """Update token ids for prefill."""
        dllm_block_length = self.dllm_block_length
        num_history_ids = self.num_history_ids

        # fill input cache
        if self.num_token_ids > dllm_block_length:
            end = self.num_token_ids - dllm_block_length
            self.history_dllm_mask[num_history_ids:end] = DLLM_CACHED
            self._num_history_ids += end
            self._num_token_ids -= end

        # decoding update
        self._update_token_ids_decode(token_ids, dllm_mask)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         dllm_mask: Tensor = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        self.arrive_time = time.perf_counter()

        token_ids: np.ndarray = _to_ndarray(token_ids)
        if dllm_mask is None:
            dllm_mask = np.full_like(token_ids, DLLM_UNMASKED, dtype=DLLM_MASK_DTYPE)
        dllm_mask: np.ndarray = _to_ndarray(dllm_mask)

        if mode == UpdateTokenMode.INPUTS:
            self._update_token_ids_inputs(token_ids, dllm_mask)
        elif mode == UpdateTokenMode.PREFILL:
            self._update_token_ids_prefill(token_ids, dllm_mask)
        else:
            self._update_token_ids_decode(token_ids, dllm_mask)

        if model_meta is not None:
            self.model_meta = model_meta


class DLLMSequenceStrategy(SequenceStrategy):

    def __init__(self,
                 block_size: int,
                 dllm_mask_token: int,
                 enable_delayed_cache: bool = False,
                 dllm_config: DLLMConfig = None) -> None:
        self.block_size = block_size
        self.dllm_mask_token = dllm_mask_token
        self.enable_delayed_cache = enable_delayed_cache
        self.dllm_config = dllm_config
        self.track = False
        if dllm_config is not None:
            self.track = dllm_config.track

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequenceDLLM':
        """Make sequence."""
        return SchedulerSequenceDLLM(seq_id=seq_id,
                                     session=session,
                                     sampling_param=sampling_param,
                                     adapter_name=adapter_name,
                                     migration_request=migration_request,
                                     resp_cache=resp_cache,
                                     preserve_cache=preserve_cache,
                                     dllm_config=self.dllm_config)

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, is_decoding: bool) -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)
        dllm_mask = batched_outputs.extra_outputs.dllm_mask
        stop_pos = batched_outputs.stop_pos

        batch_size = len(running)
        next_token_ids = next_token_ids.view(batch_size, -1).numpy()
        dllm_mask = dllm_mask.view(batch_size, -1).numpy()
        stop_pos = stop_pos.tolist()
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for idx, token in enumerate(next_token_ids):
            msg = running[idx]
            stop = stopped[idx]
            model_meta = model_metas[idx]
            mask = dllm_mask[idx]
            if msg.status != MessageStatus.LOCKED:
                continue

            prev_focus_mask = None
            if is_decoding and msg._focus_enabled:
                prev_focus_mask = msg.dllm_mask.copy()

            if is_decoding:
                if self.track:
                    processed_tokens = model_meta.get(consts.DLLM_META_PROCESSED_TOKENS)
                    msg.record_decode_stats(processed_tokens)
                if self.enable_delayed_cache:
                    # Refresh delayed-cache bookkeeping using the existing mask
                    # so newly unmasked tokens still get one more iteration.
                    msg._update_delayed_cache_state()
                if msg._focus_enabled and model_meta is not None:
                    processed_rightmost = model_meta.get('focus_processed_rightmost')
                    msg.mark_focus_processed(processed_rightmost)
            # fill token
            msg.update_token_ids(token, dllm_mask=mask, model_meta=model_meta, mode=update_mode)
            if is_decoding and msg._focus_enabled:
                curr_mask = msg.dllm_mask
                if prev_focus_mask is not None:
                    newly_unmasked = ((prev_focus_mask == DLLM_MASKED) & (curr_mask == DLLM_UNMASKED))
                    decoded_count = int(newly_unmasked.sum())
                    msg.update_focus_stats(decoded_count)
            if stop:
                msg.set_stop_pos(stop_pos[idx])
                msg.status = MessageStatus.TO_BE_MIGRATED if msg.preserve_cache else MessageStatus.STOPPED
