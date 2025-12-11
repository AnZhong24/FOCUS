# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable

from ..base.cudagraph import CudagraphStrategy


class DLLMCudagraphStrategy(CudagraphStrategy):

    _TOKEN_STEP = 256

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << ((value - 1).bit_length())

    @classmethod
    def _round_total_tokens(cls, total_tokens: int) -> int:
        """Apply dllm-specific growth rule on total tokens."""
        total_tokens = max(total_tokens, 1)
        if total_tokens <= cls._TOKEN_STEP:
            return cls._next_power_of_two(total_tokens)
        return ((total_tokens + cls._TOKEN_STEP - 1) // cls._TOKEN_STEP) * cls._TOKEN_STEP

    def get_capture_batch_size(self, target_batch_size: int, origin_batch_size: int,
                               capture_func: Callable[[int], int]) -> int:
        """Derive capture batch size using batch * block_size."""
        target_batch_size = max(target_batch_size or 1, origin_batch_size, 1)
        total_tokens = target_batch_size * self.block_size
        rounded_tokens = self._round_total_tokens(total_tokens)
        capture_batch_size = math.ceil(rounded_tokens / self.block_size)
        return max(capture_batch_size, origin_batch_size)

    def get_capture_token_bucket(self, num_tokens: int) -> int:
        """Bucket tokens for capture."""
        return self._round_total_tokens(num_tokens)

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        return batch_size * self.block_size
