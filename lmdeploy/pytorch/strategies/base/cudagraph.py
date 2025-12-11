# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Callable


class CudagraphStrategy(ABC):

    def get_capture_batch_size(self, target_batch_size: int, origin_batch_size: int,
                               capture_func: Callable[[int], int]) -> int:
        """Return capture batch size.

        Default behavior reuses the generic capture function.
        """
        return capture_func(target_batch_size)

    def get_capture_token_bucket(self, num_tokens: int) -> int:
        """Return capture token bucket."""
        return num_tokens

    @abstractmethod
    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        pass
