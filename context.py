from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_decode: bool = False
    enable_token_eviction: bool = False
    mask_indices: torch.Tensor | None = None
    processing_indices: torch.Tensor | None = None
    unprocessed_positions: torch.Tensor | None = None
    reference_indices: torch.Tensor | None = None
    K: int | None = None
    strategy: str | None = None
    first_layer_importance: torch.Tensor | None = None
    eviction_phase: bool = False  # New flag to indicate token eviction phase

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_decode, enable_token_eviction=False, mask_indices=None, processing_indices=None, K=None, strategy=None, first_layer_importance=None,
                eviction_phase=None, reference_indices=None, unprocessed_positions=None):
    global _CONTEXT
    _CONTEXT.is_decode = is_decode
    _CONTEXT.enable_token_eviction = enable_token_eviction
    if mask_indices is not None:
        _CONTEXT.mask_indices = mask_indices
    if processing_indices is not None:
        _CONTEXT.processing_indices = processing_indices
    if K is not None:
        _CONTEXT.K = K
    if strategy is not None:
        _CONTEXT.strategy = strategy
    if first_layer_importance is not None:
        _CONTEXT.first_layer_importance = first_layer_importance
    if eviction_phase is not None:
        _CONTEXT.eviction_phase = eviction_phase
    if reference_indices is not None:
        _CONTEXT.reference_indices = reference_indices
    if unprocessed_positions is not None:
        _CONTEXT.unprocessed_positions = unprocessed_positions

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
