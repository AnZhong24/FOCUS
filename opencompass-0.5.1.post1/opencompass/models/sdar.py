from __future__ import annotations

import math
import atexit
import weakref
import sys
import importlib.util
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache

from .base import BaseModel
from .sdar_utils import (
    get_context,
    get_num_transfer_tokens,
    get_transfer_index,
    sample_with_temperature_topk_topp,
    set_context,
)

# Import and register local SDAR model classes to use with HuggingFace weights
# Use importlib to handle directory names with hyphens
current_dir = Path(__file__).parent
sdar_root = current_dir / "SDAR-8B-Chat-b32"
sdar_module_path = sdar_root / "modeling_sdar.py"
sdar_package_name = "opencompass.models.sdar_local"
sdar_module_name = f"{sdar_package_name}.modeling_sdar"

if sdar_package_name not in sys.modules:
    sdar_package = types.ModuleType(sdar_package_name)
    sdar_package.__path__ = [str(sdar_root)]
    sys.modules[sdar_package_name] = sdar_package

spec = importlib.util.spec_from_file_location(sdar_module_name, sdar_module_path)
sdar_module = importlib.util.module_from_spec(spec)
sys.modules[sdar_module_name] = sdar_module
spec.loader.exec_module(sdar_module)

SDARForCausalLM = sdar_module.SDARForCausalLM
SDARConfig = sdar_module.SDARConfig

# Register the model classes so they are used when loading from HuggingFace
AutoConfig.register("sdar", SDARConfig)
AutoModelForCausalLM.register(SDARConfig, SDARForCausalLM)


PromptInput = Union[str, List[Dict[str, str]]]


def _has_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
        return True
    return any(model_dir.glob("model-*.safetensors")) or any(model_dir.glob("pytorch_model-*.bin"))


def _maybe_download_weights(
    path: str,
    *,
    hf_repo: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
) -> str:
    if not hf_repo:
        return path
    model_dir = Path(path).expanduser()
    download_patterns: List[str] = []
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    ):
        if not (model_dir / filename).exists():
            download_patterns.append(filename)
    if not (model_dir / "tokenizer.json").exists() and not (model_dir / "tokenizer.model").exists():
        download_patterns.extend(["tokenizer.json", "tokenizer.model"])
    if not _has_model_weights(model_dir):
        download_patterns.extend(
            [
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "pytorch_model-*.bin",
            ]
        )
    if not download_patterns:
        return str(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        resume_download=True,
        allow_patterns=download_patterns,
    )
    return str(model_dir)


def _convert_chat_messages(
    inputs: Sequence[Union[str, List[Dict[str, str]]]],
    *,
    merge_role: bool = True,
    skip_empty_prompt: bool = True,
    include_system_prompt: bool = True,
) -> List[List[Dict[str, str]]]:
    """Convert raw inputs to chat messages compatible with chat templates."""
    outputs: List[List[Dict[str, str]]] = []
    role_map = {
        'HUMAN': 'user',
        'human': 'user',
        'USER': 'user',
        'user': 'user',
        'BOT': 'assistant',
        'bot': 'assistant',
        'ASSISTANT': 'assistant',
        'assistant': 'assistant',
        'SYSTEM': 'system',
        'system': 'system',
    }

    for _input in inputs:
        messages: List[Dict[str, str]] = []
        if isinstance(_input, str):
            if include_system_prompt:
                messages.append({'role': 'system', 'content': 'You are a helpful assistant.'})
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                content = item.get('content', item.get('prompt', ''))
                if skip_empty_prompt and not content:
                    continue
                role = role_map.get(item.get('role', ''), item.get('role', 'user'))
                messages.append({'role': role, 'content': content})
            if include_system_prompt and messages and messages[0]['role'] != 'system':
                messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + messages

        if merge_role:
            merged_messages: List[Dict[str, str]] = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)

    return outputs


@torch.inference_mode()
def block_diffusion_generate(
    model,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.9,
    eb_threshold=None,
    stopping_criteria_idx=None,
    use_block_cache=False,
    block_size=32,
    strategy="none",
    alpha=-1.0,
):
    """SDAR diffusion sampler ported locally for focus usage."""
    model.eval()
    device = model.device
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
        .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    assert input_ids.shape[0] == 1, "Batch size must be 1 for now"
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        set_context(is_decode=False)
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
            use_block_cache=False,
            block_size=block_size,
        )

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    # Decode stage
    num_processed_tokens = 0
    num_generated_tokens = 0
    num_steps = 0
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block * block_length:(num_block + 1) * block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block * block_length:(num_block + 1) * block_length, :(num_block + 1) * block_length
        ]
        cur_position_ids = position_ids[:, num_block * block_length:(num_block + 1) * block_length]

        if use_block_cache:
            block_key_values = None
            uncached_positions = torch.ones(block_length, dtype=torch.bool, device=device)
            unprocessed_positions = torch.ones(block_length, dtype=torch.bool, device=device)

        num_decoded_tokens_list = []
        for step in range(denoising_steps + 1):
            num_steps += 1
            mask_indices = (cur_x == mask_id)
            if mask_indices.sum() == 0:
                set_context(is_decode=False)
                num_generated_tokens += block_length
                num_processed_tokens += block_length
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                    use_block_cache=False,
                )
                break

            if use_block_cache:
                processing_indices = torch.nonzero(uncached_positions, as_tuple=True)[0]
                set_context(
                    is_decode=True,
                    enable_token_eviction=(strategy != "none"),
                    mask_indices=mask_indices,
                    processing_indices=processing_indices,
                    unprocessed_positions=unprocessed_positions,
                )
                output = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                    use_block_cache=True,
                    block_key_values=block_key_values,
                )
                logits, block_key_values = output.logits, output.block_key_values

                non_mask_indices = ~mask_indices
                uncached_positions &= ~(
                    non_mask_indices[0]
                    & torch.cat([non_mask_indices[0, 1:], torch.tensor([True], device=device)], dim=0)
                )
                # uncached_positions &= ~non_mask_indices[0]

                context = get_context()
                num_processed_tokens += context.processing_indices.shape[0]
                processed_mask_indices = torch.zeros_like(mask_indices, dtype=torch.bool, device=device)
                processed_mask_indices[:, context.processing_indices] = mask_indices[:, context.processing_indices]
                unprocessed_positions[context.processing_indices] = False

                full_logits = torch.zeros(
                    cur_x.shape[0], block_length, model.config.vocab_size, device=device, dtype=logits.dtype
                )
                full_logits[:, context.processing_indices] = logits
            else:
                set_context(
                    is_decode=True,
                    enable_token_eviction=False,
                    processing_indices=torch.arange(block_length, device=device),
                )
                logits = model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    use_block_cache=False,
                    store_kv=False,
                ).logits
                num_processed_tokens += block_length
                full_logits = logits
                processed_mask_indices = mask_indices

            x0, x0_p = sample_with_temperature_topk_topp(
                full_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            transfer_index = get_transfer_index(
                remasking_strategy=remasking_strategy,
                cur_x=cur_x,
                x0=x0,
                x0_p=x0_p,
                mask_index=processed_mask_indices,
                num_transfer_tokens_for_step=num_transfer_tokens[step],
                confidence_threshold=confidence_threshold,
                eb_threshold=eb_threshold,
            )

            cur_x[transfer_index] = x0[transfer_index]
            if strategy != "none" and alpha > 0:
                num_decoded_tokens = transfer_index.sum().item()
                num_decoded_tokens_list.append(num_decoded_tokens)
                set_context(is_decode=False, K=math.ceil(np.mean(num_decoded_tokens_list) * alpha))

        x[:, num_block * block_length:(num_block + 1) * block_length] = cur_x

        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x, num_generated_tokens, num_processed_tokens, num_steps


class SDARFocus(BaseModel):
    """Focus pipeline wrapper for SDAR chat models."""

    _instances = weakref.WeakSet()
    _atexit_registered = False

    def __init__(
        self,
        path: str,
        max_seq_len: int = 4096,
        tokenizer_only: bool = False,
        meta_template: Optional[Dict] = None,
        block_length: int = 32,
        denoising_steps: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        mask_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        remasking_strategy: str = 'low_confidence_dynamic',
        confidence_threshold: float = 0.9,
        eb_threshold: Optional[float] = None,
        use_block_cache: bool = False,
        device: str = 'cuda',
        dtype: str = 'bfloat16',
        K: int = 1,
        strategy: str = 'none',
        alpha: float = -1.0,
        hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
            **kwargs,
        )
        assert strategy in ['none', 'top', 'bottom', 'random', 'dynamic']

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.block_length = block_length
        self.denoising_steps = denoising_steps
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.remasking_strategy = remasking_strategy
        self.confidence_threshold = confidence_threshold
        self.eb_threshold = eb_threshold
        self.use_block_cache = use_block_cache
        self.dtype = dtype
        self.K = K
        self.strategy = strategy
        self.alpha = alpha
        self._total_generated_tokens = 0
        self._total_processed_tokens = 0
        self._total_steps = 0
        self._stats_logged = False

        SDARFocus._instances.add(self)
        if not SDARFocus._atexit_registered:
            atexit.register(SDARFocus._log_all_stats)
            SDARFocus._atexit_registered = True

        set_context(is_decode=False, K=K, strategy=strategy)

        torch_dtype = torch.bfloat16
        if dtype == 'float16':
            torch_dtype = torch.float16
        elif dtype == 'float32':
            torch_dtype = torch.float32

        model_path = _maybe_download_weights(
            path,
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if mask_id is None:
            if hasattr(self.tokenizer, 'mask_token') and self.tokenizer.mask_token:
                mask_id = self.tokenizer(self.tokenizer.mask_token)['input_ids'][0]
            else:
                mask_id = 151669
        self.mask_id = mask_id
        self.eos_id = eos_id or self.tokenizer.eos_token_id

        if not tokenizer_only:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=False,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ).to(self.device)
            self.model.eval()
        else:
            self.model = None

    # BaseModel contract -------------------------------------------------
    def get_token_len(self, prompt: str) -> int:
        messages = _convert_chat_messages([prompt], include_system_prompt=True)[0]
        tokenized = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True
        )
        return len(tokenized['input_ids'])

    def generate(self, inputs: Sequence[PromptInput], max_out_len: int) -> List[str]:
        if self.tokenizer_only:
            raise ValueError('Cannot generate with tokenizer_only=True')
        outputs: List[str] = []

        for item in inputs:
            messages = self._normalize_prompt(item)
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            tokenized = self.tokenizer([prompt], return_tensors='pt').to(self.device)
            input_ids = tokenized['input_ids']
            seq_len = input_ids.shape[1]

            # Reset context before each diffusion pass
            set_context(is_decode=False, K=self.K, strategy=self.strategy)
            generated_ids, num_generated_tokens, num_processed_tokens, num_steps = block_diffusion_generate(
                model=self.model,
                prompt=tokenized,
                mask_id=self.mask_id,
                gen_length=max_out_len,
                block_length=self.block_length,
                denoising_steps=self.denoising_steps,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                remasking_strategy=self.remasking_strategy,
                confidence_threshold=self.confidence_threshold,
                eb_threshold=self.eb_threshold,
                stopping_criteria_idx=[self.eos_id] if self.eos_id else None,
                use_block_cache=self.use_block_cache,
                strategy=self.strategy,
                alpha=self.alpha,
            )
            self._total_generated_tokens += num_generated_tokens
            self._total_processed_tokens += num_processed_tokens
            self._total_steps += num_steps
            decoded = self.tokenizer.decode(
                generated_ids[0, seq_len:], skip_special_tokens=True
            )
            outputs.append(decoded)

        return outputs

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        raise NotImplementedError('SDARFocus does not support perplexity mode.')

    def get_ppl_tokenwise(
        self, inputs: List[str], mask_length: Optional[List[int]] = None
    ) -> List[float]:
        raise NotImplementedError('SDARFocus does not support perplexity mode.')

    def encode(self, prompt: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __del__(self):
        self._print_stats()

    def _print_stats(self, *, force: bool = False):
        if self._stats_logged and not force:
            return
        if self._total_generated_tokens > 0 and self._total_steps > 0:
            print(f"[SDARFocus] Number of generated tokens: {self._total_generated_tokens}",)
            print(f"[SDARFocus] Number of processed tokens: {self._total_processed_tokens}",)
            print(f"[SDARFocus] Average processed tokens per generated token: {self._total_processed_tokens / self._total_generated_tokens:.2f}")
            print(f"[SDARFocus] Number of steps: {self._total_steps}")
            print(f"[SDARFocus] Average tokens per step: {self._total_generated_tokens / self._total_steps:.2f}")
            self._stats_logged = True

    @classmethod
    def _log_all_stats(cls):
        for instance in list(cls._instances):
            instance._print_stats()

    # Helpers ------------------------------------------------------------
    @staticmethod
    def _normalize_prompt(item: PromptInput) -> List[Dict[str, str]]:
        if isinstance(item, str):
            return _convert_chat_messages([item], include_system_prompt=True)[0]
        if isinstance(item, list) and item:
            if item[0].get('role') != 'system':
                return [{'role': 'system', 'content': 'You are a helpful assistant.'}] + item
            return item
        return _convert_chat_messages([''], include_system_prompt=True)[0]


__all__ = ['SDARFocus']
