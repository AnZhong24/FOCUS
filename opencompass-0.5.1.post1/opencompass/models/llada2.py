from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .base import BaseModel
from .sdar_utils import get_context, set_context


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
    gen_length: int,
    block_length: int,
    denoising_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    threshold: float,
    eos_id: Optional[int],
    eos_early_stop: bool = True,
    use_block_cache: bool = False,
    strategy: str = "none",
    alpha: float = -1.0,
):
    """Block-wise diffusion generation with KV caching between blocks."""
    model.eval()
    device = getattr(model, "device", None) or next(model.parameters()).device
    input_ids = prompt["input_ids"]
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    mask_dtype = next(model.parameters()).dtype
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .log()
        .to(mask_dtype)
    )
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    set_context(is_decode=False)
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    num_transfer_tokens = model._get_num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        start = num_block * block_length
        end = (num_block + 1) * block_length
        cur_x = x[:, start:end].clone()
        cur_attn_mask = block_diffusion_attention_mask[:, :, start:end, :end]
        cur_position_ids = position_ids[:, start:end]

        if use_block_cache:
            block_key_values = None
            uncached_positions = torch.ones(block_length, dtype=torch.bool, device=device)
            unprocessed_positions = torch.ones(block_length, dtype=torch.bool, device=device)

        num_decoded_tokens_list: List[int] = []
        for step in range(denoising_steps + 1):
            mask_indices = cur_x == mask_id
            if mask_indices.sum() == 0:
                set_context(is_decode=False)
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
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
                    cur_x,
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
                processed_mask_indices = torch.zeros_like(mask_indices, dtype=torch.bool, device=device)
                processed_mask_indices[:, context.processing_indices] = mask_indices[:, context.processing_indices]
                unprocessed_positions[context.processing_indices] = False

                full_logits = torch.zeros(
                    cur_x.shape[0], block_length, model.config.vocab_size, device=device, dtype=logits.dtype
                )
                full_logits[:, context.processing_indices] = logits
                mask_for_transfer = processed_mask_indices
            else:
                set_context(
                    is_decode=True,
                    enable_token_eviction=False,
                    mask_indices=None,
                    processing_indices=torch.arange(block_length, device=device),
                )
                logits = model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits
                full_logits = logits
                mask_for_transfer = mask_indices

            x0, x0_p = model._sample_with_temperature_topk_topp(
                full_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            num_to_transfer = num_transfer_tokens[min(step, num_transfer_tokens.shape[0] - 1)].item()
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            confidence = torch.where(mask_for_transfer, x0_p, torch.full_like(x0_p, float("-inf")))

            high_conf_mask = confidence[0] > threshold
            num_high_confidence = high_conf_mask.sum().item()
            available = int(mask_for_transfer.sum().item())
            num_to_transfer = min(num_to_transfer, available)

            if num_high_confidence >= num_to_transfer:
                transfer_index[0] = high_conf_mask
            elif num_to_transfer > 0:
                _, idx = torch.topk(confidence[0], k=num_to_transfer)
                transfer_index[0, idx] = True

            if transfer_index.any():
                cur_x[transfer_index] = x0[transfer_index]

            if strategy != "none" and alpha > 0:
                num_decoded_tokens = int(transfer_index.sum().item())
                num_decoded_tokens_list.append(num_decoded_tokens)
                mean_decoded = sum(num_decoded_tokens_list) / len(num_decoded_tokens_list)
                set_context(is_decode=False, K=math.ceil(mean_decoded * alpha))

            x[:, start:end] = cur_x

            if eos_early_stop and eos_id is not None:
                eos_positions = (x[0, prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = prompt_length + eos_positions[0].item()
                    if (x[0, prompt_length:eos_pos] != mask_id).all():
                        return x[:, : eos_pos + 1]

        x[:, start:end] = cur_x

    generated_answer = x[:, : prompt_length + gen_length]

    if eos_id is not None:
        eos_positions = (generated_answer[0, prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            return generated_answer[:, : prompt_length + first_eos + 1]

    return generated_answer


class LLaDA2(BaseModel):
    """Lightweight wrapper for LLaDA2 diffusion generation."""

    def __init__(
        self,
        path: str,
        max_seq_len: int = 4096,
        tokenizer_only: bool = False,
        meta_template: Optional[Dict] = None,
        block_length: int = 32,
        steps: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        confidence_threshold: float = 0.9,
        use_block_cache: bool = False,
        K: int = 1,
        strategy: str = "none",
        alpha: float = -1.0,
        mask_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
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
        assert strategy in ["none", "top", "bottom", "random", "dynamic"]

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.block_length = block_length
        self.steps = steps
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.confidence_threshold = confidence_threshold
        self.dtype = dtype
        self.use_block_cache = use_block_cache
        self.K = K
        self.strategy = strategy
        self.alpha = alpha

        set_context(is_decode=False, K=K, strategy=strategy)

        torch_dtype = torch.bfloat16
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
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
            if getattr(self.tokenizer, "mask_token_id", None) is not None:
                mask_id = self.tokenizer.mask_token_id
            elif getattr(self.tokenizer, "mask_token", None):
                mask_token = self.tokenizer.mask_token
                mask_id = (
                    self.tokenizer(mask_token)["input_ids"][0]
                    if mask_token is not None
                    else None
                )
            if mask_id is None:
                mask_id = 156895
        self.mask_id = mask_id
        self.eos_id = eos_id or self.tokenizer.eos_token_id

        if not tokenizer_only:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ).to(self.device)
            self.model.eval()
        else:
            self.model = None

    def get_token_len(self, prompt: str) -> int:
        messages = _convert_chat_messages([prompt], include_system_prompt=True)[0]
        tokenized = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True
        )
        return len(tokenized["input_ids"])

    def generate(self, inputs: Sequence[PromptInput], max_out_len: int) -> List[str]:
        if self.tokenizer_only:
            raise ValueError("Cannot generate with tokenizer_only=True")

        outputs: List[str] = []
        for item in inputs:
            messages = self._normalize_prompt(item)
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            tokenized = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            input_ids = tokenized["input_ids"]
            seq_len = input_ids.shape[1]

            set_context(is_decode=False, K=self.K, strategy=self.strategy)
            generated_ids = block_diffusion_generate(
                model=self.model,
                prompt=tokenized,
                mask_id=self.mask_id,
                gen_length=max_out_len,
                block_length=self.block_length,
                denoising_steps=self.steps,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                threshold=self.confidence_threshold,
                eos_id=self.eos_id,
                eos_early_stop=True,
                use_block_cache=self.use_block_cache,
                strategy=self.strategy,
                alpha=self.alpha,
            )
            decoded = self.tokenizer.decode(
                generated_ids[0, seq_len:], skip_special_tokens=True
            )
            outputs.append(decoded)

        return outputs

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        raise NotImplementedError("LLaDA2 does not support perplexity mode.")

    def get_ppl_tokenwise(
        self, inputs: List[str], mask_length: Optional[List[int]] = None
    ) -> List[float]:
        raise NotImplementedError("LLaDA2 does not support perplexity mode.")

    def encode(self, prompt: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @staticmethod
    def _normalize_prompt(item: PromptInput) -> List[Dict[str, str]]:
        if isinstance(item, str):
            return _convert_chat_messages([item], include_system_prompt=True)[0]
        if isinstance(item, list) and item:
            if item[0].get("role") != "system":
                return [{"role": "system", "content": "You are a helpful assistant."}] + item
            return item
        return _convert_chat_messages([""], include_system_prompt=True)[0]


__all__ = ["LLaDA2"]
