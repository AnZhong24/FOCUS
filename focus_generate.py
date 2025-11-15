import argparse
import torch
import math
import numpy as np
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import same_seeds, sample_with_temperature_topk_topp, get_num_transfer_tokens
from sampling import get_transfer_index
from context import get_context, set_context


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
        confidence_threshold=0.9, #0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        use_block_cache=False,
        block_size=32,
        strategy="none",
        alpha=-1.0,
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()
    
    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    assert input_ids.shape[0] == 1, "Batch size must be 1 for now"
    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        # Set is_decode to False before prefilling
        set_context(is_decode=False)
        
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True,
              use_block_cache=False,
              block_size=block_size)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    num_processed_tokens = 0
    num_generated_tokens = 0
    num_steps = 0
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        
        # Initialize delay cache tracking for the current block
        if use_block_cache:
            block_key_values = None
            uncached_positions = torch.ones(block_length, dtype=torch.bool, device=cur_x.device)
            unprocessed_positions = torch.ones(block_length, dtype=torch.bool, device=cur_x.device)
        
        num_decoded_tokens_list = []
        for step in range(denoising_steps + 1):
            num_steps += 1
            mask_indices = (cur_x == mask_id)
            if mask_indices.sum() == 0:
                # Store kv cache
                set_context(is_decode=False)
                num_generated_tokens += block_length
                num_processed_tokens += block_length
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True,
                      use_block_cache=False)
                break

            # Delay cache mechanism
            if use_block_cache:
                # NOTE: kernels rely on these indices being sorted ascending,
                # torch.nonzero preserves that order so avoid reordering.
                processing_indices = torch.nonzero(uncached_positions, as_tuple=True)[0]
                set_context(is_decode=True, enable_token_eviction=(strategy!="none"), mask_indices=mask_indices,
                            processing_indices=processing_indices, unprocessed_positions=unprocessed_positions)

                # Forward pass with delay cache parameters
                output = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                    use_block_cache=True,
                    block_key_values=block_key_values)
                logits, block_key_values = output.logits, output.block_key_values

                # Delayed cache mechanism
                non_mask_indices = ~mask_indices
                uncached_positions &= ~(non_mask_indices[0] & \
                    torch.cat([non_mask_indices[0, 1:], torch.tensor([True], device=non_mask_indices.device)], dim=0))

                # Rule out the unprocessed positions from mask_index
                context = get_context()
                num_processed_tokens += context.processing_indices.shape[0]
                processed_mask_indices = torch.zeros_like(mask_indices, dtype=torch.bool, device=mask_indices.device)
                processed_mask_indices[:, context.processing_indices] = mask_indices[:, context.processing_indices]
                unprocessed_positions[context.processing_indices] = False
                if strategy != "none" and context.reference_indices is not None:
                    processed_mask_indices[:, context.reference_indices] = False
                    context.reference_indices = None

                # Get logits for the current uncached positions
                full_logits = torch.zeros(cur_x.shape[0], block_length, model.config.vocab_size, device=cur_x.device, dtype=logits.dtype)
                full_logits[:, context.processing_indices] = logits
            else:
                set_context(is_decode=True, enable_token_eviction=False, processing_indices=torch.arange(block_length, device=cur_x.device))
                # Denosing
                logits = model(cur_x,
                               attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True,
                               use_block_cache=False,
                               store_kv=False).logits
                num_processed_tokens += block_length
                full_logits = logits
                processed_mask_indices = mask_indices

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                full_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
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
            if alpha > 0:
                num_decoded_tokens = transfer_index.sum().item()
                num_decoded_tokens_list.append(num_decoded_tokens)
                set_context(is_decode=False, K=math.ceil(np.mean(num_decoded_tokens_list)*alpha))

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x, num_generated_tokens, num_processed_tokens, num_steps


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=2048,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block_length", type=int, default=4,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic",
                                 "low_confidence_static",
                                 "sequential",
                                 "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (e.g. eos_token_id)")
    parser.add_argument("--use_block_cache", action='store_true',
                        help="Enable delay cache mechanism")
    parser.add_argument("--block_size", type=int, default=32,
                        help="Block size for delay cache mechanism")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of tokens to retain")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["top_k", "bottom_k"],
                        help="Strategy for selecting tokens to retain")

    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],)
    
    args = parser.parse_args()
    
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error(
            "--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic"
        )
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error(
            "--eb_threshold is required when --remasking_strategy=entropy_bounded"
        )
    return args


if __name__ == "__main__":
    same_seeds(42)
    args = parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.dtype,
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.mask_id is None:
        args.mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    if args.stopping_criteria_idx is None:
        gen_cfg = GenerationConfig.from_pretrained(args.model_dir,)
        args.stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx,]
    args.stop_words = tokenizer.convert_ids_to_tokens(
        args.stopping_criteria_idx)
    print(f"Your Arguments: {args}")

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        # dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        dict(role="user", content="Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' \
            market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        # dict(role="user", content="A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    messages = tokenizer.apply_chat_template(
        origin_prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=args.prompt_length
    )

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    # Set K and strategy in context for token quitting
    if args.use_block_cache and args.K is not None and args.strategy is not None:
        from context import set_context
        set_context(is_decode=True, K=args.K, strategy=args.strategy)
    
    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
        stopping_criteria_idx=args.stopping_criteria_idx,
        use_block_cache=args.use_block_cache,
        block_size=args.block_size,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)
