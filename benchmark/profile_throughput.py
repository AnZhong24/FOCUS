# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import asyncio
import json
import os
import random
from queue import Queue
from typing import List, Union, Tuple

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.profiler import Profiler, Session
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import get_logger

get_logger('lmdeploy').setLevel('ERROR')
os.environ['TM_LOG_LEVEL'] = 'ERROR'


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    chat_template=None,  # Add chat_template parameter
) -> List[Tuple[str, int]]:
    """Sample requests from dataset for DLLM benchmarking.
    
    Returns:
        List of tuples containing (prompt, prompt_len)
    """
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts.
        prompt = dataset[i][0]
        
        # Apply chat template to format the prompt
        if chat_template is not None:
            # Convert raw prompt to message format
            messages = [{'role': 'user', 'content': prompt}]
            # Apply chat template
            prompt = chat_template.messages2prompt(messages, sequence_start=True, enable_thinking=False)
        
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        output_len = len(completion_token_ids)
        
        # Apply filters for prompt length only (no output length filtering)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
            
        filtered_dataset.append((prompt, prompt_len))

    print(f'#Input tokens: {np.sum([x[1] for x in filtered_dataset])}')
    print(f'#Prompts: {len(filtered_dataset)}')
    return filtered_dataset


def _subset_elapsed_time(sessions, start_ts, fallback):
    """Estimate elapsed time for a subset of profiler sessions."""
    if not sessions or start_ts is None:
        return fallback
    finish_ts = max((sess.ts[-1] for sess in sessions if sess.ts), default=None)
    if finish_ts is None:
        return fallback
    elapsed = finish_ts - start_ts
    return elapsed if elapsed > 0 else fallback


def _summarize_profiler_slice(profiler: Profiler, sessions, title: str, hyperparams):
    """Print profiler summary for the provided `sessions`."""
    if not sessions:
        return
    partial_profiler = Profiler(profiler.stream_output, profiler.percentages)
    partial_profiler.sessions = list(sessions)
    partial_profiler.t_start = getattr(profiler, 't_start', None)
    fallback_elapsed = getattr(profiler, 'elapsed_time', 0) or 1e-9
    partial_profiler.elapsed_time = _subset_elapsed_time(partial_profiler.sessions,
                                                         partial_profiler.t_start,
                                                         fallback_elapsed)
    partial_profiler.compute_metrics()
    partial_profiler.summarize(title=title, hyperparams=hyperparams)


def _full_batch_completion_target(total_prompts: int, concurrency: int) -> int:
    """Return the number of completions that keep the system fully saturated."""
    if total_prompts <= 0 or concurrency <= 0:
        return 0
    target = total_prompts - concurrency + 1
    if target <= 0 or target >= total_prompts:
        return 0
    return target


def _earliest_completed_sessions(profiler: Profiler, count: int):
    """Pick the earliest `count` completed sessions based on finish timestamps."""
    if count <= 0:
        return []
    completed = [sess for sess in profiler.sessions if sess.status == Session.SUCCESS and sess.ts]
    if not completed:
        return []
    completed.sort(key=lambda sess: sess.ts[-1])
    return completed[:min(count, len(completed))]


class Engine:

    def __init__(self, model_path: str, engine_config: Union[PytorchEngineConfig, TurbomindEngineConfig]):
        self.tokenizer = Tokenizer(model_path)
        
        # Automatically detect and use the model's corresponding chat template
        from lmdeploy.model import ChatTemplateConfig, best_match_model
        chat_template_name = best_match_model(model_path)
        self.chat_template_config = ChatTemplateConfig(model_name=chat_template_name, model_path=model_path)
        self.chat_template = self.chat_template_config.chat_template
        
        if isinstance(engine_config, TurbomindEngineConfig):
            from lmdeploy.turbomind import TurboMind
            tm_model = TurboMind.from_pretrained(model_path,
                                           engine_config=engine_config,
                                           chat_template_name=chat_template_name)
            self.backend = 'turbomind'
        elif isinstance(engine_config, PytorchEngineConfig):
            from lmdeploy.pytorch.engine import Engine as PytorchEngine
            tm_model = PytorchEngine.from_pretrained(model_path, engine_config=engine_config)
            self.backend = 'pytorch'

        self.tm_model = tm_model
        self.pbar = None

    async def _inference(self, req_queue: Queue, session_id: int, temperature: float, top_p: float, top_k: int,
                     stream_output: bool, skip_tokenize: bool, skip_detokenize: bool, concurrency: int,
                     max_new_tokens: int):
        model_inst = self.tm_model.create_instance()
        sess: Session = None
        for prompt, input_len, cancel_after, sess in iter(req_queue.get_nowait, None):

            sess.tick(0)

            if skip_tokenize:
                input_ids = prompt
            else:
                # print("Prompt:")
                # print(prompt)
                input_ids = self.tokenizer(prompt).input_ids

            state = DetokenizeState(len(input_ids))

            n_token = 0
            token_ids = input_ids.copy()

            generator = model_inst.async_stream_infer(session_id,
                                                      input_ids=input_ids,
                                                      gen_config=GenerationConfig(max_new_tokens=max_new_tokens,
                                                                                  temperature=temperature,
                                                                                  top_p=top_p,
                                                                                  top_k=top_k,
                                                                                  stop_token_ids=[self.tokenizer.eos_token_id]),
                                                      sequence_start=True,
                                                      sequence_end=True,
                                                      stream_output=stream_output)
            try:
                async for outputs in generator:
                    n_token += len(outputs.token_ids)
                    token_ids += outputs.token_ids
                    if not skip_detokenize:
                        _, state = self.tokenizer.detokenize_incrementally(token_ids, state)
                        # text, state = self.tokenizer.detokenize_incrementally(token_ids, state)
                        # print(text, end='')
                    sess.tick(n_token)
                    # No need to check for cancel_after since we're generating until EOS
                sess.finish(Session.SUCCESS)
                # print()
            finally:
                await generator.aclose()

            # for pytorch engine to restart a session
            if self.backend == 'pytorch':
                await model_inst.async_end(session_id)

            self.pbar.update(1)
            session_id += concurrency

    def process_request(self, requests, profiler: Profiler, concurrency, temperature, top_p, top_k, stream_output,
                        skip_tokenize, skip_detokenize, max_new_tokens):
        req_queue = Queue()

        # feed request to q
        for prompt, input_len in requests:
            # Set a large cancel_after value since we're generating until EOS
            cancel_after = max_new_tokens
            sess = profiler.new_session(input_len, 0)  # Start with 0 output tokens
            req = [prompt, input_len, cancel_after, sess]
            if skip_tokenize:
                req[0] = self.tokenizer.encode(prompt)
            req_queue.put(req)
        for i in range(concurrency):
            req_queue.put(None)

        # start threads
        tasks = []
        for i in range(concurrency):
            task = self._inference(req_queue, i, temperature, top_p, top_k, stream_output, skip_tokenize,
                                   skip_detokenize, concurrency, max_new_tokens)
            tasks.append(task)

        async def _gather_tasks(tasks):
            return await asyncio.gather(*tasks)

        self.pbar = tqdm(total=len(requests))

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        profiler.start()

        asyncio.run(_gather_tasks(tasks))

        profiler.finish()

        self.pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark DLLM request throughput of lmdeploy '
                                     'in localhost',
                                     formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument('dataset', type=str, help='the path dataset')
    parser.add_argument('model_path',
                        type=str,
                        help='the path of model in localhost or '
                        'the repo_id of model in huggingface.co')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        help='Number of working threads to process the sampled prompts',
                        default=256)
    parser.add_argument('-n', '--num-prompts', type=int, help='Number of prompts to process', default=5000)
    parser.add_argument('--no-stream-output', action='store_true', help='Use stream output')
    parser.add_argument('--skip-tokenize', action='store_true', help='Pre-tokenize input prompts before starting')
    parser.add_argument('--skip-detokenize', action='store_true', help='Skip detokenizing output tokens')
    parser.add_argument('--use-uvloop', action='store_true')
    parser.add_argument('--csv', type=str, help='Where to save the result.', default='./profile_throughput_dllm.csv')
    parser.add_argument('--seed', type=int, default=0, help='Seed used in sampling prompts from dataset')
    parser.add_argument('--distributed-executor-backend',
                        type=str,
                        default=None,
                        choices=['uni', 'mp', 'ray'],
                        help='backend of executor backend')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=16384,
        help='Maximum number of new tokens to generate for each request.',
    )
    # other args
    ArgumentHelper.top_p(parser)
    ArgumentHelper.temperature(parser)
    ArgumentHelper.top_k(parser)
    ArgumentHelper.backend(parser)

    # pytorch engine args
    pt_group = parser.add_argument_group('PyTorch engine arguments')
    ArgumentHelper.eager_mode(pt_group)
    ArgumentHelper.dllm_block_length(pt_group)
    ArgumentHelper.dllm_unmasking_strategy(pt_group)
    ArgumentHelper.dllm_denoising_steps(pt_group)
    ArgumentHelper.dllm_confidence_threshold(pt_group)
    ArgumentHelper.dllm_enable_delayed_cache(pt_group)

    tp_act = ArgumentHelper.tp(pt_group)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
    quant_policy_act = ArgumentHelper.quant_policy(pt_group, default=0)
    dtype_act = ArgumentHelper.dtype(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    tb_group._group_actions.append(quant_policy_act)
    tb_group._group_actions.append(dtype_act)

    ArgumentHelper.dp(tb_group)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.num_tokens_per_iter(tb_group)
    ArgumentHelper.max_prefill_iters(tb_group)
    ArgumentHelper.communicator(tb_group)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.backend == 'pytorch', 'only support pytorch backend now'
    random.seed(args.seed)
    if args.backend == 'turbomind':
        engine_config = TurbomindEngineConfig(
            max_batch_size=args.concurrency // args.dp,
            tp=args.tp,
            dp=args.dp,
            cache_max_entry_count=args.cache_max_entry_count,
            cache_block_seq_len=args.cache_block_seq_len,
            model_format=args.model_format,
            quant_policy=args.quant_policy,
            num_tokens_per_iter=args.num_tokens_per_iter,
            max_prefill_iters=args.max_prefill_iters,
            enable_prefix_caching=args.enable_prefix_caching,
            dtype=args.dtype,
            communicator=args.communicator,
        )
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            eager_mode=args.eager_mode,
            enable_prefix_caching=args.enable_prefix_caching,
            quant_policy=args.quant_policy,
            dtype=args.dtype,
            distributed_executor_backend=args.distributed_executor_backend,
            dllm_block_length=args.dllm_block_length,
            dllm_unmasking_strategy=args.dllm_unmasking_strategy,
            dllm_denoising_steps=args.dllm_denoising_steps,
            dllm_confidence_threshold=args.dllm_confidence_threshold,
            dllm_enable_delayed_cache=args.dllm_enable_delayed_cache,
            max_prefill_token_num=args.concurrency * args.dllm_block_length if args.dllm_block_length is not None else 4096,
        )

    if args.use_uvloop:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    engine = Engine(args.model_path, engine_config)

    requests = sample_requests(
        dataset_path=args.dataset,
        num_requests=args.num_prompts,
        tokenizer=engine.tokenizer.model.model,
        chat_template=engine.chat_template,  # Pass the automatically detected chat template
    )

    stream_output = not args.no_stream_output

    profiler = Profiler(stream_output, [50, 75, 95, 99])
    effective_concurrency = args.concurrency if args.concurrency < args.num_prompts else args.num_prompts

    engine.process_request(requests,
                           profiler,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           top_k=args.top_k,
                           concurrency=effective_concurrency,
                           stream_output=not args.no_stream_output,
                           skip_tokenize=args.skip_tokenize,
                           skip_detokenize=args.skip_detokenize,
                           max_new_tokens=args.max_new_tokens)

    # Get and display the used chat template name
    chat_template_name = engine.chat_template_config.model_name
    hyperparams = [('Concurrency', args.concurrency),
                   ('Max new tokens', args.max_new_tokens),
                   ('Stream output', str(stream_output).lower()),
                   ('Skip tokenize', str(args.skip_tokenize).lower()),
                   ('Skip detokenize', str(args.skip_detokenize).lower()),
                   ('Chat template', chat_template_name)]
    total_requests = len(requests)
    completion_target = _full_batch_completion_target(total_requests, effective_concurrency)
    subset_sessions = _earliest_completed_sessions(profiler, completion_target)
    if subset_sessions:
        partial_hyperparams = hyperparams + [('Prompts covered', len(subset_sessions))]
        _summarize_profiler_slice(profiler,
                                  sessions=subset_sessions,
                                  title='Profile LLM Throughput (Full Batches)',
                                  hyperparams=partial_hyperparams)

    profiler.compute_metrics()
    profiler.summarize(title='Profile LLM Throughput', hyperparams=hyperparams)
    if args.csv:
        profiler.save_csv(args.csv, (
            ('backend', args.backend),
            ('bs', args.concurrency),
            ('max_new_tokens', args.max_new_tokens),
            ('num_prompts', args.num_prompts),
            ('chat_template', chat_template_name),
        ))


if __name__ == '__main__':
    main()
