from opencompass.models import TurboMindModelwithChatTemplate


model_configs = [
    dict(
        abbr='llada2-mini-lmdeploy-focus',
        path='inclusionAI/LLaDA2.0-mini',
        block_length=32,
        confidence_threshold=0.9,
        focus_alpha=1.5,
        eager_mode=False,
        num_gpus=1,
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=64,
        tp=1,
    ),
]

models = []
for cfg in model_configs:
    threshold = cfg['confidence_threshold']
    if 0 < threshold < 1.0:
        unmasking_strategy = 'low_confidence_dynamic'
    elif threshold == 1.0:
        unmasking_strategy = 'low_confidence_static'
    else:
        raise ValueError('confidence_threshold must be in (0, 1].')

    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=cfg['abbr'],
            path=cfg['path'],
            backend='pytorch',
            engine_config=dict(
                tp=cfg['tp'],
                max_batch_size=cfg['batch_size'],
                session_len=cfg['max_seq_len'],
                dllm_block_length=cfg['block_length'],
                dllm_denoising_steps=cfg['block_length'],
                dllm_confidence_threshold=threshold,
                dllm_unmasking_strategy=unmasking_strategy,
                dllm_enable_delayed_cache=False,
                dllm_enable_focus=False,
                dllm_focus_alpha=cfg['focus_alpha'],
                eager_mode=cfg['eager_mode'],
            ),
            gen_config=dict(
                top_p=1.0,
                top_k=0,
                # LLaDA2 diffusion sampling is temperature-sensitive even when
                # `do_sample=False` (unlike standard greedy decoding). Match the
                # HF config for pass@1 stability.
                temperature=0.0,
                do_sample=False,
                max_new_tokens=cfg['max_out_len'],
            ),
            # LLaDA2 chat_template uses a non-special `<|role_end|>` delimiter,
            # so we stop on it to avoid trailing markers in outputs.
            stop_words=['<|role_end|>'],
            include_system_prompt=True,
            system_prompt='You are a helpful assistant.',
            max_seq_len=cfg['max_seq_len'],
            max_out_len=cfg['max_out_len'],
            batch_size=cfg['batch_size'],
            run_cfg=dict(num_gpus=cfg['num_gpus']),
        )
    )
