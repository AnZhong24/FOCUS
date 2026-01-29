from opencompass.models import Fast_dLLM_v2


models = [
    dict(
        type=Fast_dLLM_v2,
        abbr='fast-dllm-v2-7b',
        pkg_root='opencompass/models/Fast_dLLM_v2_7B',
        ckpt_path='opencompass/models/Fast_dLLM_v2_7B',
        max_seq_len=2048,
        max_out_len=2048,
        batch_size=1,
        block_size=32,
        small_block_size=32,
        threshold=0.9,
        mask_id=151665,
        use_block_cache=False,
        device='cuda',
        run_cfg=dict(num_gpus=1),
    )
]