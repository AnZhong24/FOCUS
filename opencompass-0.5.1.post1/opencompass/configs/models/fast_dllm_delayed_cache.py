from opencompass.models import Fast_dLLM_v2_Delayed_Cache


models = [
    dict(
        type=Fast_dLLM_v2_Delayed_Cache,
        abbr='fast-dllm-v2-7b-delayed-cache',
        pkg_root='opencompass/models/Fast_dLLM_v2_7B_Delayed_Cache',
        ckpt_path='opencompass/models/Fast_dLLM_v2_7B_Delayed_Cache',
        max_seq_len=2048,
        max_out_len=2048,
        batch_size=1,
        block_size=32,
        threshold=0.9,
        mask_id=151665,
        device='cuda',
        run_cfg=dict(num_gpus=1),
    )
]