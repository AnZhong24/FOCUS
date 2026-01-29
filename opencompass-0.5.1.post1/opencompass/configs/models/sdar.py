from opencompass.models import SDARFocus


models = [
    dict(
        type=SDARFocus,
        abbr='sdar-focus-8b',
        path='opencompass/models/SDAR-8B-Chat-b32',
        hf_repo='JetLM/SDAR-8B-Chat-b32',
        max_seq_len=1024,
        max_out_len=1024,
        batch_size=1,
        block_length=32,
        denoising_steps=32,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        mask_id=151669,
        confidence_threshold=0.8,
        use_block_cache=True,
        device='cuda',
        strategy='dynamic',
        # K=8,
        alpha=1.5,
        run_cfg=dict(num_gpus=1),
    )
]
