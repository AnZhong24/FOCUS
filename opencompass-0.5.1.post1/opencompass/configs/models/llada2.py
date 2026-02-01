from opencompass.models import LLaDA2


models = [
    dict(
        type=LLaDA2,
        abbr="llada2-mini",
        path="opencompass/models/LLaDA2.0-mini",
        hf_repo="inclusionAI/LLaDA2.0-mini",
        max_seq_len=1024,
        max_out_len=1024,
        batch_size=1,
        block_length=32,
        steps=32,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.8,
        use_block_cache=True,
        # FOCUS token-eviction selection strategy.
        # Options: "none", "dynamic", "top", "bottom", "random".
        # - "none": disable token eviction (baseline; no eviction step)
        # - "top"/"bottom": keep K tokens with highest/lowest scores
        # - "random": keep K random tokens
        # - "dynamic": default FOCUS rule
        strategy="dynamic",
        # K=8,
        alpha=1.5,
        mask_id=156895,
        eos_id=156892,
        device="cuda",
        dtype="bfloat16",
        run_cfg=dict(num_gpus=1),
    )
]
