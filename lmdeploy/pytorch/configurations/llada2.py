# Copyright (c) OpenMMLab. All rights reserved.
from .default import AutoModelConfigBuilder, DefaultModelConfigBuilder


class LLaDA2ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['llada2', 'llada2_moe']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        # LLaDA2 is a diffusion LLM and is typically evaluated with
        # `block_length=32, steps=32` (see the official model card). LMDeploy's
        # generic DLLM fallback defaults to `block_length=4`, which can
        # noticeably degrade quality (e.g., HumanEval) if users don't override
        # it via `PytorchEngineConfig.dllm_block_length`.
        if cfg.dllm_block_length is None:
            cfg.dllm_block_length = 32
        cfg.dllm_mask_token = 156895
        cfg.model_paradigm = 'dllm'
        return cfg
