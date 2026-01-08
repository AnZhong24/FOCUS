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
        cfg.model_paradigm = 'dllm'
        mask_token = getattr(hf_config, 'mask_token_id', None)
        if mask_token is None:
            mask_token = 156895
        cfg.dllm_mask_token = mask_token
        return cfg
