from mmengine.config import read_base

with read_base():
    # from .gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, F403
    from .gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets  # noqa: F401, F403