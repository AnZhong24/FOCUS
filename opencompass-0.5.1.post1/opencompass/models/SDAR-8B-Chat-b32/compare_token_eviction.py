import argparse
from typing import Tuple

import torch


def select_retain_indices(
    delta_importance: torch.Tensor,
    K: int,
    strategy: str,
    device: torch.device,
) -> torch.Tensor:
    """Replicates the retain_indices selection logic from modeling_sdar."""
    if strategy == "dynamic":
        mean_importance = torch.mean(delta_importance)
        std_importance = torch.std(delta_importance)
        threshold = mean_importance + std_importance
        outlier_mask = delta_importance >= threshold
        outlier_indices = torch.nonzero(outlier_mask, as_tuple=True)[1]
        if len(outlier_indices) < K:
            sorted_indices = torch.argsort(
                delta_importance, stable=True, descending=True
            )
            retain_indices = sorted_indices[0, :K]
        else:
            retain_indices = outlier_indices
    elif strategy == "top":
        sorted_indices = torch.argsort(
            delta_importance, stable=True, descending=True
        )
        retain_indices = sorted_indices[0, :K]
    elif strategy == "bottom":
        sorted_indices = torch.argsort(
            delta_importance, stable=True, descending=False
        )
        retain_indices = sorted_indices[0, :K]
    elif strategy == "random":
        sorted_indices = torch.randperm(delta_importance.shape[1], device=device)
        retain_indices = sorted_indices[:K]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return retain_indices


def eviction_logic_new(
    delta_importance: torch.Tensor,
    mask_indices: torch.Tensor,
    processing_indices: torch.Tensor,
    unprocessed_positions: torch.Tensor,
    K: int,
    strategy: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token eviction logic from `modeling_sdar.py`."""
    # context.mask_indices[0, context.processing_indices]
    seq_mask = mask_indices[0, processing_indices]
    masked_positions = torch.nonzero(seq_mask, as_tuple=True)[0]
    if masked_positions.numel() == 0:
        raise ValueError("No masked positions in processing_indices.")

    assert (
        delta_importance.shape[1] == masked_positions.numel()
    ), "delta_importance must match number of masked positions"

    retain_indices = select_retain_indices(delta_importance, K, strategy, device)

    retain_mask = torch.zeros_like(masked_positions, dtype=torch.bool, device=device)
    retain_mask[retain_indices] = True

    adjacent_and_right_selected = (
        (masked_positions[1:] - masked_positions[:-1]) == 1
    ) & retain_mask[1:] & ~retain_mask[:-1]
    retain_mask[:-1][adjacent_and_right_selected] = True

    rightmost_retain_idx = masked_positions.masked_fill(~retain_mask, -1).max()
    evicted_before_rightmost = (masked_positions < rightmost_retain_idx) & ~retain_mask
    if evicted_before_rightmost.any():
        # context.unprocessed_positions[context.processing_indices[masked_positions]]
        unprocessed_mask = unprocessed_positions[processing_indices[masked_positions]]
        reinstate_mask = evicted_before_rightmost & unprocessed_mask
        if reinstate_mask.any():
            retain_mask |= reinstate_mask

    processing_mask = torch.ones_like(
        processing_indices, dtype=torch.bool, device=device
    )
    processing_mask[masked_positions] = retain_mask
    retained_processing_indices = processing_indices[processing_mask]

    return retain_mask, processing_mask, retained_processing_indices


def eviction_logic_backup(
    delta_importance: torch.Tensor,
    mask_indices: torch.Tensor,
    processing_indices: torch.Tensor,
    unprocessed_positions: torch.Tensor,
    K: int,
    strategy: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token eviction logic from `modeling_sdar_backup.py`."""
    seq_mask = mask_indices[0, processing_indices]
    masked_positions = torch.nonzero(seq_mask, as_tuple=True)[0]
    if masked_positions.numel() == 0:
        raise ValueError("No masked positions in processing_indices.")

    assert (
        delta_importance.shape[1] == masked_positions.numel()
    ), "delta_importance must match number of masked positions"

    retain_indices = select_retain_indices(delta_importance, K, strategy, device)

    retain_mask = torch.zeros_like(masked_positions, dtype=torch.bool, device=device)
    retain_mask[retain_indices] = True

    adjacent_and_right_selected = (
        (masked_positions[1:] - masked_positions[:-1]) == 1
    ) & retain_mask[1:] & ~retain_mask[:-1]
    retain_mask[:-1][adjacent_and_right_selected] = True

    # Old logic for reinstating unprocessed evicted tokens
    rightmost_retain_idx = masked_positions[
        torch.nonzero(retain_mask, as_tuple=True)[0][-1]
    ]
    evicted_before_rightmost = (masked_positions < rightmost_retain_idx) & ~retain_mask
    evicted_before_rightmost_positions = processing_indices[
        masked_positions[evicted_before_rightmost]
    ]
    unprocessed_evicted_mask = unprocessed_positions[evicted_before_rightmost_positions]

    if unprocessed_evicted_mask.any():
        unprocessed_evicted_indices = torch.nonzero(
            evicted_before_rightmost
            & torch.isin(
                processing_indices[masked_positions],
                evicted_before_rightmost_positions[unprocessed_evicted_mask],
            ),
            as_tuple=True,
        )[0]
        retain_mask[unprocessed_evicted_indices] = True

    evict_token_indices = processing_indices[masked_positions[~retain_mask]]

    retain_block_mask = torch.zeros_like(
        mask_indices[0], dtype=torch.bool, device=device
    )
    retain_block_mask[processing_indices] = True
    retain_block_mask[evict_token_indices] = False

    processing_mask = torch.ones_like(
        processing_indices, dtype=torch.bool, device=device
    )
    processing_mask[masked_positions[~retain_mask]] = False

    new_processing_indices = torch.nonzero(retain_block_mask, as_tuple=True)[0]

    return retain_mask, processing_mask, new_processing_indices


def generate_scenario(
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Generate a random scenario consistent with the SDAR context assumptions."""
    while True:
        # Random mask pattern over the block
        mask_indices = torch.rand(1, seq_len, device=device) < 0.5
        total_masked = int(mask_indices.sum().item())
        if total_masked < 2:
            continue

        # Random subset of positions to process this step
        processing_mask = torch.rand(seq_len, device=device) < 0.8
        processing_indices = torch.nonzero(processing_mask, as_tuple=True)[0]
        if processing_indices.numel() == 0:
            continue

        seq_mask = mask_indices[0, processing_indices]
        masked_positions = torch.nonzero(seq_mask, as_tuple=True)[0]
        num_masked_in_processing = masked_positions.numel()
        if num_masked_in_processing < 2:
            continue

        # Choose K so that token_evicting condition mask_indices.sum() > K holds
        max_k = num_masked_in_processing - 1
        if max_k < 1:
            continue
        K = int(torch.randint(1, max_k + 1, (1,), device=device).item())
        if total_masked <= K:
            continue

        # Random unprocessed positions over the block
        unprocessed_positions = torch.rand(seq_len, device=device) < 0.5

        # Delta importance for masked tokens
        delta_importance = torch.randn(1, num_masked_in_processing, device=device)

        return (
            mask_indices,
            processing_indices,
            unprocessed_positions,
            delta_importance,
            K,
        )


def run_tests(
    num_tests: int = 100,
    seq_len: int = 64,
    device: torch.device | None = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategies = ["top", "bottom", "random", "dynamic"]

    for strategy in strategies:
        for i in range(num_tests):
            (
                mask_indices,
                processing_indices,
                unprocessed_positions,
                delta_importance,
                K,
            ) = generate_scenario(seq_len, device)

            if strategy == "random":
                torch.manual_seed(0)
            new_retain, new_proc_mask, new_proc_idx = eviction_logic_new(
                delta_importance,
                mask_indices,
                processing_indices,
                unprocessed_positions,
                K,
                strategy,
                device,
            )

            if strategy == "random":
                torch.manual_seed(0)
            old_retain, old_proc_mask, old_proc_idx = eviction_logic_backup(
                delta_importance,
                mask_indices,
                processing_indices,
                unprocessed_positions,
                K,
                strategy,
                device,
            )

            if not torch.equal(new_proc_mask, old_proc_mask) or not torch.equal(
                new_proc_idx, old_proc_idx
            ):
                print(f"Mismatch detected for strategy='{strategy}' at test #{i + 1}")
                print(f"mask_indices: {mask_indices}")
                print(f"processing_indices: {processing_indices}")
                print(f"unprocessed_positions: {unprocessed_positions}")
                print(f"delta_importance: {delta_importance}")
                print(f"K: {K}")
                print(f"new_retain_mask: {new_retain}")
                print(f"old_retain_mask: {old_retain}")
                print(f"new_processing_mask: {new_proc_mask}")
                print(f"old_processing_mask: {old_proc_mask}")
                print(f"new_processing_indices: {new_proc_idx}")
                print(f"old_processing_indices: {old_proc_idx}")
                raise SystemExit(1)

        print(f"Strategy '{strategy}': all {num_tests} tests passed.")

    print("All strategies passed all tests.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare token eviction logic between modeling_sdar.py and "
            "modeling_sdar_backup.py using randomized scenarios."
        )
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=100,
        help="Number of random test cases per strategy.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length to use when generating random scenarios.",
    )
    args = parser.parse_args()

    run_tests(num_tests=args.num_tests, seq_len=args.seq_len)


if __name__ == "__main__":
    main()

