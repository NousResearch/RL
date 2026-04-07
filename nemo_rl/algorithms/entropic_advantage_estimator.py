"""Entropic Adaptive-Beta Advantage Estimator for TTT-Discover.

Implements the Leave-One-Out (LOO) entropic advantage from
"Learning to Discover at Test Time" (arXiv:2601.16175).

Instead of standard group-relative advantages (Adv = R - mean(R)),
this estimator:
  1. Solves for β such that KL(softmax_β(R) || uniform) = γ  (default γ = ln(2))
  2. Computes LOO advantages: w_i = exp(β·r_i) / Z_{-i} - 1
     where Z_{-i} is the normalizer excluding the i-th sample.

Properties:
  - Shift-invariant, approximately scale-invariant
  - Monotone in reward
  - Approximately mean-zero
  - Adaptive scaling via β solves the reward-scale sensitivity of standard GRPO
"""

import math
from typing import Optional

import torch


def _solve_beta(
    rewards: torch.Tensor,
    gamma: float = math.log(2),
    max_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """Solve for β such that KL(softmax_β(R) || uniform) = γ via bisection.

    Args:
        rewards: [K] tensor of rewards for one group.
        gamma: Target KL divergence. Default ln(2) as in the paper.
        max_iter: Maximum bisection iterations.
        tol: Convergence tolerance on β.

    Returns:
        Scalar β value.
    """
    K = rewards.shape[0]
    if K <= 1:
        return 0.0

    log_K = math.log(K)
    r = rewards.double()
    r_max = r.max()

    def kl_at_beta(b: float) -> float:
        logits = b * (r - r_max)
        log_Z = torch.logsumexp(logits, dim=0)
        logq = logits - log_Z
        q = logq.exp()
        kl = (q * (logq + log_K)).sum().item()
        return kl

    # Bisect: KL is monotonically increasing in |β| for non-constant rewards
    # Find upper bound for β
    lo, hi = 0.0, 1.0
    while kl_at_beta(hi) < gamma and hi < 1e8:
        hi *= 2.0

    # Edge case: all rewards identical → β = 0, KL = 0 for any β
    if hi >= 1e8:
        return 0.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        kl = kl_at_beta(mid)
        if abs(kl - gamma) < tol:
            return mid
        if kl < gamma:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def compute_entropic_advantages(
    rewards: torch.Tensor,
    gamma: float = math.log(2),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute LOO entropic advantages for a group of rewards.

    Args:
        rewards: [K] tensor of rewards for one group.
        gamma: Target KL for adaptive β.
        eps: Small constant for numerical stability.

    Returns:
        [K] tensor of advantages.
    """
    K = rewards.shape[0]
    if K <= 1:
        return torch.zeros_like(rewards)

    beta = _solve_beta(rewards, gamma=gamma)
    if beta == 0.0:
        return torch.zeros_like(rewards)

    r = rewards.double()
    r_max = r.max()
    e = torch.exp(beta * (r - r_max))

    if K == 1:
        Z_loo = e
    else:
        # Leave-one-out normalizer: Z_{-i} = (sum(e) - e_i) / (K - 1)
        Z_loo = (e.sum() - e) / (K - 1)

    w = e / (Z_loo + eps)
    advantages = (w - 1.0).to(rewards.dtype)
    return advantages


class EntropicAdaptiveBetaAdvantageEstimator:
    """Advantage estimator using entropic adaptive-β LOO weighting.

    Follows the same interface as GRPOAdvantageEstimator:
        compute_advantage(prompt_ids, rewards, mask, **kwargs) -> [B, S] tensor

    Config keys (under grpo.adv_estimator):
        gamma: Target KL for β search. Default ln(2) ≈ 0.693.
        eps: Numerical stability constant. Default 1e-8.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.gamma = estimator_config.get("gamma", math.log(2))
        self.eps = estimator_config.get("eps", 1e-8)

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute per-token advantages using entropic adaptive-β LOO.

        Args:
            prompt_ids: [B] or [B, S] tensor identifying which prompt each
                        sample belongs to (same prompt = same group).
            rewards: [B] scalar rewards per sample.
            mask: [B, S] response token mask (1 = generation token).

        Returns:
            [B, S] advantages tensor. Each generation token gets the
            sample-level advantage; non-generation tokens get 0.
        """
        batch_size, seq_len = mask.shape
        advantages = torch.zeros_like(mask, dtype=rewards.dtype)

        # Group by prompt (same as GRPO's per-prompt baseline)
        if prompt_ids.dim() > 1:
            # prompt_ids is [B, S] — use first token as group key
            group_ids = prompt_ids[:, 0]
        else:
            group_ids = prompt_ids

        unique_prompts = group_ids.unique()

        for pid in unique_prompts:
            group_mask = group_ids == pid
            group_rewards = rewards[group_mask]

            group_adv = compute_entropic_advantages(
                group_rewards, gamma=self.gamma, eps=self.eps
            )

            # Expand sample-level advantages to [group_size, seq_len]
            # and mask to generation tokens only
            group_indices = group_mask.nonzero(as_tuple=True)[0]
            for i, idx in enumerate(group_indices):
                advantages[idx] = group_adv[i] * mask[idx]

        return advantages
