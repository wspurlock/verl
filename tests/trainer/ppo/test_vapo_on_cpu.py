# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_decoupled_gae,
    compute_gae_advantage_return,
    compute_gae_raw,
    compute_length_adaptive_lambdas,
    compute_vapo_data_metrics,
    masked_explained_variance,
    positive_lm_loss,
)
from verl.utils.torch_functional import masked_whiten


def test_gae_raw_ragged_monte_carlo_returns():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 99.0, 99.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    advantages, returns = compute_gae_raw(rewards, values, mask, gamma=1.0, lam=1.0)

    torch.testing.assert_close(returns, torch.tensor([[6.0, 5.0, 3.0, 0.0], [9.0, 5.0, 0.0, 0.0]]))
    torch.testing.assert_close(advantages, returns)


def test_gae_raw_per_sequence_lambda_and_padding_invariance():
    rewards = torch.tensor([[0.0, 1.0, 7.0], [0.0, 0.0, 1.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    lambdas = torch.tensor([0.0, 1.0])

    advantages, returns = compute_gae_raw(rewards, values, mask, gamma=1.0, lam=lambdas)

    torch.testing.assert_close(advantages, torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]))
    torch.testing.assert_close(returns, advantages)


def test_raw_gae_matches_legacy_scalar_path():
    torch.manual_seed(7)
    rewards = torch.randn(3, 5)
    values = torch.randn(3, 5)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    legacy_advantages, legacy_returns = compute_gae_advantage_return(rewards, values, mask, 0.97, 0.91)
    raw_advantages, returns = compute_gae_raw(rewards, values, mask, 0.97, 0.91)

    torch.testing.assert_close(returns, legacy_returns * mask)
    torch.testing.assert_close(masked_whiten(raw_advantages, mask), legacy_advantages)


def test_decoupled_targets_do_not_depend_on_policy_lambda():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.1, 0.2, 0.3]])
    mask = torch.ones_like(rewards)

    _, returns_low, raw_low = compute_decoupled_gae(rewards, values, mask, 1.0, 0.0, 1.0)
    _, returns_high, raw_high = compute_decoupled_gae(rewards, values, mask, 1.0, 1.0, 1.0)

    torch.testing.assert_close(returns_low, returns_high)
    assert not torch.allclose(raw_low, raw_high)


def test_length_adaptation_boundary_and_metrics():
    lengths = [1, 19, 20, 21, 100]
    mask = torch.zeros((len(lengths), max(lengths)))
    for row, length in enumerate(lengths):
        mask[row, :length] = 1

    lambdas, metrics = compute_length_adaptive_lambdas(mask)

    torch.testing.assert_close(lambdas, torch.tensor([0.0, 0.0, 0.0, 1 - 1 / 1.05, 0.8]))
    assert metrics["vapo/policy_lambda_clamped_low_frac"] == pytest.approx(3 / 5)
    assert metrics["vapo/policy_lambda_max"] == pytest.approx(0.8)


def test_empty_response_rejected():
    with pytest.raises(ValueError, match="at least one"):
        compute_length_adaptive_lambdas(torch.tensor([[1, 0], [0, 0]]))


def test_masked_explained_variance_and_constant_convention():
    returns = torch.tensor([[1.0, 2.0, 100.0]])
    values = torch.tensor([[1.0, 2.0, -100.0]])
    mask = torch.tensor([[1, 1, 0]])
    assert masked_explained_variance(returns, values, mask).item() == pytest.approx(1.0)
    assert masked_explained_variance(torch.ones_like(returns), values, mask).item() == 0.0


def test_vapo_data_metrics_exclude_balancing_padding():
    advantages = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    returns = torch.tensor([[2.0, 6.0], [0.0, 0.0]])
    response_mask = torch.ones_like(advantages)
    real_sample_mask = torch.tensor([True, False])

    metrics = compute_vapo_data_metrics(advantages, returns, response_mask, real_sample_mask)

    assert metrics["vapo/actor_advantage_mean"] == pytest.approx(2.0)
    assert metrics["vapo/actor_advantage_std"] == pytest.approx(1.0)
    assert metrics["vapo/critic_return_mean"] == pytest.approx(4.0)
    assert metrics["vapo/critic_return_std"] == pytest.approx(2.0)


def test_positive_lm_loss_formula_and_masking():
    log_prob = torch.tensor([[-1.0, -2.0, -30.0], [-4.0, -5.0, -60.0]], requires_grad=True)
    positive_mask = torch.tensor([[1, 1, 0], [0, 0, 0]])

    loss = positive_lm_loss(log_prob, positive_mask, global_positive_token_count=2)

    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.tensor([[-0.5, -0.5, 0.0], [0.0, 0.0, 0.0]]))


def test_positive_lm_loss_empty_is_differentiable_zero():
    log_prob = torch.randn(2, 3, requires_grad=True)

    loss = positive_lm_loss(log_prob, torch.zeros_like(log_prob), global_positive_token_count=0)

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.zeros_like(log_prob))


def test_positive_lm_loss_matches_single_rank_when_partitioned():
    full_log_prob = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    full_mask = torch.tensor([1, 0, 1, 1])
    full_loss = positive_lm_loss(full_log_prob, full_mask, global_positive_token_count=3)
    full_loss.backward()

    rank_log_probs = [
        torch.tensor([-1.0, -2.0], requires_grad=True),
        torch.tensor([-3.0, -4.0], requires_grad=True),
    ]
    rank_masks = [torch.tensor([1, 0]), torch.tensor([1, 1])]
    rank_losses = [
        positive_lm_loss(log_prob, mask, global_positive_token_count=3, dp_size=2)
        for log_prob, mask in zip(rank_log_probs, rank_masks, strict=True)
    ]
    # DDP averages rank gradients, so average the independently computed rank
    # losses to model its reduction.
    distributed_loss = sum(rank_losses) / 2
    distributed_loss.backward()

    assert distributed_loss.item() == pytest.approx(full_loss.item())
    torch.testing.assert_close(
        torch.cat([log_prob.grad for log_prob in rank_log_probs]),
        full_log_prob.grad,
    )
