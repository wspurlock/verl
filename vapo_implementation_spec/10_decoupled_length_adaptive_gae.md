# Phase 1: Decoupled and Length-Adaptive GAE

**Prerequisite:** read `00_design.md` completely.  
**Scope:** typed configuration, pure estimator functions, v1 advantage preparation, metrics, and compatibility tests.  
**Out of scope:** value-warmup lifecycle, positive-example LM loss, and final training recipe.

## 1. Objective

Replace the single shared GAE result in the v1 GAE path with independently computed:

- actor advantages using a scalar or per-sequence policy lambda; and
- critic returns using a scalar critic lambda.

The implementation must retain an exact disabled path for ordinary PPO.

## 2. Estimator API

Implement a low-level masked GAE primitive that can return unnormalized advantages and returns:

```python
raw_advantages, returns = compute_gae_raw(
    token_level_rewards,
    values,
    response_mask,
    gamma,
    lam,
)
```

Required lambda inputs:

- scalar Python float;
- scalar tensor; or
- tensor of shape `[batch]`, broadcast across time.

Do not accept a `[batch, time]` lambda in the initial implementation; VAPO defines one lambda per sequence.

The recurrence must:

- operate backwards over response actions;
- reset correctly at each sequence boundary using `response_mask`;
- exclude padding;
- return zeros outside the response mask;
- avoid gradients through rewards, values, advantages, or returns;
- preserve dtype/device without CPU round trips.

Build the public VAPO calculation from this primitive:

```python
actor_raw_adv, _ = compute_gae_raw(..., lam=policy_lambdas)
_, critic_returns = compute_gae_raw(..., lam=critic_lambda)
actor_advantages = masked_whiten(actor_raw_adv, response_mask)
```

Do not derive `critic_returns` as `whitened_actor_advantages + values`.

## 3. Length calculation

Compute:

```python
lengths = response_mask.sum(dim=-1)
raw_lambdas = 1 - 1 / (length_alpha * lengths)
policy_lambdas = raw_lambdas.clamp(policy_lambda_min, policy_lambda_max)
```

Requirements:

- validate that every sequence has at least one valid response token;
- use floating-point lengths on the same device as values;
- count EOS exactly when it is marked valid by `response_mask`;
- use the observed valid length for truncated responses;
- do not use padded tensor width or configured maximum response length;
- detach lambda from all model graphs.

If length adaptation is disabled, use the configured scalar `policy_lambda`.

## 4. v1 wiring

Modify the v1 advantage-preparation path rather than creating a trainer fork. The v1 path currently:

1. reads values and rewards from the transfer queue;
2. constructs token-level rewards;
3. computes advantages/returns;
4. writes nested results back to the queue.

Preserve that lifecycle. Write only:

- `advantages`: whitened actor advantages;
- `returns`: independently computed critic targets.

Both tensors must use the existing response-nested representation when stored.

The estimator runs once per sampled batch before PPO minibatch updates. It must not run in the actor or critic loss function.

Do not alter GRPO, RLOO, REINFORCE++, or other estimators. VAPO validation must reject any advantage estimator other than GAE.

## 5. Configuration

Add a typed GAE subconfiguration with the semantics from the design document. Backward compatibility:

- when `decoupled=false` and `length_adaptive=false`, preserve the existing `algorithm.lam` behavior;
- do not silently ignore both legacy `algorithm.lam` and new GAE fields when they conflict;
- either define a documented precedence rule with a deprecation warning or reject ambiguous simultaneous overrides.

Recommended rule:

- legacy path uses `algorithm.lam`;
- decoupled path uses `algorithm.gae.policy_lambda` and `critic_lambda`;
- startup validation warns if `algorithm.lam` was explicitly overridden while the decoupled path is enabled.

## 6. Metrics

Record lambda distribution before actor/critic updates:

```text
vapo/policy_lambda_mean
vapo/policy_lambda_min
vapo/policy_lambda_max
vapo/policy_lambda_clamped_low_frac
vapo/policy_lambda_clamped_high_frac
```

Also log masked actor-advantage and critic-return mean/std. Metrics must exclude padding and padded balancing samples.

## 7. Unit tests

### 7.1 Reference recurrence

Use a small, straight-line NumPy or explicit Python reference implementation for ragged sequences. Test:

- sparse terminal reward;
- dense token rewards;
- nonzero values;
- mixed response lengths;
- truncated responses;
- scalar lambda;
- per-sequence lambda;
- gamma values 1.0 and below 1.0.

### 7.2 Monte Carlo critic target

With `gamma=1`, `critic_lambda=1`, terminal reward, and terminal bootstrap zero, every valid position's critic return must equal the undiscounted return-to-go. Include dense rewards to prove it is a cumulative return rather than a copied terminal score.

### 7.3 Decoupling

Changing policy lambda must not change critic returns. Changing critic lambda must not change pre-whitening actor advantages.

### 7.4 Length adaptation

Hand-check lengths around the paper formula's boundary:

- `L=1`;
- `L=19`;
- `L=20`;
- `L=21`;
- one representative long sequence.

With alpha 0.05 and lower clamp 0, `L<=20` must clamp to 0. Verify custom lower/upper bounds and all clamp metrics.

### 7.5 Padding invariance

Appending arbitrary padded columns or changing values/rewards under a zero mask must not change valid advantages, returns, lambdas, or metrics.

### 7.6 Legacy equivalence

When decoupling and length adaptation are disabled, outputs must match the pre-change v1 GAE path within dtype-appropriate tolerance. If deterministic integration infrastructure permits it, compare actor and critic parameter checksums after several steps.

## 8. Integration smoke test

Run v1 synchronous PPO with FSDP2 and vLLM on a small text model:

- GAE enabled;
- `critic_lambda=1`;
- length adaptation enabled;
- actor and critic each update;
- all required metrics finite;
- lambda increases monotonically with observed response length before clamping;
- no NaN/Inf in advantages, returns, losses, or gradients.

The smoke test need not establish a performance improvement.

## 9. Stop conditions

Stop and surface the issue rather than proceeding if:

- v1 cannot carry actor advantages and critic returns as separate transfer-queue fields;
- advantage whitening is inseparable from return construction without changing non-GAE estimators;
- FSDP2 global loss normalization cannot be made microbatch invariant;
- response masks cannot distinguish real repeated trajectories from balancing padding.

