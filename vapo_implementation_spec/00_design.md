# Design: VAPO for verl v1

**Status:** implementation design  
**Algorithm source:** *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks* (arXiv:2504.05118)  
**Target:** verl v1 synchronous PPO, FSDP2 training, vLLM rollout, text-only causal language models  
**Companion specifications:** `10_decoupled_length_adaptive_gae.md`, `20_value_pretraining.md`, `30_positive_lm_loss.md`, and `40_recipe_and_validation.md`

## 1. Purpose and scope

This specification defines a clean-room VAPO implementation from the public paper. VAPO remains actor-critic PPO. It combines:

1. value-model pretraining;
2. decoupled actor and critic GAE;
3. length-adaptive GAE for actor advantages;
4. token-level policy-loss aggregation;
5. asymmetric PPO clipping ("Clip-Higher");
6. a positive-example language-model loss; and
7. repeated sampling from each prompt.

The implementation must preserve ordinary PPO behavior when all VAPO-specific options are disabled. It must not create a separate trainer that copies the PPO loop. Extend the shared v1 PPO path through small, independently testable primitives and configuration.

This initial implementation deliberately supports one execution stack:

- `trainer.use_v1=true`;
- `trainer.v1.trainer_mode=sync`;
- actor, reference, and critic training engine `fsdp2`;
- vLLM rollout;
- text-only causal language models;
- single-turn verifier-scored trajectories.

If VAPO is enabled with an unsupported trainer mode, model engine, rollout backend, or modality, startup must fail with a clear error. Do not silently run a partially supported configuration. Algorithm-level tensor functions should remain backend-neutral where practical, but support for other engines is not an acceptance requirement.

## 2. Scientific objective and non-goals

The objective is to reproduce the algorithm described in the paper closely enough to run controlled ablations, not to claim reproduction of the reported AIME score. The paper does not publish all data, reward, checkpoint, and operational details.

Non-goals for the initial implementation:

- asynchronous or off-policy v1 trainer modes;
- Megatron, VeOmni, TorchTitan, or other training engines;
- SGLang or other rollout backends;
- multi-turn tool trajectories;
- visual-language models;
- automatic hyperparameter tuning;
- automatic early stopping of value pretraining;
- changing PPO's KL formulation beyond existing verl configuration;
- group-relative advantage normalization.

Repeated prompt sampling is a rollout allocation strategy only. VAPO remains value-based PPO and must not acquire GRPO-style group normalization.

## 3. Algorithm

### 3.1 Rewards and values

Let `r_ext[i,t]` be verl's existing token-level PPO reward, including the terminal verifier score and, if configured, the existing KL-in-reward term. Let `V[i,t]` be the extrinsic critic value. This implementation adds no new reward channel.

All reductions and estimators operate over valid response tokens only. Prompt tokens and padding are excluded.

### 3.2 Decoupled GAE

VAPO uses different GAE calculations for actor and critic:

```text
actor_advantages = GAE-advantages(r_ext, V, gamma, policy_lambda)
critic_returns   = GAE-returns(r_ext, V, gamma, critic_lambda)
```

The paper uses `critic_lambda=1.0`. Actor advantages and critic returns must be computed independently; critic returns must never be reconstructed from already-whitened actor advantages.

Only actor advantages are whitened using verl's existing masked convention. Critic returns remain in reward/value units.

### 3.3 Length-adaptive actor lambda

For sequence `i` with `L_i` valid response tokens:

```text
raw_policy_lambda_i = 1 - 1 / (length_alpha * L_i)
policy_lambda_i = clamp(raw_policy_lambda_i, policy_lambda_min, policy_lambda_max)
```

Paper default:

```text
length_alpha = 0.05
policy_lambda_min = 0.0
policy_lambda_max = 1.0
```

Clamping is an explicit implementation decision because the paper's formula produces non-positive lambda for responses of length 20 or less and does not document edge handling. `L_i` counts valid response actions, including EOS when EOS is part of the response mask. Truncated responses use their number of valid generated actions. Empty responses are invalid input and must fail before division.

Length adaptation applies only to actor advantages. The critic uses its configured scalar lambda, default 1.0.

### 3.4 Actor objective

The actor objective is:

```text
L_actor = L_PPO + positive_lm_coef * L_positive_lm
```

The PPO term uses:

- token-level global mean aggregation;
- lower clip ratio 0.20;
- upper clip ratio 0.28;
- the length-adaptive actor advantages.

The positive-example loss is the negative log-likelihood of response tokens belonging to correct sampled trajectories:

```text
L_positive_lm =
    -sum(correct_i * response_mask[i,t] * log_pi[i,t])
     / sum(correct_i * response_mask[i,t])
```

The numerator and denominator are global across the data-parallel training group. If a global actor minibatch has no positive tokens, this term is exactly zero and produces no gradient.

Correctness must come from an explicitly configured verifier output field. It must not be inferred from KL-shaped reward, length penalties, or arbitrary total reward.

### 3.5 Value pretraining

For the first `value_warmup_steps` outer synchronous PPO iterations:

- generate trajectories from the fixed initial actor;
- compute rewards and critic values normally;
- construct critic targets with `critic_lambda=1.0`;
- update only the critic;
- do not update actor parameters;
- continue synchronizing the unchanged actor to rollout replicas only as required by the existing v1 lifecycle.

Default:

```text
value_warmup_steps = 50
```

This matches the paper's reported warmup length. The implementation must log value loss and masked explained variance throughout warmup and joint training. A checkpoint at warmup completion must be optionally saveable and reusable as the initial critic checkpoint for later experiments.

### 3.6 Repeated prompt sampling

Paper recipe:

```text
unique prompts per outer batch = 512
rollout.n = 16
generated trajectories = 8192
actor PPO minibatch size = 512
critic PPO minibatch size = 512
```

The exact interpretation of global minibatch size must follow current verl configuration semantics. The recipe must not multiply a user-specified global minibatch size twice.

All trajectories derived from one prompt retain the same prompt UID. Each generated trajectory remains a separate actor-critic sample. Batch balancing, transfer-queue storage, and minibatch construction must not accidentally drop or overwrite repeated trajectories.

## 4. Configuration contract

Use typed configuration rather than scattered unstructured `get()` calls. Names may be adjusted to match repository conventions, but the following semantics are required:

```yaml
algorithm:
  adv_estimator: gae
  gamma: 1.0
  gae:
    decoupled: false
    critic_lambda: 1.0
    policy_lambda: 0.95
    length_adaptive: false
    length_alpha: 0.05
    policy_lambda_min: 0.0
    policy_lambda_max: 1.0

actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.2
    loss_agg_mode: token-mean
    positive_lm:
      enabled: false
      coef: 0.1
      correctness_key: acc
      correctness_threshold: 1.0

trainer:
  value_warmup_steps: 0
  save_value_warmup_checkpoint: false
```

The VAPO recipe enables decoupling, length adaptation, positive LM loss, Clip-Higher, and 50 warmup steps. Generic PPO defaults remain behavior-preserving.

Validation requirements:

- `gamma` in `[0,1]`;
- all lambda bounds in `[0,1]`;
- `policy_lambda_min <= policy_lambda_max`;
- `length_alpha > 0`;
- `value_warmup_steps >= 0`;
- `positive_lm.coef >= 0`;
- non-empty correctness key when positive LM is enabled;
- finite correctness threshold;
- VAPO requires GAE and a critic;
- VAPO initial support requires v1 sync + FSDP2 + vLLM.

Do not introduce a single opaque `algorithm=vapo` branch that bypasses typed validation. A convenience `vapo.enabled` flag or recipe is acceptable if it expands into explicit settings and validation.

## 5. Invariants

These invariants are load-bearing:

1. **Disabled equivalence:** with new options disabled, actor advantages, critic returns, losses, and parameter updates match the pre-change v1 PPO path.
2. **Target separation:** actor advantage normalization cannot affect critic returns.
3. **Masking:** prompt and padding tokens never enter lambda length, loss denominators, whitening, or explained variance.
4. **Global normalization:** token-mean PPO and positive LM losses are invariant to data-parallel rank count and microbatch partitioning.
5. **Correctness provenance:** positive examples are selected only from the configured verifier field.
6. **Warmup isolation:** actor parameters and optimizer state do not change during value warmup.
7. **Synchronous on-policy operation:** initial support must reject async trainer modes rather than relying on unspecified policy-staleness behavior.
8. **No group normalization:** repeated samples do not change GAE into a group-relative estimator.
9. **One computation per rollout batch:** advantages and returns are fixed before PPO epochs and are not recomputed inside actor or critic minibatches.

## 6. Required metrics

Log at least:

```text
vapo/policy_lambda_mean
vapo/policy_lambda_min
vapo/policy_lambda_max
vapo/policy_lambda_clamped_low_frac
vapo/policy_lambda_clamped_high_frac
vapo/actor_advantage_mean
vapo/actor_advantage_std
vapo/critic_return_mean
vapo/critic_return_std
critic/explained_variance
trainer/value_warmup_active
actor/positive_lm_loss
actor/positive_sequence_fraction
actor/positive_token_count
actor/positive_lm_coef
actor/pg_loss
actor/entropy
actor/ppo_kl
actor/pg_clipfrac
response_length/mean
response_length/p95
response_length/max_fraction
```

Use existing metric names where they already carry identical semantics. Do not duplicate an existing metric under a second name merely to add a VAPO prefix.

## 7. Implementation sequence

Implementation is staged:

1. `10_decoupled_length_adaptive_gae.md`
2. `20_value_pretraining.md`
3. `30_positive_lm_loss.md`
4. `40_recipe_and_validation.md`

Each phase must pass its acceptance tests before the next begins. If the current branch's v1 or FSDP2 interfaces differ from the locations described in a phase, adapt to the current abstraction rather than copying legacy trainer code.

## 8. Experimental interpretation

The paper's ablations suggest value pretraining and decoupled GAE are the most important components, followed by length adaptation. Implementation validation must therefore emphasize value-target correctness and training stability before tuning positive LM loss or sampling allocation.

The first scientific comparison should include:

- unchanged v1 PPO;
- PPO with configuration-only token mean and Clip-Higher;
- decoupled GAE;
- decoupled plus length-adaptive GAE;
- value-pretrained VAPO without positive LM;
- full VAPO;
- full VAPO with `rollout.n=1`, holding generated trajectory or token budget approximately fixed.

Never compare runs with different total generated tokens without labeling the compute difference.

