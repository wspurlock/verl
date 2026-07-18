# Phase 2: Value Pretraining and Calibration

**Prerequisite:** Phase 1 landed and passed all acceptance tests.  
**Scope:** critic-only warmup in v1 sync, calibration metrics, checkpoint boundary, and restart behavior.  
**Out of scope:** positive-example LM loss and final large-scale recipe.

## 1. Objective

Train the critic against Monte Carlo-style targets before the actor begins updating. The rollout policy remains fixed during this period.

Default:

```text
value_warmup_steps = 50
critic_lambda = 1.0
```

An outer training step means one v1 synchronous rollout batch followed by the update phase. Do not reinterpret the default as 50 critic optimizer microsteps or 50 PPO epochs.

## 2. Trainer lifecycle

For global steps in the warmup interval:

1. sample prompts and generate responses normally;
2. compute verifier rewards;
3. compute old/reference log probabilities if required by the existing pipeline;
4. compute critic values;
5. compute critic returns with `critic_lambda=1`;
6. update the critic for its configured PPO epochs;
7. skip the actor update entirely;
8. advance the outer trainer step and logging normally.

The actor optimizer must not step, and actor parameters must not change. The fixed actor continues generating new stochastic samples; "fixed policy" does not mean replaying one static rollout batch.

At the first step after warmup, actor updates begin using length-adaptive actor advantages and the independently computed critic returns.

Use the v1 trainer's existing critic-warmup control if its semantics match this contract. Consolidate or deprecate duplicate warmup keys rather than creating two competing gates.

## 3. Initialization

The critic checkpoint remains configurable through verl's normal critic model/checkpoint path. It may be initialized from:

- the actor/SFT model plus a value head;
- a reward-model-derived checkpoint; or
- a previously saved calibrated value checkpoint.

Do not hard-code a reward-model architecture. Record the resolved critic initialization path in run metadata.

## 4. Explained variance

Implement one masked explained-variance function and use it consistently:

```text
EV = 1 - Var_masked(returns - values) / Var_masked(returns)
```

Requirements:

- response tokens only;
- balancing padding excluded;
- compute over the full rollout batch when feasible;
- numerically safe when target variance is near zero;
- document the near-zero convention;
- detached metric only;
- same implementation during warmup and joint training.

Log:

```text
critic/explained_variance
trainer/value_warmup_active
trainer/value_warmup_step
trainer/value_warmup_remaining
```

Also retain existing value loss, value mean, return mean, and gradient metrics.

The initial implementation uses a fixed number of warmup steps. Metrics do not automatically shorten or extend warmup.

## 5. Warmup checkpoint

When `save_value_warmup_checkpoint=true`, save a checkpoint immediately after the final critic update of warmup and before the first actor update.

The checkpoint must:

- include critic weights and any critic optimizer/scheduler state required for exact continuation;
- include enough trainer metadata to identify the completed warmup boundary;
- use existing checkpoint infrastructure and naming conventions;
- avoid an unsynchronized full-state gather from background threads;
- be loadable as the critic initialization for another run.

Two supported restart intents must be explicit:

1. **Resume the same training run:** restore trainer/global step and do not repeat completed warmup.
2. **Reuse a calibrated critic in a new experiment:** load critic weights as initialization and configure `value_warmup_steps=0`.

Do not infer intent solely from the checkpoint filename.

## 6. Scheduler semantics

Specify and test whether critic optimizer/scheduler step counts include warmup. Recommended behavior:

- critic scheduler begins at the first critic warmup update;
- actor scheduler and optimizer remain at step zero throughout warmup;
- actor learning-rate warmup begins only when actor updates begin;
- total actor training-step calculation excludes critic-only warmup where current scheduler APIs permit it.

Log actor and critic learning rates so this behavior is auditable.

## 7. Unit and integration tests

### 7.1 Actor isolation

Run several warmup steps and assert:

- actor parameter checksums unchanged;
- actor optimizer state unchanged or unconstructed, according to normal lifecycle;
- critic parameters change;
- rollout samples can differ because sampling remains stochastic.

### 7.2 Target semantics

During warmup, critic targets must be the Phase-1 `critic_lambda=1` returns. Changing configured actor policy lambda or length distribution must not change those targets.

### 7.3 Boundary

For `value_warmup_steps=N`:

- exactly N outer steps omit actor updates;
- the next step performs an actor update;
- no off-by-one behavior on fresh start or resume;
- `N=0` exactly preserves ordinary training startup.

### 7.4 Scheduler

Verify actor scheduler is still at its initial step after critic warmup and begins advancing with the first actor update. Verify critic scheduler advances during warmup.

### 7.5 Checkpoint

Save at the boundary, resume, and compare against uninterrupted training under deterministic settings. Separately load the calibrated critic into a new run with warmup disabled and verify identical initial critic outputs on a fixed batch.

### 7.6 Calibration smoke

On a small verifier task, run at least the configured warmup duration. Require:

- finite value loss and explained variance;
- actor entropy/log probabilities stable apart from sampling noise because actor weights are fixed;
- no response-length collapse during warmup;
- critic outputs and returns saved in diagnostics for a small fixed probe batch before and after warmup.

Do not require explained variance to cross an arbitrary universal threshold. Report its trajectory.

## 8. Failure behavior

Startup must reject:

- negative warmup steps;
- warmup without a critic;
- warmup with a non-GAE estimator;
- unsupported async v1 mode;
- an ambiguous resume configuration that would accidentally repeat completed warmup.

