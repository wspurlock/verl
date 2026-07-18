# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest

from verl.trainer.ppo.v1.trainer_base import (
    PPOTrainer,
    _optimizer_training_steps,
    _validate_vapo_resume_settings,
)


def _trainer_at_step(*, step: int, warmup_steps: int, save: bool):
    trainer_type = type(
        "WarmupCheckpointHarness",
        (),
        {"_should_save_value_warmup_checkpoint": PPOTrainer._should_save_value_warmup_checkpoint},
    )
    trainer = trainer_type()
    trainer.global_steps = step
    trainer.config = SimpleNamespace(
        trainer={
            "value_warmup_steps": warmup_steps,
            "save_value_warmup_checkpoint": save,
        }
    )
    return trainer


def test_value_warmup_checkpoint_is_selected_only_at_boundary():
    assert not _trainer_at_step(step=49, warmup_steps=50, save=True)._should_save_value_warmup_checkpoint()
    assert _trainer_at_step(step=50, warmup_steps=50, save=True)._should_save_value_warmup_checkpoint()
    assert not _trainer_at_step(step=51, warmup_steps=50, save=True)._should_save_value_warmup_checkpoint()
    assert not _trainer_at_step(step=50, warmup_steps=50, save=False)._should_save_value_warmup_checkpoint()


def test_optimizer_horizons_exclude_warmup_only_for_actor():
    actor_steps, critic_steps = _optimizer_training_steps(
        total_training_steps=100,
        parameter_sync_step=2,
        value_warmup_steps=50,
    )

    assert actor_steps == 100
    assert critic_steps == 200


def test_vapo_resume_rejects_changed_warmup_or_objective_settings():
    saved = {
        "trainer.value_warmup_steps": 50,
        "algorithm.gae.length_alpha": 0.05,
    }
    current = {
        "trainer.value_warmup_steps": 0,
        "algorithm.gae.length_alpha": 0.1,
    }

    with pytest.raises(ValueError, match="trainer.value_warmup_steps"):
        _validate_vapo_resume_settings(saved, current)


def test_vapo_resume_accepts_identical_settings():
    settings = {
        "trainer.value_warmup_steps": 50,
        "algorithm.gae.length_alpha": 0.05,
    }

    _validate_vapo_resume_settings(settings, settings.copy())
