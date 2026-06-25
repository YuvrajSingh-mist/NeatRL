"""Smoke tests: verify all public train functions are importable and Config instantiates."""
import dataclasses
import importlib

import pytest


def test_all_train_functions_importable():
    from neatrl import (
        train_a2c,
        train_a2c_cnn,
        train_ddpg,
        train_ddpg_cnn,
        train_dqn,
        train_dueling_dqn,
        train_ppo,
        train_ppo_cnn,
        train_ppo_rnd,
        train_ppo_rnd_cnn,
        train_reinforce,
        train_reinforce_cnn,
        train_sac,
        train_sac_cnn,
        train_td3,
        train_td3_cnn,
    )

    for fn in [
        train_a2c, train_a2c_cnn,
        train_ddpg, train_ddpg_cnn,
        train_dqn,
        train_dueling_dqn,
        train_ppo, train_ppo_cnn,
        train_ppo_rnd, train_ppo_rnd_cnn,
        train_reinforce, train_reinforce_cnn,
        train_sac, train_sac_cnn,
        train_td3, train_td3_cnn,
    ]:
        assert callable(fn), f"{fn.__name__} is not callable"


@pytest.mark.parametrize("module_name", [
    "neatrl.a2c",
    "neatrl.ddpg",
    "neatrl.dqn",
    "neatrl.dueling_dqn",
    "neatrl.ppo",
    "neatrl.reinforce",
    "neatrl.rnd",
    "neatrl.sac",
    "neatrl.td3",
])
def test_unsuffixed_algorithm_modules_are_removed(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", [
    "neatrl.a2c_mlp",
    "neatrl.a2c_cnn",
    "neatrl.ddpg_mlp",
    "neatrl.ddpg_cnn",
    "neatrl.dqn_mlp",
    "neatrl.dueling_dqn_mlp",
    "neatrl.ppo_mlp",
    "neatrl.ppo_cnn",
    "neatrl.reinforce_mlp",
    "neatrl.reinforce_cnn",
    "neatrl.rnd_mlp",
    "neatrl.rnd_cnn",
    "neatrl.sac_mlp",
    "neatrl.sac_cnn",
    "neatrl.td3_mlp",
    "neatrl.td3_cnn",
])
def test_config_is_dataclass(module_name):
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "Config"), f"{module_name} has no Config class"
    assert dataclasses.is_dataclass(mod.Config), f"{module_name}.Config is not a dataclass"


@pytest.mark.parametrize("module_name", [
    "neatrl.a2c_mlp",
    "neatrl.a2c_cnn",
    "neatrl.ddpg_mlp",
    "neatrl.ddpg_cnn",
    "neatrl.dqn_mlp",
    "neatrl.dueling_dqn_mlp",
    "neatrl.ppo_mlp",
    "neatrl.ppo_cnn",
    "neatrl.reinforce_mlp",
    "neatrl.reinforce_cnn",
    "neatrl.rnd_mlp",
    "neatrl.rnd_cnn",
    "neatrl.sac_mlp",
    "neatrl.sac_cnn",
    "neatrl.td3_mlp",
    "neatrl.td3_cnn",
])
def test_config_instantiates_with_defaults(module_name):
    mod = importlib.import_module(module_name)
    cfg = mod.Config()
    assert cfg is not None
    fields = dataclasses.fields(cfg)
    assert len(fields) > 0, f"{module_name}.Config has no dataclass fields"


def test_config_repr_is_informative():
    from neatrl.ppo_mlp import Config
    cfg = Config()
    r = repr(cfg)
    assert "Config" in r
    assert "seed" in r


def test_config_equality():
    from neatrl.ppo_mlp import Config
    assert Config() == Config()
    c1 = Config()
    c2 = Config()
    c2.seed = 99
    assert c1 != c2
