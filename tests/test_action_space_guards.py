"""Discrete-by-default trainers must reject Box action spaces with a clear error.

PPO, PPO-RND, A2C and REINFORCE ship categorical actors. Naming a continuous
environment used to fail deep inside gymnasium with

    IndexError: invalid index to scalar variable.

after the first rollout step, which reads like a NeatRL bug. These tests pin the
explicit ``ValueError`` that names the environment and the alternatives; A2C
already had that guard, the others now do too.

Continuous control still works with a custom network (``actor_class`` for PPO /
RND / A2C, ``custom_agent`` for REINFORCE), so the guards only fire for the
default actor — they must not be unconditional.
"""

import inspect

import pytest

CASES = [
    ("neatrl.ppo_mlp", "train_ppo", {"total_timesteps": 2048, "max_steps": 64}),
    ("neatrl.ppo_cnn", "train_ppo_cnn", {"total_timesteps": 2048, "max_steps": 64}),
    ("neatrl.rnd_mlp", "train_ppo_rnd", {"total_timesteps": 2048, "max_steps": 64}),
    ("neatrl.rnd_cnn", "train_ppo_rnd_cnn", {"total_timesteps": 2048, "max_steps": 64}),
    ("neatrl.a2c_mlp", "train_a2c", {"total_timesteps": 512}),
    ("neatrl.reinforce_mlp", "train_reinforce", {"total_episodes": 2}),
    ("neatrl.reinforce_cnn", "train_reinforce_cnn", {"total_episodes": 2}),
]


@pytest.mark.parametrize("module,function,extra", CASES, ids=[c[1] for c in CASES])
def test_continuous_env_raises_a_readable_error(module, function, extra):
    import importlib

    trainer = getattr(importlib.import_module(module), function)

    with pytest.raises(ValueError) as info:
        trainer(
            env_id="Pendulum-v1",
            use_wandb=False,
            capture_video=False,
            eval_every=1024,
            save_every=1024,
            **extra,
        )

    message = str(info.value)
    assert "only supports discrete action spaces" in message
    assert "Pendulum-v1" in message
    # The message has to point somewhere: a trainer that does continuous control.
    assert "train_sac" in message or "train_sac_cnn" in message


def test_guards_check_the_default_network_not_the_environment():
    """A custom network is the documented continuous escape hatch, so the guard
    must compare against the built-in actor class rather than testing every
    Box action space."""
    from neatrl import ppo_mlp

    source = inspect.getsource(ppo_mlp.train_ppo)
    assert "actor_class is ActorNet" in source


def test_cli_reports_discrete_only_as_a_user_error(capsys):
    """`neatrl train ppo Pendulum-v1` is a mistake, not a crash: one line, exit 2.

    The trainer raises a ValueError whose advice (`actor_class`) the CLI cannot
    act on, so the recipe layer rewrites it into a command the user can run.
    """
    from neatrl.cli.main import main

    with pytest.raises(SystemExit) as info:
        main(["train", "ppo", "Pendulum-v1", "--steps", "100"])

    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "trains discrete action spaces only" in err
    assert "neatrl train sac Pendulum-v1" in err
    assert "Traceback" not in err
