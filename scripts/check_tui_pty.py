#!/usr/bin/env python3
"""Drive the NeatRL TUI inside a real pseudo-terminal and check it renders.

This is the one check that cannot run under pytest: the Dashboard only activates
when stdout is a TTY. We allocate a pty with Python's ``pty`` module, run a short
training for each trainer inside it, and assert that ANSI dashboard frames were
actually drawn, the l/c/v views cycle, and the process exits cleanly.

Usage:
    python scripts/check_tui_pty.py
    python scripts/check_tui_pty.py --only double_dqn
"""

import argparse
import os
import pty
import re
import select
import sys
import time

# algo -> (env, steps, extra kwargs)
_MLP_EXTRA = {
    "learning_starts": 1000,
    "buffer_size": 50000,
    "batch_size": 32,
    "eval_every": 10000,
    "grid_env": False,
    "atari_wrapper": False,
}
_CNN_EXTRA = {
    "learning_starts": 200,
    "buffer_size": 5000,
    "batch_size": 8,
    "eval_every": 1000,
}

ALGOS = {
    "dqn": ("CartPole-v1", 30000, _MLP_EXTRA),
    "double_dqn": ("CartPole-v1", 30000, _MLP_EXTRA),
    "dqn_cnn": ("ALE/Pong-v5", 1500, _CNN_EXTRA),
    "double_dqn_cnn": ("ALE/Pong-v5", 1500, _CNN_EXTRA),
}

# Module that defines each trainer (the Dashboard lives there).
ALGO_MODULE = {
    "dqn": "dqn_mlp",
    "double_dqn": "double_dqn_mlp",
    "dqn_cnn": "dqn_cnn",
    "double_dqn_cnn": "double_dqn_cnn",
}

CODE = """
import sys
from neatrl import {fn}
import neatrl.{module} as m

orig_update = m.Dashboard.update
frames = {{"n": 0}}
def counting_update(self, **kw):
    frames["n"] += 1
    return orig_update(self, **kw)
m.Dashboard.update = counting_update

{fn}(
    env_id={env!r}, total_timesteps={steps}, seed=0,
    save_every=0, num_eval_eps=1, use_wandb=False,
    capture_video=False, device="cpu", {extra}
)
print(f"FRAMES={{frames['n']}}")
print("TUI_TEST_OK")
"""


def run_in_pty(code, timeout=180):
    """Run python code with stdin/stdout on a pty; return captured output."""
    pid, fd = pty.fork()
    if pid == 0:  # child
        os.environ["PYTHONPATH"] = "src"
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = "120"
        os.environ["LINES"] = "40"
        os.execvp(sys.executable, [sys.executable, "-c", code])
        os._exit(1)

    # Toggle views mid-run, the way a user pressing keys would.
    pending = [(1.5, b"l"), (2.5, b"c"), (3.5, b"v"), (4.5, b"\x1b")]
    chunks = []
    start = time.time()
    exited = False
    while not exited and time.time() - start < timeout:
        if pending and time.time() - start >= pending[0][0]:
            _, key = pending.pop(0)
            try:
                os.write(fd, key)
            except OSError:
                pass

        r, _, _ = select.select([fd], [], [], 0.1)
        if not r:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            exited = True
            break
        chunks.append(data)

    try:
        os.close(fd)
    except OSError:
        pass
    _, status = os.waitpid(pid, 0)
    return b"".join(chunks).decode("utf-8", "replace"), status


def check_algo(algo):
    env, steps, extra = ALGOS[algo]
    code = CODE.format(
        fn=f"train_{algo}",
        module=ALGO_MODULE[algo],
        env=env,
        steps=steps,
        extra=", ".join(f"{k}={v!r}" for k, v in extra.items()),
    )
    out, status = run_in_pty(code)

    problems = []
    if "TUI_TEST_OK" not in out:
        problems.append("training did not finish inside the pty")
    if status != 0:
        problems.append(f"child exited with status {status}")

    # Dashboard frames are drawn with ANSI cursor-control sequences.
    ansi = re.findall(r"\x1b\[\d*A|\x1b\[2J|\x1b\[H", out)
    if len(ansi) < 3:
        problems.append(f"only {len(ansi)} ANSI frame sequences; TUI never drew")

    for needle in ("NeatRL", env):
        if needle not in out:
            problems.append(f"missing {needle!r} in rendered output")

    frames = re.search(r"FRAMES=(\d+)", out)
    if not frames or int(frames.group(1)) == 0:
        problems.append("Dashboard.update was never called")

    if problems:
        print(f"FAIL {algo}")
        for p in problems:
            print(f"  - {p}")
        print("\n--- tail of pty output ---")
        print(out[-2000:])
        return False

    print(
        f"PASS {algo:<18s} {env:<14s} frames={frames.group(1):>6s} ansi_ops={len(ansi)}"
    )
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="substring filter on algo name")
    args = parser.parse_args(argv)

    algos = list(ALGOS)
    if args.only:
        algos = [a for a in algos if args.only in a]

    print(f"checking TUI in a pty for: {', '.join(algos)}\n")
    results = [check_algo(a) for a in algos]
    ok = sum(results)
    print(f"\n{ok}/{len(algos)} trainers rendered the TUI cleanly")
    return 0 if ok == len(algos) else 1


if __name__ == "__main__":
    sys.exit(main())
