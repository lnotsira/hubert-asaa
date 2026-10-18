import os
import subprocess # calls terminal commands in python
import sys
from pathlib import Path


def cmd_exec(
    cmd: list[str],
    env_update: dict[str, str] | None = None,
) -> None:
    cmd = list(cmd)

    if cmd[0] == "python" or cmd[0] == sys.executable:
        cmd = [sys.executable, "-u", *cmd[1:]]

    print("\ncmd:", flush=True)
    print(" ".join(cmd), flush=True)
    print(flush=True)

    env = os.environ.copy()

    if env_update is not None:
        env.update(env_update)

    subprocess.run(
        cmd,
        check=True,
        env=env,
    )


def check_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wipe_dir(path: Path) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
