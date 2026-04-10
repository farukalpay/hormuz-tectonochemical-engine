from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_install_commands(venv_path: str) -> list[list[str]]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_executable = "/opt/homebrew/bin/python3.11" if system == "darwin" and "arm" in machine else "python3"
    commands = [
        [python_executable, "-m", "venv", venv_path],
        [f"{venv_path}/bin/pip", "install", "-r", "code/requirements.txt"],
    ]
    if system == "darwin" and "arm" in machine:
        commands.append([f"{venv_path}/bin/pip", "install", "-r", "code/requirements-tf-macos.txt"])
        commands.append([f"{venv_path}/bin/pip", "install", "-e", "./code[dev,tensorflow-apple]"])
    else:
        commands.append([f"{venv_path}/bin/pip", "install", "-r", "code/requirements-tf-linux-windows.txt"])
        commands.append([f"{venv_path}/bin/pip", "install", "-e", "./code[dev,tensorflow]"])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a local MCP host environment")
    parser.add_argument("--venv", default=".venv-tf")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = build_install_commands(args.venv)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    for command in commands:
        subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
