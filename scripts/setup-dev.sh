#!/usr/bin/env zsh
set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h}
cd "$repo_root"

if [[ ! -d .venv ]]; then
  # .venv is gitignored, so each fresh workspace needs its own environment.
  python3 -m venv .venv
fi

if [[ ! -r .venv/bin/activate ]]; then
  print -u2 "Existing .venv is missing bin/activate; remove it and rerun scripts/setup-dev.sh."
  exit 1
fi

source .venv/bin/activate
python -m pip install -e ".[dev]"
