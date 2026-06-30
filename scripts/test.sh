#!/usr/bin/env zsh
set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h}
cd "$repo_root"

if [[ ! -r .venv/bin/activate ]]; then
  ./scripts/setup-dev.sh
fi

source .venv/bin/activate
python -m pytest "$@"
