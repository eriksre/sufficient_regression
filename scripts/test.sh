#!/usr/bin/env zsh
set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h}
cd "$repo_root"

if [[ ! -r .venv/bin/activate ]]; then
  ./scripts/setup-dev.sh
fi

source .venv/bin/activate

if ! python -c "import pytest" >/dev/null 2>&1; then
  # A gitignored .venv can exist without the editable dev install; tests own
  # bootstrapping that state so a fresh workspace fails at test failures, not setup.
  ./scripts/setup-dev.sh
  source .venv/bin/activate
fi

python -m pytest "$@"
