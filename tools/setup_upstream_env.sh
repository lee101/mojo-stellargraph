#!/usr/bin/env bash
# Create (or refresh) the pinned Python 3.8 + `stellargraph` 1.2.1 environment
# used for parity testing.
#
# This cannot be a pixi environment: upstream `stellargraph` 1.2.1 declares
# `python_requires < 3.9` and depends on TensorFlow 2.1+, and the pixi
# environment for this repo is Python 3.13 for the Mojo toolchain. So it is a
# separate `uv`-managed virtualenv under `.upstream/`, which is gitignored.
#
# Regenerate the reference arrays afterwards with `pixi run dump-upstream`.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$repo_dir/.upstream"

uv_bin="$(command -v uv || true)"
if [ -z "$uv_bin" ]; then
    echo "uv not found: install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [ ! -x "$env_dir/.venv/bin/python" ]; then
    "$uv_bin" venv --python 3.8 "$env_dir/.venv"
fi

# `chardet` is an undeclared import in `stellargraph.data.epgm`, which
# `import stellargraph` pulls in transitively; without it the package cannot be
# imported at all.
UV_LINK_MODE=copy "$uv_bin" pip install --python "$env_dir/.venv/bin/python" \
    "stellargraph==1.2.1" "numpy<1.24" "scipy<1.11" chardet

"$env_dir/.venv/bin/python" - <<'PY'
import sys
import stellargraph
print("upstream env ready:", sys.version.split()[0], "stellargraph",
      getattr(stellargraph, "__version__", "?"))
PY