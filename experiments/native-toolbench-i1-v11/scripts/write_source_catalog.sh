#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out=${1:?usage: write_source_catalog.sh OUTPUT}
mkdir -p "$(dirname -- "$out")"
(
  cd "$root"
  find src scripts slurm tests -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' -o -name '*.json' -o -name '*.toml' \) -print0 \
    | sort -z \
    | xargs -0 sha256sum
  sha256sum compat/stdatomic.h
  sha256sum README.md PROTOCOL.md pyproject.toml
) > "$out"
sha256sum "$out" | awk '{print $1}'
