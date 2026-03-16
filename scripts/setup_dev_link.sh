#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
plugin_src="$repo_root/ai_vectorizer"
plugins_dir="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
plugin_link="$plugins_dir/ai_vectorizer"

mkdir -p "$plugins_dir"
ln -sfn "$plugin_src" "$plugin_link"

printf 'Linked %s -> %s\n' "$plugin_link" "$plugin_src"
