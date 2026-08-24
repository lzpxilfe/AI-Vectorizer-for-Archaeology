#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
plugin_src="${ARCHAEOTRACE_PLUGIN_SOURCE:-$repo_root/ai_vectorizer}"
plugins_dir="${ARCHAEOTRACE_QGIS_PLUGINS_DIR:-$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins}"
plugin_link="$plugins_dir/ai_vectorizer"

if [[ ! -d "$plugin_src" ]]; then
    printf 'Plugin source directory does not exist: %s\n' "$plugin_src" >&2
    exit 1
fi
if [[ ! -f "$plugin_src/metadata.txt" ]]; then
    printf 'Plugin source is missing metadata.txt: %s\n' "$plugin_src" >&2
    exit 1
fi
plugin_src="$(cd "$plugin_src" && pwd -P)"

mkdir -p "$plugins_dir"

if [[ -L "$plugin_link" ]]; then
    existing_target="$(readlink "$plugin_link")"
    if [[ "$existing_target" == "$plugin_src" ]]; then
        printf 'Already linked %s -> %s\n' "$plugin_link" "$plugin_src"
        exit 0
    fi
    printf 'Refusing to replace a link to another source: %s -> %s\n' \
        "$plugin_link" "$existing_target" >&2
    exit 1
elif [[ -e "$plugin_link" ]]; then
    printf 'Refusing to replace an existing plugin directory: %s\n' "$plugin_link" >&2
    printf 'Move or remove it explicitly, then run this script again.\n' >&2
    exit 1
fi

ln -s "$plugin_src" "$plugin_link"
if [[ ! -L "$plugin_link" || "$(readlink "$plugin_link")" != "$plugin_src" ]]; then
    printf 'Failed to create the expected plugin link: %s\n' "$plugin_link" >&2
    exit 1
fi

printf 'Linked %s -> %s\n' "$plugin_link" "$plugin_src"
