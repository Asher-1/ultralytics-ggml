#!/usr/bin/env bash
#
# apply_ggml_patches.sh
#
# Apply the in-tree ggml integration patch to third_party/ggml. Idempotent:
# re-running is a no-op once it is applied.
#
# Exactly one patch must live in third_party/ggml-patches/. It is the complete
# diff from the pinned ggml submodule revision to the integrated source tree.
#
# Usage:
#   bash scripts/apply_ggml_patches.sh
#
# Exits 0 on success, non-zero on any failure. Designed to be called by CMake
# during configure but also runnable standalone for debugging.

set -euo pipefail

# Resolve the project root from the script's own location so this works from
# any CWD (including CMake's build dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${PROJECT_ROOT}/third_party/ggml"
PATCH_DIR="${PROJECT_ROOT}/third_party/ggml-patches"

if [[ ! -d "${GGML_DIR}" ]]; then
    echo "error: ggml submodule not found at ${GGML_DIR}" >&2
    echo "       did you forget 'git submodule update --init --recursive'?" >&2
    exit 1
fi

if [[ ! -d "${GGML_DIR}/.git" && ! -f "${GGML_DIR}/.git" ]]; then
    echo "error: ${GGML_DIR} is not a git repository" >&2
    exit 1
fi

if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "error: patch directory not found at ${PATCH_DIR}" >&2
    exit 1
fi

shopt -s nullglob
PATCHES=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if [[ ${#PATCHES[@]} -ne 1 ]]; then
    echo "error: expected exactly one ggml integration patch in ${PATCH_DIR}, found ${#PATCHES[@]}" >&2
    exit 1
fi
PATCH="${PATCHES[0]}"
PATCH_NAME="$(basename "${PATCH}")"

cd "${GGML_DIR}"

# Record the exact patch and resulting files, and scope the stamp to this git
# worktree. The tree signature also invalidates stale stamps after local edits.
PATCH_SIGNATURE="$(git hash-object "${PATCH}")"
mapfile -t PATCHED_PATHS < <(sed -n 's|^+++ b/||p' "${PATCH}" | LC_ALL=C sort -u)

patched_tree_signature() {
    for path in "${PATCHED_PATHS[@]}"; do
        if [[ -f "${path}" ]]; then
            printf '%s ' "${path}"
            git hash-object "${path}"
        else
            printf '%s missing\n' "${path}"
        fi
    done | git hash-object --stdin
}

STAMP_FILE="$(git rev-parse --git-path yolo-ggml-patches.signature)"
STAMP_VALUE="${PATCH_SIGNATURE} $(patched_tree_signature)"
if [[ -f "${STAMP_FILE}" ]] && [[ "$(<"${STAMP_FILE}")" == "${STAMP_VALUE}" ]]; then
    echo "ggml patch: ${PATCH_NAME} already applied"
    exit 0
fi

# Serialise concurrent invocations against the shared submodule tree: a
# best-effort flock on a sentinel file alongside the submodule serialises the
# window between `git apply --check` and the actual `git apply`.
if [[ -z "${YOLO_PATCH_FLOCK_HELD:-}" ]] && command -v flock >/dev/null 2>&1; then
    LOCK_FILE="${PROJECT_ROOT}/third_party/.ggml-patch.lock"
    : > "${LOCK_FILE}" 2>/dev/null || true
    if [[ -e "${LOCK_FILE}" ]]; then
        export YOLO_PATCH_FLOCK_HELD=1
        SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
        exec flock "${LOCK_FILE}" bash "${SCRIPT_PATH}" "$@"
    fi
fi

# Already applied? `git apply --check --reverse` succeeds iff every hunk is
# currently present in the tree (i.e. we could roll it back).
if git apply --check --reverse "${PATCH}" >/dev/null 2>&1; then
    printf '%s %s\n' "${PATCH_SIGNATURE}" "$(patched_tree_signature)" > "${STAMP_FILE}"
    echo "ggml patch: ${PATCH_NAME} already applied"
    exit 0
fi

if ! git apply --check "${PATCH}" >/dev/null 2>&1; then
    echo "error: cannot apply ${PATCH_NAME}" >&2
    echo "       'git apply --check' output (forward):" >&2
    git apply --check "${PATCH}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       'git apply --check --reverse' output:" >&2
    git apply --check --reverse "${PATCH}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       submodule HEAD: $(git rev-parse HEAD)" >&2
    echo "       try: cd ${GGML_DIR} && git status" >&2
    exit 1
fi

if ! git apply "${PATCH}"; then
    echo "error: failed to apply ${PATCH_NAME} after --check succeeded" >&2
    echo "       this should not happen; the submodule tree may be dirty" >&2
    exit 1
fi

printf '%s %s\n' "${PATCH_SIGNATURE}" "$(patched_tree_signature)" > "${STAMP_FILE}"

echo "ggml patch: applied ${PATCH_NAME}"
