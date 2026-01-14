#!/usr/bin/env bash
set -euo pipefail

readonly BRANCH_NAME="${BRANCH_NAME:?BRANCH_NAME is required}"
readonly PATTERN="^(feature|fix|hotfix|refactor|docs|chore|update)/.+"

main() {
    echo "Checking branch name: $BRANCH_NAME"

    if [[ "$BRANCH_NAME" =~ $PATTERN ]]; then
        echo "Branch name '$BRANCH_NAME' follows the naming convention."
        exit 0
    fi

    echo "::error::Branch name '$BRANCH_NAME' does not follow the naming convention."
    echo ""
    echo "Expected format: <type>/<description>"
    echo "Valid prefixes: feature/, fix/, hotfix/, refactor/, docs/, chore/, update/"
    echo ""
    echo "Examples:"
    echo "  - feature/add-tetromino"
    echo "  - fix/score-bug"
    echo "  - update/zio-version (for dependencies)"
    echo ""
    echo "Please rename your branch and create a new PR."
    exit 1
}

main "$@"
