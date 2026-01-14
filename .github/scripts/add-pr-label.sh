#!/usr/bin/env bash
set -euo pipefail

# Add label to PR based on branch name prefix
readonly BRANCH_NAME="${BRANCH_NAME:?BRANCH_NAME is required}"
readonly PR_NUMBER="${PR_NUMBER:?PR_NUMBER is required}"
readonly REPO="${REPO:?REPO is required}"

get_label_for_branch() {
    local branch="$1"
    case "$branch" in
        feature/*)  echo "enhancement" ;;
        fix/*)      echo "bug" ;;
        hotfix/*)   echo "bug" ;;
        refactor/*) echo "refactor" ;;
        docs/*)     echo "documentation" ;;
        chore/*)    echo "chore" ;;
        update/*)   echo "dependencies" ;;
        *)          echo "" ;;
    esac
}

main() {
    echo "Branch name: $BRANCH_NAME"
    local label
    label=$(get_label_for_branch "$BRANCH_NAME")

    if [[ -z "$label" ]]; then
        echo "No matching prefix found. Skipping label assignment."
        exit 0
    fi

    echo "Assigning label: $label"

    # Add label to PR using GitHub CLI
    gh pr edit "$PR_NUMBER" --repo "$REPO" --add-label "$label"

    echo "Label '$label' has been added to PR #$PR_NUMBER"
}

main "$@"
