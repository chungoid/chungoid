# Contributing to Chungoid Core

Thank you for your interest in contributing to the Chungoid core engine! Please follow these guidelines to ensure a smooth development process.

## Branching Strategy

We use a simple branching model based on Gitflow principles:

*   **`main`**: This branch represents the latest stable, released version. Direct commits to `main` are prohibited. Merges into `main` typically come only from the `testing` branch during a release process.
*   **`testing`**: This is the primary development branch. It contains the latest features and fixes that are ready for testing and integration before a release. All feature branches should be based on `testing` and merged back into `testing` via Pull Requests. **Meta-developers working within the `metachungoid` environment should typically have `chungoid-core` checked out to this branch.**
*   **Feature Branches (`feature/<feature-name>`)**: For developing new features.
    *   Create from the latest `testing` branch (`git checkout -b feature/<your-feature-name> testing`).
    *   Name descriptively (e.g., `feature/improved-state-handling`).
    *   Once complete, open a Pull Request to merge back into `testing`.
*   **Bugfix Branches (`fix/<issue-number>` or `fix/<short-description>`)**: For fixing bugs found in `testing` or `main`.
    *   If fixing a bug in `testing`, branch from `testing`.
    *   If fixing a critical bug in `main` (hotfix), branch from `main`. Hotfixes are merged back to both `main` and `testing`.
    *   Name descriptively (e.g., `fix/123` or `fix/race-condition-on-start`).
    *   Open a Pull Request to merge back into the appropriate branch (`testing` or potentially both `main` and `testing` for hotfixes).

## Commit Messages

Please follow Conventional Commits guidelines for clear and consistent commit messages. See: [https://www.conventionalcommits.org/](https://www.conventionalcommits.org/)

## Pull Requests

*   Base all Pull Requests against the `testing` branch unless it's a hotfix for `main`.
*   Ensure your code passes all tests (`pytest tests/`).
*   Provide a clear description of the changes in the Pull Request.
*   Link any relevant issues.

## Setting Up (for Meta-Developers)

Refer to the `metachungoid` repository's `dev/ONBOARDING.md` file for instructions on setting up the full development environment, including cloning this repository into the correct location (`chungoid-core/`) and checking out the `testing` branch. 