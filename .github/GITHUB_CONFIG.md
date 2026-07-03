# GitHub Repository Configuration
# This file documents recommended settings to configure in GitHub web UI

## Branch Protection Rules (main)

### Require pull request reviews
- ✅ Require pull request reviews before merging
- Required approving reviews: 1
- Dismiss stale pull request approvals when new commits are pushed
- Require review from Code Owners

### Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require status checks to pass before merging
  - tests
  - lint

### Require conversation resolution before merging
- ✅ Require conversations to be resolved before merging

## Auto-merge
- ✅ Allow auto-merge commits
- ✅ Allow squash merging
- ✅ Allow rebase merging

## Code Security
- ✅ Enable security and analysis
  - Dependabot alerts
  - Dependabot security updates
  - Secret scanning

---

## To Apply These Settings:

1. Go to repository Settings → Branches
2. Click "Add branch protection rule"
3. Branch name pattern: `main`
4. Check all boxes as indicated above
5. Save changes

---

## Environment Variables for Actions

Add these in Settings → Secrets and variables → Actions:

- None required (app uses environment detection)
- Optional: `HF_TOKEN` for Hugging Face Hub

---

See [GitHub Docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository) for details.
