#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/release.sh 0.1.0b8
# Or:    make release V=0.1.0b8

VERSION="${1:?Usage: release.sh <version>  (e.g. 0.1.0b8)}"
TAG="v${VERSION}"

# Ensure working tree is clean
if [ -n "$(git status --porcelain)" ]; then
  echo "Error: working tree is dirty. Commit or stash changes first."
  exit 1
fi

# Update version in pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml
echo "Updated pyproject.toml to version ${VERSION}"

# Commit, tag, push
git add pyproject.toml
git commit -m "release: v${VERSION}"
git tag "${TAG}"
git push origin main "${TAG}"

echo ""
echo "Done! Tag ${TAG} pushed. GitHub Actions will publish to PyPI."
echo "Track: https://github.com/tyxben/arcana/actions"
