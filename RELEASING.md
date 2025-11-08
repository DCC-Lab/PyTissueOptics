# Releasing PyTissueOptics

## 1. Prerequisites

- You must have permission to publish to [PyPI](https://pypi.org/project/pytissueoptics/).
- Ensure the following tools are installed and up to date:
  ```bash
  pip install --upgrade build twine
    ```
- Local git repository is clean.
    ```bash
    git checkout main
    git pull origin main
    ```

A note on versioning:
> PyTissueOptics uses `setuptools_scm`, which automatically infers the package version from Git tags.
> Follow [Semantic Versioning](https://semver.org/) to tag releases as `vMAJOR.MINOR.PATCH` (e.g., `v2.0.1`).
Development versions are built from commits after the latest tag and look like `v2.0.1.dev3+gabc123.d20251108`.

## 2. Create a Git tag
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```
  
## 3. Build the package
```bash
rm -rf dist/
python -m build
```

## 4. Upload to PyPI
```bash
python -m twine upload dist/*
```

## 5. Publish GitHub Release
Used to announce and document the changes in the new version.
- Go to the [GitHub releases page](https://github.com/DCC-Lab/PyTissueOptics/releases)
- Click on "Draft a new release"
- Select the tag you created earlier
- Fill in the release title and description (generate changelog from last tag).
- Click "Publish release"
