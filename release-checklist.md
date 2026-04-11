# Release Checklist

## Version Information
- Current version: 0.1.4.post6
- Target version: 1.3.0
- Previous version: 1.2.3

## Pre-Release Steps

### 1. Version Updates
- [ ] Update `pyproject.toml` version from 0.1.4.post6 to 1.3.0
- [ ] Update version references in README.md
- [ ] Update version references in any other documentation files
- [ ] Check for version strings in Python files

### 2. Changelog Preparation
- [ ] Review all commits since previous release
- [ ] Categorize commits (Features, Bug Fixes, Breaking Changes, etc.)
- [ ] Update CHANGELOG.md with all changes
- [ ] Ensure changelog follows existing format

### 3. Code Cleanup
- [ ] Search for and resolve all TODO(v1.3) items
- [ ] Search for and resolve all FIXME(v1.3) items
- [ ] Remove any debug code or temporary comments
- [ ] Run tests to ensure everything works

### 4. Documentation
- [ ] Update README.md if needed
- [ ] Update any relevant documentation
- [ ] Ensure all examples are current

### 5. Release Commit
- [ ] Create release commit with message: "Release v1.3.0"
- [ ] Tag the commit with v1.3.0
- [ ] Push commit and tag to remote

## Post-Release Steps
- [ ] Update version to next development version
- [ ] Announce release
- [ ] Monitor for issues
