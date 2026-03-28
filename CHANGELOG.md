# Changelog

All notable changes to this project will be documented in this file.

This repository uses `towncrier` to collect release-note fragments for unreleased changes.

<!-- towncrier release notes start -->

## 0.2.6 (2026-03-28)

### Miscellaneous

- Added Towncrier-based changelog management, seeded the historical changelog, enforced fragment checks in pull-request CI, and automated release-branch changelog generation for PRs into `master`.


## 0.2.5 (2025-12-11)

### 🔧 Bugfixes

- Improved comparison handling so non-comparable variable types are skipped more safely.

## 0.2.2 (2024-08-06)

### Documentation

- Removed the README table of contents and updated image links to point to the GitHub-hosted assets.

## 0.2.1 (2024-08-06)

### 🚀 Features

- Reworked comparison result rendering around the `Comparison` model and the Rich-based terminal table output.

### 🔧 Bugfixes

- Improved variable selection, relative-error handling, and the internal comparison flow performance.

## 0.2.0 (2024-07-30)

### 🚀 Features

- Switched core path handling to `pathlib`.

## 0.1.3 (2024-07-30)

### 🚀 Features

- Introduced the compare result model used by the dataset comparison flow.

## 0.1.2 (2024-07-30)

### Miscellaneous

- Published the 0.1.2 package version update.

## 0.1.1 (2024-07-30)

### 🚀 Features

- Published the first working package release.

## 0.1.0 (2024-07-22)

### 🚀 Features

- Created the initial project skeleton.
