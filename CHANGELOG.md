# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-04-11

### Added
- Major version bump to 1.3.0
- Release checklist for future releases
- Comprehensive release documentation

### Changed
- Updated version from 0.1.4.post6 to 1.3.0
- Standardized release process
- Updated version references in nanobot/__init__.py

### Fixed
- Verified no TODO(v1.3) or FIXME(v1.3) items remain in codebase

## [0.1.4.post6] - 2026-03-27

### Added
- Architecture decoupling improvements
- End-to-end streaming support
- WeChat channel integration
- Security fixes for litellm removal

### Changed
- Removed litellm dependency completely
- Improved streaming delta coalescing at boundaries

### Fixed
- Security vulnerability related to litellm supply chain

## [0.1.4.post5] - 2026-03-16

### Added
- DingTalk rich media support
- Enhanced built-in skills
- Improved model compatibility

### Changed
- Refinement-focused release with stronger reliability

### Fixed
- Channel stability improvements

## [0.1.4.post4] - 2026-03-08

### Added
- Safer defaults for security
- Better multi-instance support
- Sturdier MCP integration
- Major channel and provider improvements

### Changed
- Hardened provider & channel stability

### Fixed
- Multiple reliability issues

## [0.1.4.post3] - 2026-02-28

### Added
- Cleaner context management
- Session history hardening
- Smarter agent behavior

### Fixed
- Session poisoning issues
- WhatsApp deduplication
- Windows path handling

## [0.1.4.post2] - 2026-02-24

### Added
- Redesigned heartbeat system
- Prompt cache optimization

### Changed
- Hardened provider & channel stability

### Fixed
- Virtual tool-call heartbeat issues
- Slack mrkdwn rendering
- Agent reliability improvements

## [0.1.4.post1] - 2026-02-21

### Added
- New provider support
- Media support across channels
- Major stability improvements

### Changed
- Enhanced Feishu multimodal file reception
- Improved memory reliability

### Fixed
- Slack thread isolation
- Discord typing indicators
- Agent reliability issues

## [0.1.4] - 2026-02-17

### Added
- MCP (Model Context Protocol) support
- Progress streaming
- New provider integrations
- Multiple channel improvements

### Changed
- Enhanced ClawHub skill integration

### Fixed
- Various channel and provider issues

## [0.1.3.post7] - 2026-02-13

### Added
- Security hardening features
- Multiple reliability improvements

### Fixed
- Security vulnerabilities (upgrade recommended)

## [0.1.3.post6] - 2026-02-10

### Added
- Enhanced CLI experience
- MiniMax provider support

### Changed
- Improved user experience

## [0.1.3.post5] - 2026-02-07

### Added
- Qwen provider support
- Enhanced security hardening

### Fixed
- Multiple key improvements

## [0.1.3.post4] - 2026-02-04

### Added
- Multi-provider support
- Docker support

### Changed
- Improved deployment options

## [0.1.3] - 2026-02-02

### Added
- Initial nanobot release
- Core agent functionality
- Basic channel integrations
- Provider system
- Memory system
- Skills framework
