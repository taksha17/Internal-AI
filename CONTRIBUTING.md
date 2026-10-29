# Contributing to Internal-AI Platform

Thank you for your interest in contributing to the Internal-AI platform! This document outlines the guidelines and processes for contributing to this project.

## Who can write to `main`

Anyone can contribute. A contribution is a pull request. Further work on that change is another push to the pull request's branch.

`main` on `taksha17/Internal-AI` is different. Commits and pushes to `main`, including the merge of a pull request, a force-push, and deleting the branch, are accepted only from the repository owner, GitHub account `taksha17`. Every other account is rejected. You do not need write access on this repository to contribute.

## Table of Contents

1. [Who can write to `main`](#who-can-write-to-main)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Running Tests](#running-tests)
6. [Contributing Process](#contributing-process)
7. [Pull Request Guidelines](#pull-request-guidelines)
8. [Documentation Guidelines](#documentation-guidelines)
9. [Security Considerations](#security-considerations)
10. [Community Guidelines](#community-guidelines)
11. [License and Attribution](#license-and-attribution)

## Getting Started

### Prerequisites

Before contributing, ensure you have the following installed:

- Python 3.12+
- `virtualenv` or `poetry` (recommended)
- SQLite 3.35+ (for the internal database)

### Fork and Clone

1. Fork [taksha17/Internal-AI](https://github.com/taksha17/Internal-AI) on GitHub. Your fork is where your branches live.
2. Clone your fork locally:

```bash
git clone https://github.com/[your-username]/Internal-AI.git
cd Internal-AI
```

3. Add this repository as `upstream`. Push to `origin` (your fork), never to `upstream` `main`:

```bash
git remote add upstream https://github.com/taksha17/Internal-AI.git
```

## Development Setup

### Environment Setup

We recommend using a virtual environment for development:

```bash
# Using virtualenv
python3 -m venv .venv
.venv/bin/pip install -e .[dev]

# Using poetry (optional)
poetry install
```

### Project Structure

The main components are:

- `src/internal_ai/` - Core application code
- `tests/` - Test suite
- `docs/` - Documentation
- `scripts/` - Development scripts

### Development Workflow

1. Start a branch from the owner's `main`. Push that branch to your fork:

```bash
git fetch upstream
git checkout -b your-feature-branch upstream/main
git push -u origin your-feature-branch
```

2. Keep your changes focused on a single issue or feature
3. Follow the existing code style and patterns
4. Write tests for new functionality
5. Update documentation as needed

## Code Style Guidelines

### Python Code Style

This project uses:

- **Python 3.12** with type hints
- **Black** for code formatting
- **ruff** for linting and formatting
- **mypy** for type checking

#### Formatting

```bash
.venv/bin/ruff format src/
.venv/bin/ruff format tests/
```

#### Linting

```bash
.venv/bin/ruff check src/
.venv/bin/ruff check tests/
```

#### Type Checking

```bash
.venv/bin/mypy src/
```

### Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a concise summary (limit to 50 characters)
- Explain the "why" behind the change in the body
- Reference relevant issues or pull requests
- Use the present tense
- Avoid commit messages that only describe what the code does

Example:

```
feat: add user authentication endpoint

- Implement new POST /api/auth endpoint
- Add validation for credentials
- Return JWT token on successful authentication

Closes #123
```

## Running Tests

### Test Suite

Run the full test suite:

```bash
.venv/bin/python -m pytest
```

### Test Categories

- **Unit tests**: Test individual components and functions
- **Integration tests**: Test interactions between components
- **Acceptance tests**: Test end-to-end functionality

### Running Specific Tests

Run tests for a specific module:

```bash
.venv/bin/python -m pytest tests/test_file.py
```

Run tests with specific markers:

```bash
.venv/bin/python -m pytest -m "integration"
```

### Test Output

Tests should pass without errors or warnings. If there are linting issues, run the formatting and linting commands before committing.

## Contributing Process

### 1. Identify an Issue

- Review existing issues and feature requests
- If no issue exists, create a new issue describing your proposed changes
- Clearly describe the problem and proposed solution

### 2. Fork and Branch

- Fork the repository
- Create a feature branch from `main`
- Give your branch a descriptive name (e.g., `feature/add-user-auth`, `bugfix/fix-login-issue`)

### 3. Implement Changes

- Implement your changes following the code style guidelines
- Write tests for your changes
- Update documentation as needed

### 4. Review Your Changes

- Ensure all tests pass
- Run linting and formatting tools
- Perform a final review of your changes

### 5. Open a pull request, then keep pushing to it

- Push the branch to your fork: `git push -u origin your-feature-branch`
- Open a pull request from that branch into `taksha17/Internal-AI` `main`
- Fill out the pull request template and describe why the change is needed
- To update the pull request, commit on the same branch and push to your fork again. The open pull request picks up those commits. Do not open a second pull request for the same change, and do not push the branch to `upstream` `main`

## Pull Request Guidelines

### Pull Request Template

Please use the following template for your pull requests:

```markdown
## Summary
[Provide a concise summary of the changes]

## Description
[Detailed description of the changes and why they are needed]

## Testing
[Describe how to test the changes and any new tests added]

## Documentation
[Describe any documentation changes]

## Checklist
- [ ] Code follows project coding standards
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No linting issues
- [ ] Changes are focused and minimal
```

### Review Process

- **Review**: The owner reads the pull request. Automated checks run the tests and the linter.
- **Updates**: Push more commits to the same branch on your fork. That updates the open pull request.
- **Merge**: Only `taksha17` merges into `main`. A merge is a commit on `main`, so it uses the owner's credentials. You will not be able to merge your own pull request or push to `main`.

## Documentation Guidelines

### Documentation Structure

Documentation is stored in the `docs/` directory:

- `docs/` - Main documentation
- `docs/acceptance/` - Acceptance test documentation
- `docs/changes/` - Change logs and release notes

### Writing Documentation

- Use Markdown format
- Follow the existing documentation style
- Include examples and code snippets where appropriate
- Keep documentation clear and concise

### Updating Documentation

- Update relevant documentation when making changes
- Document new APIs and interfaces
- Update examples and code snippets

## Security Considerations

### Safe Practices

- Never commit sensitive information (API keys, passwords, etc.)
- Use environment variables for configuration
- Regularly review and update dependencies
- Run security scans periodically

### Vulnerability Reporting

If you discover a security vulnerability, please report it responsibly:

1. Email the security team at security@internal-ai.com
2. Provide detailed information about the vulnerability
3
. Describe potential impact and remediation steps

## Community Guidelines

### Code of Conduct

We follow a code of conduct that promotes:

- Respectful communication
- Constructive feedback
- Collaboration and teamwork
- Diversity and inclusion

### Communication

- Be patient and respectful with others
- Provide constructive feedback
- Ask for clarification when needed

## License and Attribution

### License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**. For more details, see the `LICENSE` file.

### Attribution

- Project logo and brand elements are owned by the Internal-AI team
- Third-party libraries and tools should be attributed according to their own licenses
- All contributors are acknowledged in the repository's commit history

## Getting Help

If you have questions about contributing to this project:

1. **GitHub Issues**: Post questions in the repository's issue tracker
2. **Discussions**: Participate in repository discussions
3. **Contact**: Reach out to the project maintainers directly

## Contributing Checklist

Before submitting your changes, ensure you have:

- [ ] Updated the `CHANGELOG.md` file (if applicable)
- [ ] Updated the `README.md` if your changes affect usage
- [ ] Added tests for new functionality
- [ ] Followed code style guidelines
- [ ] Updated documentation
- [ ] No linting or formatting issues
- [ ] Changes are focused and minimal

## Thank You

We appreciate your contributions to the Internal-AI platform! Your help makes this project better for everyone.

---

*This Contributing Guide was last updated on 29 October 2026.*

For more information, please visit the repository on [GitHub](https://github.com/taksha17/Internal-AI).