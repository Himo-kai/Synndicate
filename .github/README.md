# Synndicate CI/CD Pipeline

This directory contains the complete CI/CD and code quality infrastructure for the Synndicate AI platform.

## 🚀 **Implemented Workflows**

### Security & Vulnerability Scanning

- **`semgrep.yml`** - Advanced security scanning with SARIF upload
- **`bandit.yml`** - Python-specific vulnerability detection

### Code Quality & Type Safety  

- **`lint-type.yml`** - Ruff linting + MyPy strict type checking
- **`test.yml`** - Comprehensive test suite with coverage reporting

### Automated Code Review

- **`danger.yml`** - Intelligent PR review with Synndicate-specific rules
- **`semantic.yml`** - Conventional commit enforcement

### Status & Monitoring

- **`ci-status.yml`** - Pipeline status dashboard

## 🎯 **Quality Gates**

All PRs must pass:

- ✅ Semgrep security scan (no high-severity issues)
- ✅ Bandit vulnerability scan (no medium+ issues)  
- ✅ Ruff linting compliance
- ✅ MyPy strict type checking
- ✅ Test coverage ≥70%
- ✅ Conventional commit format
- ✅ Automated code review checks

## 🔒 **Security Features**

### Automated Security Scanning

- **Weekly scheduled scans** for continuous security monitoring
- **SARIF integration** with GitHub Security tab
- **Security-sensitive pattern detection** in code changes
- **API/Auth change notifications** for security review

### Synndicate-Specific Security Rules

- Critical system file change alerts (`core/orchestrator.py`, `api/server.py`)
- Observability change validation requirements
- Configuration change environment compatibility checks
- Performance-critical area benchmarking reminders

## 📊 **Coverage & Reporting**

### Artifacts Generated

- **MyPy HTML reports** (type coverage analysis)
- **Test coverage reports** (XML + HTML formats)
- **Bandit security reports** (JSON format)
- **Semgrep SARIF files** (GitHub Security integration)

### Retention Policies

- Security reports: 30 days
- Coverage reports: 30 days  
- Type checking reports: 7 days
- CI logs: GitHub default

## 🛠️ **Development Workflow**

### For Contributors

1. **Fork & Branch**: Create feature branches from `develop`
2. **Develop**: Write code with proper type annotations
3. **Test Locally**: Run `ruff check`, `mypy`, and `pytest` before pushing
4. **Create PR**: Use conventional commit format in PR title
5. **Review**: Address automated feedback from Danger.js
6. **Merge**: All quality gates must pass

### PR Title Format

```
<type>(<scope>): <description>

Examples:
feat(api): add JWT token refresh endpoint
fix(rag): resolve vectorstore connection timeout
docs(observability): update tracing configuration guide
```

### Supported Scopes

- `api`, `core`, `rag`, `models`, `observability`
- `agents`, `config`, `security`, `tests`, `docs`
- `build`, `ci`, `perf`, `refactor`

## 🔧 **Configuration Files**

### Core Configuration

- **`pyproject.toml`** - MyPy, Ruff, and test configuration
- **`semantic.yml`** - PR title validation rules
- **`dangerfile.ts`** - Automated code review logic

### Environment Requirements

- **Python 3.13** (primary target)
- **Node.js 20** (for Danger.js)
- **GitHub Actions** (Ubuntu latest)

## 📈 **Monitoring & Alerts**

### Automated Notifications

- Large PR warnings (>600 LOC)
- Critical system file modifications
- Security-sensitive code patterns
- Missing test coverage for source changes
- Documentation update reminders

### Performance Monitoring

- Performance-critical area change detection
- Benchmarking recommendations
- Resource usage tracking in CI

## 🚨 **Troubleshooting**

### Common Issues

1. **MyPy failures**: Ensure all functions have return type annotations
2. **Ruff violations**: Run `ruff format` to auto-fix formatting
3. **Test failures**: Check test isolation and async/await patterns
4. **Coverage drops**: Add tests for new code paths

### Getting Help

- Check workflow logs in GitHub Actions tab
- Review Danger.js comments on PRs
- Consult MyPy error codes documentation
- Reference Synndicate coding standards

## 🎯 **Next Steps**

### Phase 3 Integration

- [ ] Integrate with existing observability stack
- [ ] Add performance benchmarking automation  
- [ ] Enhance security scanning rules
- [ ] Add deployment pipeline integration

### Future Enhancements

- [ ] Advanced security policy enforcement
- [ ] Automated dependency updates
- [ ] Release automation workflows

---

**Built for Synndicate AI Platform** - Production-ready CI/CD with security-first approach and strict type safety.
