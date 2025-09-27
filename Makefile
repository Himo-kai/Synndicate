# Synndicate AI - Development and Audit Makefile

.PHONY: help install test lint format audit clean dev run

# Default target
help:
	@echo "Synndicate AI - Available targets:"
	@echo "  install     - Install dependencies and setup environment"
	@echo "  test        - Run test suite"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  audit       - Run comprehensive audit (coverage, linting, deps)"
	@echo "  clean       - Clean build artifacts"
	@echo "  dev         - Start development server"
	@echo "  run         - Run the application"

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -e .
	pip install pytest pytest-cov ruff black mypy

# Run tests
test:
	pytest tests/ -v --tb=short

# Run linting
lint:
	ruff check src/
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/
	ruff check src/ --fix

# Comprehensive audit target (as specified in requirements)
audit:
	@echo "🔍 Running comprehensive audit..."
	@mkdir -p artifacts
	pytest -q --maxfail=1 --disable-warnings --cov=synndicate --cov-report=xml:artifacts/coverage.xml src/ tests/
	ruff check src/ | tee artifacts/ruff.txt || true
	python -m pip list --format=json > artifacts/pip_freeze.json
	@echo "✅ Audit complete - check artifacts/ directory"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development server (when API is implemented)
dev:
	@echo "🚀 Starting development server..."
	python -m uvicorn synndicate.api.server:app --reload --host 0.0.0.0 --port 8000

# Run application
run:
	python -m synndicate.main

# Security audit (placeholder for future Rust integration)
security-audit:
	@echo "🔒 Security audit (placeholder for Rust sandbox integration)"
	@echo "TODO: Implement Rust executor security checks"
	@echo "TODO: Scan for eval/exec usage"
	@echo "TODO: Validate --danger flag implementation"

# Generate project tree for audit bundle
tree:
	@mkdir -p synndicate_audit
	tree -a -h -L 4 -I "__pycache__|.git|target|node_modules|venv" > synndicate_audit/tree.txt || echo "tree command not available"

# Full audit bundle generation
audit-bundle: audit tree
	@echo "📦 Generating audit bundle..."
	@mkdir -p synndicate_audit/{configs,artifacts,logs,endpoints}
	@cp -r artifacts/* synndicate_audit/artifacts/ 2>/dev/null || true
	@echo "✅ Audit bundle ready in synndicate_audit/"
