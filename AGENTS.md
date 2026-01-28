# Kimi1 Project - AI Coding Agent Guide

## Project Overview

Kimi1 is an AI-focused software project currently in initialization phase. This document provides essential guidance for AI coding agents working on this project.

## Current Status

⚠️ **PROJECT INITIALIZATION**: This project directory is currently empty. The following guidelines establish conventions and expectations for project development.

## Technology Stack (To Be Determined)

As this is a new project, the technology stack will be determined based on project requirements. Common AI/ML project stacks include:

### Python-Based AI Projects
- **Language**: Python 3.8+
- **ML Frameworks**: PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers
- **Data Processing**: pandas, numpy, scipy
- **API Framework**: FastAPI, Flask, or Django
- **Environment**: conda, pipenv, or poetry
- **Configuration**: pyproject.toml, requirements.txt

### JavaScript/Node.js AI Projects
- **Runtime**: Node.js 16+
- **ML Libraries**: TensorFlow.js, ONNX.js
- **Web Framework**: Express.js, Next.js
- **Package Manager**: npm or yarn
- **Configuration**: package.json, tsconfig.json

## Project Structure Guidelines

### Recommended Directory Structure
```
kimi1/
├── src/                    # Source code
│   ├── models/            # ML models and algorithms
│   ├── data/              # Data processing utilities
│   ├── api/               # API endpoints and services
│   ├── utils/             # Utility functions
│   └── config/            # Configuration files
├── tests/                 # Test files
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks (if applicable)
├── config/                # Configuration files
├── .github/               # GitHub workflows
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Python project configuration
├── package.json           # Node.js dependencies (if applicable)
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-container setup
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
└── AGENTS.md             # This file
```

## Development Conventions

### Code Style
- Follow PEP 8 for Python projects
- Use ESLint/Prettier for JavaScript/TypeScript projects
- Maintain consistent indentation (4 spaces for Python, 2 spaces for JS)
- Use meaningful variable and function names
- Add docstrings to all public functions and classes

### Version Control
- Use conventional commits: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`
- Create feature branches: `feature/description` or `feat/description`
- Use pull requests for code review
- Maintain a clean git history

### Documentation
- Document all public APIs and functions
- Include type hints for Python code
- Maintain up-to-date README.md
- Update AGENTS.md when project structure changes

## Testing Strategy

### Test Requirements
- Write unit tests for all core functionality
- Maintain minimum 80% code coverage
- Use pytest for Python projects
- Use Jest/Mocha for JavaScript projects
- Include integration tests for API endpoints
- Test edge cases and error conditions

### Test Structure
```
tests/
├── conftest.py           # Pytest configuration
├── test_models.py        # Model tests
├── test_api.py           # API tests
├── test_utils.py         # Utility tests
└── fixtures/             # Test data and mocks
```

## Security Considerations

### Best Practices
- Never commit API keys or secrets to version control
- Use environment variables for sensitive configuration
- Validate all user inputs
- Implement proper authentication and authorization
- Use HTTPS for all API communications
- Regularly update dependencies for security patches

### Security Checklist
- [ ] No hardcoded secrets in code
- [ ] Input validation implemented
- [ ] SQL injection prevention (if applicable)
- [ ] XSS protection (if web interface)
- [ ] Rate limiting for API endpoints
- [ ] Secure session management

## Build and Deployment

### Development Environment
```bash
# Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
# or
poetry install
```

### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### Build Process
```bash
# Python package build
python -m build

# Docker build
docker build -t kimi1 .

# Docker compose
docker-compose up
```

## AI/ML Specific Guidelines

### Model Development
- Version control model files using DVC or similar
- Document model architectures and hyperparameters
- Include model performance metrics and benchmarks
- Use proper train/validation/test splits
- Implement model monitoring and logging

### Data Handling
- Document data sources and preprocessing steps
- Ensure data privacy and compliance (GDPR, etc.)
- Use appropriate data validation techniques
- Implement data versioning
- Consider data bias and fairness

### Model Deployment
- Containerize models for consistent deployment
- Implement proper model versioning
- Monitor model performance in production
- Plan for model retraining and updates
- Consider model explainability requirements

## Common Commands Reference

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Stage changes
git add .

# Commit with conventional format
git commit -m "feat: add new model architecture"

# Push to remote
git push origin feature/new-feature
```

### Development Tools
```bash
# Code formatting (Python)
black src/
isort src/

# Type checking (Python)
mypy src/

# Linting (Python)
flake8 src/
pylint src/

# Security scanning
bandit -r src/
safety check
```

## Project-Specific Notes

### To Be Updated
- [ ] Technology stack selection
- [ ] Project requirements and goals
- [ ] Specific dependencies and versions
- [ ] Deployment architecture
- [ ] Performance requirements
- [ ] Integration specifications

### Next Steps
1. Define project requirements and scope
2. Select appropriate technology stack
3. Set up initial project structure
4. Configure development environment
5. Establish CI/CD pipeline
6. Create initial documentation

## Resources and References

### AI/ML Development
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Best Practices
- [Python Best Practices](https://realpython.com/python-best-practices/)
- [Clean Code Principles](https://github.com/ryanmcdermott/clean-code-javascript)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

---

**Last Updated**: January 28, 2026
**Project Status**: Initialization Phase
**Next Review**: After technology stack selection