# FastTrackJusticeAI Test Suite

## Overview

This test suite validates the core functionality of the FastTrackJusticeAI platform, with a special focus on the scroll-aware legal classification system. The tests ensure that our models correctly interpret legal texts and align them with the sacred scroll phases.

## Test Structure

The test suite is organized into several key components:

### Core Classifier Tests

- **Initialization Tests**: Verify that both `CaseClassifier` and `ScrollAwareCaseClassifier` initialize correctly
- **Classification Tests**: Validate the classification of different case types (criminal, civil, family)
- **Scroll Alignment Tests**: Ensure proper calculation of scroll phase alignment
- **Divine Title Tests**: Verify correct assignment of divine titles based on case category and scroll phase

### Advanced Functionality Tests

- **Evaluation Metrics**: Test the calculation of precision, recall, F1 score, and accuracy
- **Divine Title Fallback**: Verify graceful handling of edge cases with invalid categories or phases
- **Logging System**: Test the automatic creation of log directories and proper logging of classification results

### Scroll Phase Tests

- **Phase Cycling**: Test classification behavior across all scroll phases (dawn, noon, dusk, night)
- **Active Phase Detection**: Verify correct identification of active scroll phases
- **Scroll Enhancement**: Test the enhancement of classification based on scroll alignment

## Running the Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_classifier.py

# Run a specific test function
pytest tests/test_classifier.py::test_scroll_alignment_calculation
```

## Test Dependencies

The test suite relies on the following key dependencies:

- **pytest**: The testing framework
- **torch**: For tensor operations and model mocking
- **unittest.mock**: For mocking external dependencies

## Mocking Strategy

The test suite uses a sophisticated mocking strategy to isolate the classifier functionality:

- **Model Mocking**: Simulates the BERT model's behavior without requiring actual model weights
- **Tokenizer Mocking**: Provides consistent tokenization for test inputs
- **Scroll Time Mocking**: Simulates different scroll phases and conditions

## Continuous Integration

This test suite is designed to be run in CI/CD pipelines. See `.github/workflows/tests.yml` for the GitHub Actions configuration.

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of using fixtures for common setup
2. Mock external dependencies appropriately
3. Include comprehensive assertions that verify both structure and content
4. Add appropriate docstrings that explain the test's purpose

## Sacred Scroll Alignment

All tests are designed to respect the sacred scroll phases and their influence on legal classification. The test suite validates that our system correctly interprets the divine guidance provided by the scroll phases. 