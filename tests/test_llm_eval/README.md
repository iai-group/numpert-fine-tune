# OllamaLLM Test Suite

This directory contains comprehensive tests for the `ollama_lib.py` module, which provides a wrapper for Ollama chat completions.

## Test Files

- `test_ollama_lib.py` - Main test suite covering core functionality
- `test_ollama_lib_advanced.py` - Advanced tests for thinking functionality and edge cases
- `conftest.py` - Shared pytest fixtures
- `run_tests.py` - Test runner script
- `requirements.txt` - Test dependencies

## Features Tested

### Core Functionality
- ✅ OllamaLLM initialization and configuration loading
- ✅ Chat completion requests with proper parameter handling
- ✅ Model pulling when models are not found (404 errors)
- ✅ Error handling for various HTTP status codes
- ✅ Model capability checking

### Thinking Functionality
- ✅ Thinking parameter configuration from YAML
- ✅ Thinking enabled/disabled scenarios
- ✅ Proper thinking output handling
- ✅ Alternative thinking parameter names

### Configuration Management
- ✅ YAML configuration loading and validation
- ✅ Different format specifications (JSON, YAML, XML)
- ✅ Stream configuration options
- ✅ Model options and parameters

### Edge Cases
- ✅ Empty or malformed configuration files
- ✅ Network timeouts and connection errors
- ✅ Multiple consecutive errors
- ✅ Special characters in model names
- ✅ Empty message lists

## Running the Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# Using the test runner
python run_tests.py

# Using pytest directly
pytest test_ollama_lib.py test_ollama_lib_advanced.py -v
```

### Run Specific Tests

```bash
# Run only thinking-related tests
python run_tests.py -k thinking

# Run only core functionality tests
pytest test_ollama_lib.py -v

# Run only advanced/edge case tests
pytest test_ollama_lib_advanced.py -v
```

### Run with Coverage

```bash
pytest --cov=ollama_lib --cov-report=html
```

### Run Specific Test Classes or Methods

```bash
# Run specific test class
pytest test_ollama_lib.py::TestOllamaLLM -v

# Run specific test method
pytest test_ollama_lib.py::TestOllamaLLM::test_chat_success -v
```

## Test Structure

### Mock Objects

The tests use comprehensive mock objects to simulate Ollama server responses:

- `MockMessage` - Simulates Ollama response messages
- `MockResponse` - Simulates complete Ollama chat responses
- `MockThinkingResponse` - Specialized for testing thinking functionality

### Fixtures

- `base_config` - Base configuration dictionary
- `temp_config_file` - Temporary YAML configuration file
- `mock_ollama_client` - Mock Ollama client with common methods
- `sample_messages` - Basic test messages
- `fact_check_messages` - Fact-checking specific messages

### Test Categories

1. **Initialization Tests**
   - Configuration loading
   - Error handling for missing/invalid files
   - Parameter validation

2. **Chat Functionality Tests**
   - Successful chat completions
   - Format detection and handling
   - Parameter passing verification

3. **Error Handling Tests**
   - Model not found (404) scenarios
   - Other HTTP errors
   - Network issues

4. **Thinking Feature Tests**
   - Thinking parameter configuration
   - Thinking output validation
   - Alternative parameter formats

5. **Integration Tests**
   - Complete workflows
   - Error recovery chains
   - Multi-step operations

## Mocking Strategy

The tests mock the Ollama client to avoid dependencies on:
- Running Ollama server
- Network connectivity
- Specific model availability
- External API calls

This ensures:
- ✅ Fast test execution
- ✅ Reliable and repeatable results
- ✅ No external dependencies
- ✅ Comprehensive error scenario testing

## Example Test Output

```
tests/test_llm_eval/test_ollama_lib.py::TestOllamaLLM::test_init_success PASSED
tests/test_llm_eval/test_ollama_lib.py::TestOllamaLLM::test_chat_success PASSED
tests/test_llm_eval/test_ollama_lib.py::TestOllamaLLM::test_chat_model_not_found PASSED
tests/test_llm_eval/test_ollama_lib_advanced.py::TestOllamaLLMThinking::test_thinking_enabled_in_config PASSED
...
```

## Contributing

When adding new tests:

1. Use descriptive test names that explain what is being tested
2. Include both positive and negative test cases
3. Mock external dependencies appropriately
4. Add docstrings explaining test purpose
5. Use parametrized tests for testing multiple scenarios
6. Follow the existing test structure and patterns
