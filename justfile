# Spio project commands

# Default recipe: list available commands
default:
    @just --list

# Run all tests with parallel compilation
test:
    SPIO_WORKERS=$(nproc) pytest tests

# Run tests with limited params for faster iteration
test-quick *ARGS:
    SPIO_MAX_TEST_PARAMS=5 SPIO_WORKERS=$(nproc) pytest tests {{ARGS}}

# Run smoke tests (quick regression check, excluding slow)
smoke:
    SPIO_ENABLE_CPP_TESTS=1 pytest -m "smoke and not slow"

# Run all smoke tests including slow ones
smoke-all:
    SPIO_ENABLE_CPP_TESTS=1 pytest -m "smoke"

# Run C++ tests
test-cpp *FILTER:
    SPIO_ENABLE_CPP_TESTS=1 pytest -s tests/test_cpp.py {{ if FILTER != "" { "--" } else { "" } }} {{ if FILTER != "" { "-k" } else { "" } }} {{FILTER}}

# Run a specific test file
test-file FILE *ARGS:
    SPIO_WORKERS=$(nproc) pytest {{FILE}} {{ARGS}}

# Run tests excluding slow ones
test-fast:
    SPIO_WORKERS=$(nproc) pytest -m "not slow"

# Activate virtual environment (prints command to eval)
venv:
    @echo "source .venv/bin/activate"
