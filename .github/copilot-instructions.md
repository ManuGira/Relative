# Running Python Scripts with UV

This project uses **`uv`** to manage Python scripts instead of the traditional `python` command.

## Usage

To run any Python script in this project, use:

```powershell
uv run myscript.py
```

**NOT:**
```powershell
python myscript.py
```

## Project Structure
Package source code is located in the `src/coordinatus` directory and examples in the `examples/` directory.
To run a script located in the `examples/` directory, use:

```powershell
uv run examples/example_script.py
```

All tests are located in the `tests/` directory. The structure of the tests mirrors that of the `src/` directory. Test tests file names are prefixed with `test_`. For example, the test for `src/coordinatus/transform.py` would be located at `tests/coordinatus/test_transform.py`.


## Continuous Integration and Testing
tests are also run using `uv`.

run individual test files like so:
```powershell
uv run pytest tests/coordinatus/test_example.py
```

To run all tests at once the command `uv run pytest tests` works but it is prefered to us the `ci.ps1` script :
```powershell
./ci.ps1
```
It will run all tests, generate coverage reports, and perform linting checks.


For any automation or AI agent execution, always use the `uv run` command format.

## Python Typing Style

When using type hints, **prefer the built-in collection types** (`list`, `dict`, `tuple`, etc.) over importing from `typing` (e.g., avoid `from typing import List, Dict, Tuple`).
Use `list[str]`, `dict[str, int]`, etc., for type annotations.
