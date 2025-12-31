# Running Python Scripts with UV

This project uses **`uv`** to manage Python scripts instead of the traditional `python` command.

## Usage

To run any Python script in this project, use:

```bash
uv run myscript.py
```

**NOT:**
```bash
python myscript.py
```

## Project Structure
All source code is located in the `src/coordinatus` directory. For example, to run a script located at `src/coordinatus/main.py`, use:

```bash
uv run src/coordinatus/transform.py
```

All tests are located in the `tests/` directory. The structure of the tests mirrors that of the `src/` directory. Test tests file names are prefixed with `test_`. For example, the test for `src/coordinatus/transform.py` would be located at `tests/coordinatus/test_transform.py`.


## Testing
tests are also run using `uv`.
To run all tests at once, use:
```bash
uv run pytest tests
```

run individual test files like so:
```bash
uv run pytest tests/test_example.py
```

For any automation or AI agent execution, always use the `uv run` command format.

## Python Typing Style

When using type hints, **prefer the built-in collection types** (`list`, `dict`, `tuple`, etc.) over importing from `typing` (e.g., avoid `from typing import List, Dict, Tuple`).
Use `list[str]`, `dict[str, int]`, etc., for type annotations.
