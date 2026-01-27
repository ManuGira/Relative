# Examples

This folder contains practical examples demonstrating how to use the Coordinatus package.

## Running Examples

All examples can be run using `uv`:

```bash
# Space basis vectors visualization (requires matplotlib)
uv run --group dev examples/space_basis_vectors.py

# Space-perspective visualization (requires matplotlib)
uv run --group dev examples/space_visualization.py
```

## Examples Overview

### `nested_spaces.py`
Visualizes how transformations accumulate through nested spaces:
- 10 nested spaces with identical TRS transformations
- Plotting unit squares at each hierarchy level
- Shows translation, rotation, and scaling accumulation
- Creates a visual plot using matplotlib

### `space_visualization.py`
Demonstrates how the same scene looks from different reference spaces:
- Two independent spaces (not nested) with different transformations
- Two points defined in Space 1
- Three side-by-side views: absolute space, from Space 1, from Space 2
- Shows how coordinates change depending on the observer's reference space
- Illustrates the relativity principle in coordinate transformations

