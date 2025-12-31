# Examples

This folder contains practical examples demonstrating how to use the Coordinatus package.

## Running Examples

All examples can be run using `uv`:

```bash
# Frame basis vectors visualization (requires matplotlib)
uv run --group dev examples/frame_basis_vectors.py

# Frame-perspective visualization (requires matplotlib)
uv run --group dev examples/frame_visualization.py
```

## Examples Overview

### `nested_frames.py`
Visualizes how transformations accumulate through nested frames:
- 10 nested frames with identical TRS transformations
- Plotting unit squares at each hierarchy level
- Shows translation, rotation, and scaling accumulation
- Creates a visual plot using matplotlib

### `frame_visualization.py`
Demonstrates how the same scene looks from different reference frames:
- Two independent frames (not nested) with different transformations
- Two points defined in Frame 1
- Three side-by-side views: absolute space, from Frame 1, from Frame 2
- Shows how coordinates change depending on the observer's reference frame
- Illustrates the relativity principle in coordinate transformations

