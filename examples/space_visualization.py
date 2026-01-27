"""Space-visualization scene.

Shows how the same scene (2 spaces and 2 points) looks different when viewed
from different reference spaces: absolute space, space 1, and space 2.
"""

from coordinatus import Space, Point, create_space
from coordinatus.transforms import trks2D
from coordinatus.visualization import draw_space_axes, draw_points
import numpy as np
import matplotlib.pyplot as plt


def setup_subplot(ax, title):
    """Setup common subplot properties."""
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')


def main():
    print("=== Space-Visualization Scene Visualization ===\n")
    
    # Create two non-nested spaces (both are root spaces)
    space1 = Space(
        parent=None,
        transform=trks2D(
            tx=-1, ty=3,
            angle_rad=-np.pi/4,  # -45 degrees
            kx=1, ky=0,
            sx=0.8, sy=0.8
        )
    )

    space2 = create_space(
        parent=None,
        tx=2, ty=1,
        angle_rad=np.pi/6,  # 30 degrees
        sx=1.2, sy=0.6
    )
    
    print("Space 1: trs2D(tx=2, ty=1, angle=π/6, sx=1.2, sy=1.2)")
    print("Space 2: trks2D(tx=-1, ty=3, angle=-π/4, kx=1, ky=0, sx=0.8, sy=0.8)")
    
    # Create 5 points in space 1 representing a F letter shape
    point1 = Point(np.array([2.0, 2.0]), space=space1)
    point2 = Point(np.array([1.0, 2.0]), space=space1)
    point3 = Point(np.array([1.0, 1.0]), space=space1)
    point4 = Point(np.array([1.0, 3.0]), space=space1)
    point5 = Point(np.array([2.5, 3.0]), space=space1)

    points = [point1, point2, point3, point4, point5]
       
    # Show coordinates in all three reference spaces
    print("\n--- Point Coordinates in Different Reference Spaces ---")
    
    print("\nIn Absolute Space:")
    for i, p in enumerate(points, 1):
        abs_coords = p.to_absolute().coords
        print(f"  P{i}: ({abs_coords[0]:.3f}, {abs_coords[1]:.3f})")
    
    print("\nIn Space 1 (original):")
    for i, p in enumerate(points, 1):
        f1_coords = p.relative_to(space1).coords
        print(f"  P{i}: ({f1_coords[0]:.3f}, {f1_coords[1]:.3f})")
    
    print("\nIn Space 2:")
    for i, p in enumerate(points, 1):
        f2_coords = p.relative_to(space2).coords
        print(f"  P{i}: ({f2_coords[0]:.3f}, {f2_coords[1]:.3f})")
    
    # Create visualization with 3 subplots

    items = [
        (space1, 'Space1', 'blue'),
        (None, 'Absolute', 'black'),
        (space2, 'Space2', 'green')
    ]

    for view_space, label, color in items:
        _, ax = plt.subplots(figsize=(8, 6))
        draw_space_axes(ax, space1, view_space, color='blue', label=label, alpha=1.0 if view_space == space1 else 0.4)
        draw_space_axes(ax, None, view_space, color='black', label=label, alpha=1.0 if view_space is None else 0.4)
        draw_space_axes(ax, space2, view_space, color='green', label=label, alpha=1.0 if view_space == space2 else 0.4)

        draw_points(ax, points, view_space, color='red')
        setup_subplot(ax, f'View from {label}' if view_space else 'View from Absolute Space')

        plt.tight_layout()
        print(f"\nPlot saved to: examples/space_visualization_{label}.png")
        # plt.draw()
        plt.savefig(f'examples/space_visualization_{label}.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()
