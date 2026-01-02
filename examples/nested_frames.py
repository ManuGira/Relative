"""Nested frames visualization example.

Creates 4 nested frames with the same TRS transformation applied at each level,
showing how transformations accumulate through the hierarchy.
"""

from coordinatus import Frame, Point, create_frame
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=== Nested Frames Visualization ===\n")
    
    # Create root frame
    root = Frame()
    
    # Create 4 nested frames, each with the same TRS transform
    # trs2D(tx=1, ty=0, angle_rad=pi/4, sx=0.75, sy=0.75)
    frames = [root]
    
    print("Creating nested frames with transform: trs2D(1, 0, π/4, 0.75, 0.75)")
    for i in range(10):
        frame = create_frame(
            parent=frames[-1],
            tx=1,
            ty=0,
            angle_rad=np.pi/4,
            sx=1/2**0.5,
            sy=1/2**0.5
        )
        frames.append(frame)
        print(f"  Frame {i+1} created")
    
    # Define 4 local points for each frame (unit triangle corners)
    local_points = [
        (0, 0),  # Origin
        (1, 0),  # Right
        (0, 1),  # Top
    ]
    
    print(f"\nLocal points in each frame: {local_points}")
    print("\nConverting all points to absolute coordinates...\n")
    
    # Collect all absolute points, organized by frame
    all_absolute_points = []
    frame_labels = []
    
    for frame_idx, frame in enumerate(frames):
        frame_points = []
        for x, y in local_points:
            point = Point(np.array([x, y]), frame=frame)
            absolute = point.to_absolute()
            frame_points.append((absolute.coords[0], absolute.coords[1]))
            all_absolute_points.append((absolute.coords[0], absolute.coords[1]))
            frame_labels.append(frame_idx)
        
        print(f"Frame {frame_idx} absolute points:")
        for i, (ax, ay) in enumerate(frame_points):
            print(f"  Local {local_points[i]} → Absolute ({ax:.3f}, {ay:.3f})")
    
    # Prepare data for plotting
    x_coords = [p[0] for p in all_absolute_points]
    y_coords = [p[1] for p in all_absolute_points]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
   
    # Plot each frame's points with a different color
    for frame_idx in range(len(frames)):
        # Get points for this frame
        frame_mask = [i for i, label in enumerate(frame_labels) if label == frame_idx]
        frame_x = [x_coords[i] for i in frame_mask]
        frame_y = [y_coords[i] for i in frame_mask]
        
        # Plot the triangle for this frame
        # Close the triangle by adding the first point at the end
        triangle_x = frame_x + [frame_x[0]]
        triangle_y = frame_y + [frame_y[0]]
        
        ax.plot(triangle_x, triangle_y, '.-', 
                linewidth=2, 
                markersize=8,
                label=f'Frame {frame_idx}',
                alpha=0.7)
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (absolute)', fontsize=12)
    ax.set_ylabel('Y (absolute)', fontsize=12)
    ax.set_title('Nested Frames with Accumulated TRS Transformations\n' +
                 'Each frame: translate(1,0), rotate(π/4), scale(√2/2)',
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add origin marker
    ax.plot(0, 0, 'k*', markersize=15, label='World Origin', zorder=10)
    
    plt.tight_layout()
    plt.savefig('examples/nested_frames.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/nested_frames.png")
    plt.show()


if __name__ == "__main__":
    main()
