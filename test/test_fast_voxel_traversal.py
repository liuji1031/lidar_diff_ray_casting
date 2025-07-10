"""Test script for fast voxel traversal algorithm."""

import torch
import numpy as np
import os

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.fast_voxel_traversal import (
    fast_voxel_traversal_2d,
    fast_voxel_traversal_3d,
    fast_voxel_traversal,
    get_voxel_centers_2d,
    get_voxel_centers_3d,
)


def test_2d_traversal():
    """Test 2D voxel traversal with visualization."""
    print("Testing 2D Fast Voxel Traversal...")

    # Set up 2D grid
    grid_boundaries = torch.tensor(
        [[-5.0, -5.0], [5.0, 5.0]], dtype=torch.float32
    )
    cell_size = 0.5

    # Define start point and ray directions
    start_point = torch.tensor([0.2, 0.1], dtype=torch.float32)
    ray_directions = torch.tensor(
        [
            [1.0, 0.0],  # East
            [0.0, 1.0],  # North
            [1.0, 1.0],  # Northeast
            [-1.0, 1.0],  # Northwest
            [-1.0, -1.0],  # Southwest
            [-1.0,0.0],  # West
            [0.0, -1.0],  # South
            [0.7, -0.3],  # Southeast-ish
        ],
        dtype=torch.float32,
    )

    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(
        ray_directions, dim=1, keepdim=True
    )

    # Run traversal
    results = fast_voxel_traversal_2d(
        start_point, ray_directions, grid_boundaries, cell_size
    )

    print(f"Number of rays: {len(results)}")
    for i, voxels in enumerate(results):
        print(f"Ray {i}: {len(voxels)} voxels traversed")
        print(f"  First few voxels: {voxels[:5].tolist()}")

        # Convert to world coordinates
        world_coords = get_voxel_centers_2d(voxels, grid_boundaries, cell_size)
        print(f"  World coordinates: {world_coords.tolist()}")

    return results, start_point, ray_directions, grid_boundaries, cell_size


def test_3d_traversal():
    """Test 3D voxel traversal."""
    print("\nTesting 3D Fast Voxel Traversal...")

    # Set up 3D grid
    grid_boundaries = torch.tensor(
        [[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]], dtype=torch.float32
    )
    cell_size = 0.25

    # Define start point and ray directions
    start_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ray_directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # X-axis
            [0.0, 1.0, 0.0],  # Y-axis
            [0.0, 0.0, 1.0],  # Z-axis
            [1.0, 1.0, 1.0],  # Diagonal
            [1.0, -1.0, 0.5],  # Mixed direction
        ],
        dtype=torch.float32,
    )

    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(
        ray_directions, dim=1, keepdim=True
    )

    # Run traversal
    results = fast_voxel_traversal_3d(
        start_point, ray_directions, grid_boundaries, cell_size
    )

    print(f"Number of rays: {len(results)}")
    for i, voxels in enumerate(results):
        print(f"Ray {i}: {len(voxels)} voxels traversed")
        print(f"  First few voxels: {voxels[:5].tolist()}")

        # Convert to world coordinates
        world_coords = get_voxel_centers_3d(voxels, grid_boundaries, cell_size)
        print(f"  World coordinates: {world_coords[:3].tolist()}")

    return results


def test_unified_interface():
    """Test the unified interface."""
    print("\nTesting Unified Interface...")

    # Test 2D case
    grid_boundaries_2d = torch.tensor(
        [[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float32
    )
    start_point_2d = torch.tensor([0.0, 0.0], dtype=torch.float32)
    ray_directions_2d = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    ray_directions_2d = ray_directions_2d / torch.norm(
        ray_directions_2d, dim=1, keepdim=True
    )

    results_2d = fast_voxel_traversal(
        start_point_2d, ray_directions_2d, grid_boundaries_2d, 0.5
    )
    print(f"2D unified: {len(results_2d[0])} voxels")

    # Test 3D case
    grid_boundaries_3d = torch.tensor(
        [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=torch.float32
    )
    start_point_3d = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ray_directions_3d = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    ray_directions_3d = ray_directions_3d / torch.norm(
        ray_directions_3d, dim=1, keepdim=True
    )

    results_3d = fast_voxel_traversal(
        start_point_3d, ray_directions_3d, grid_boundaries_3d, 0.5
    )
    print(f"3D unified: {len(results_3d[0])} voxels")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases...")

    # Test with different cell sizes
    grid_boundaries = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
    )
    start_point = torch.tensor([0.0, 0.0], dtype=torch.float32)
    ray_directions = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    # Scalar cell size
    results1 = fast_voxel_traversal_2d(
        start_point, ray_directions, grid_boundaries, 0.2
    )
    print(f"Scalar cell size: {len(results1[0])} voxels")

    # Tensor cell size (uniform)
    cell_size_tensor = torch.tensor(0.2, dtype=torch.float32)
    results2 = fast_voxel_traversal_2d(
        start_point, ray_directions, grid_boundaries, cell_size_tensor
    )
    print(f"Tensor cell size (uniform): {len(results2[0])} voxels")

    # Tensor cell size (non-uniform)
    cell_size_nonuniform = torch.tensor([0.2, 0.3], dtype=torch.float32)
    results3 = fast_voxel_traversal_2d(
        start_point, ray_directions, grid_boundaries, cell_size_nonuniform
    )
    print(f"Non-uniform cell size: {len(results3[0])} voxels")

    # Test with max_steps
    results4 = fast_voxel_traversal_2d(
        start_point, ray_directions, grid_boundaries, 0.2, max_steps=5
    )
    print(f"With max_steps=5: {len(results4[0])} voxels")


def test_batch_processing():
    """Test batch processing with multiple start points."""
    print("\nTesting Batch Processing...")

    grid_boundaries = torch.tensor(
        [[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float32
    )

    # Multiple start points
    start_points = torch.tensor(
        [[0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
    )

    ray_directions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=torch.float32
    )
    ray_directions = ray_directions / torch.norm(
        ray_directions, dim=1, keepdim=True
    )

    results = fast_voxel_traversal_2d(
        start_points, ray_directions, grid_boundaries, 0.25
    )

    print(f"Batch processing: {len(results)} rays")
    for i, voxels in enumerate(results):
        print(f"  Ray {i}: {len(voxels)} voxels")


def visualize_2d_traversal(
    results, start_point, ray_directions, grid_boundaries, cell_size
):
    """Visualize 2D traversal results."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return

    try:
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Draw grid
        grid_min = grid_boundaries[0].numpy()
        grid_max = grid_boundaries[1].numpy()

        if isinstance(cell_size, torch.Tensor):
            cell_size_np = cell_size.numpy()
        else:
            cell_size_np = np.array([cell_size, cell_size])

        # Draw grid lines
        x_lines = np.arange(
            grid_min[0], grid_max[0] + cell_size_np[0], cell_size_np[0]
        )
        y_lines = np.arange(
            grid_min[1], grid_max[1] + cell_size_np[1], cell_size_np[1]
        )

        for x in x_lines:
            ax.axvline(x, color="lightgray", linewidth=0.5)
        for y in y_lines:
            ax.axhline(y, color="lightgray", linewidth=0.5)

        # Plot start point
        ax.plot(
            start_point[0],
            start_point[1],
            "ro",
            markersize=8,
            label="Start Point",
        )

        # Plot traversed voxels for each ray
        colors = ["blue", "green", "orange", "purple", "brown"]

        for i, (voxels, ray_dir) in enumerate(zip(results, ray_directions)):
            if len(voxels) == 0:
                continue

            color = colors[i % len(colors)]

            # Convert voxel indices to world coordinates
            world_coords = get_voxel_centers_2d(
                voxels, grid_boundaries, cell_size
            )

            # Plot voxel centers
            ax.scatter(
                world_coords[:, 0],
                world_coords[:, 1],
                c=color,
                s=20,
                alpha=0.7,
                label=f"Ray {i}",
            )

            # Draw ray direction
            ray_end = start_point + ray_dir * 3.0  # Scale for visibility
            ax.arrow(
                start_point[0],
                start_point[1],
                ray_end[0] - start_point[0],
                ray_end[1] - start_point[1],
                head_width=0.1,
                head_length=0.1,
                fc=color,
                ec=color,
                alpha=0.5,
            )

        ax.set_xlim(grid_min[0] - 0.5, grid_max[0] + 0.5)
        ax.set_ylim(grid_min[1] - 0.5, grid_max[1] + 0.5)
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title("2D Fast Voxel Traversal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "2d_voxel_traversal.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"2D visualization saved as '{save_path}'")

    except Exception as e:
        print(f"Error creating visualization: {e}")


def main():
    """Run all tests."""
    print("Fast Voxel Traversal Algorithm Tests")
    print("=" * 50)

    # Test 2D traversal
    results_2d, start_point, ray_directions, grid_boundaries, cell_size = (
        test_2d_traversal()
    )

    # Test 3D traversal
    test_3d_traversal()

    # Test unified interface
    test_unified_interface()

    # Test edge cases
    test_edge_cases()

    # Test batch processing
    test_batch_processing()

    # Visualize 2D results
    visualize_2d_traversal(
        results_2d, start_point, ray_directions, grid_boundaries, cell_size
    )

    print("\n" + "=" * 50)
    print("All tests completed successfully!")


if __name__ == "__main__":
    # main()
    results_2d, start_point, ray_directions, grid_boundaries, cell_size = (
        test_2d_traversal()
    )
    visualize_2d_traversal(
        results_2d, start_point, ray_directions, grid_boundaries, cell_size
    )
