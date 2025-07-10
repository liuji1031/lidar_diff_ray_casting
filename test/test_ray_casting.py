"""Test for ray casting functionality."""

import torch
import numpy as np

from src.ray_casting import cal_expected_return, get_out_of_grid_dist


def test_cal_expected_return():
    """Test the expected return calculation."""
    dist = torch.tensor([1.0, 2.0, 3.0])
    occupancy_prob = torch.tensor([0.8, 0.6, 0.4])
    out_of_grid_dist = 5.0
    expected = cal_expected_return(dist, occupancy_prob, out_of_grid_dist)
    expected_return = (
        1.0 * 0.8 * 1.0
        + 0.2 * 0.6 * 2.0
        + 0.2 * 0.4 * 0.4 * 3.0
        + (0.2 * 0.4 * 0.6 * 5.0)
    )
    assert torch.isclose(expected, torch.tensor(expected_return)), (
        f"Expected {expected_return}, got {expected.item()}"
    )


def test_out_of_grid_dist():
    """Test the out of grid distance calculation."""
    # Test 1
    sensor_origin = torch.tensor([0.5, 0.5, 0.5])
    ray_direction = torch.tensor([1.0, 0.2, 0.0])
    grid_boundaries = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    out_of_grid_dist = get_out_of_grid_dist(
        sensor_origin, ray_direction, grid_boundaries
    )
    expected_val = np.sqrt(0.5**2 + 0.1**2)
    assert torch.isclose(
        out_of_grid_dist,
        torch.tensor(expected_val, dtype=torch.float32),
        rtol=1e-5,
    ), f"Expected {expected_val}, got {out_of_grid_dist.item()}"

    # Test 2
    ray_direction = torch.tensor([1.0, 1.0, 1.0])
    out_of_grid_dist = get_out_of_grid_dist(
        sensor_origin, ray_direction, grid_boundaries
    )
    expected_val = np.sqrt(0.5**2 + 0.5**2 + 0.5**2)
    assert torch.isclose(
        out_of_grid_dist,
        torch.tensor(expected_val, dtype=torch.float32),
        rtol=1e-5,
    ), f"Expected {expected_val}, got {out_of_grid_dist.item()}"

    # Test 3, parallel to x-axis
    ray_direction = torch.tensor([1.0, 0.0, 0.0])
    out_of_grid_dist = get_out_of_grid_dist(
        sensor_origin, ray_direction, grid_boundaries
    )
    expected_val = np.sqrt(0.5**2 + 0.0**2 + 0.0**2)
    assert torch.isclose(
        out_of_grid_dist,
        torch.tensor(expected_val, dtype=torch.float32),
        rtol=1e-5,
    ), f"Expected {expected_val}, got {out_of_grid_dist.item()}"


def test_diff_ray_casting():
    """Test the differential ray casting functionality."""
    # Define sensor origin and ray directions
    sensor_origin = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
    grid_boundaries = torch.tensor(
        [[-5.0, -5.0, 0.0], [5.0, 5.0, 2.0]], dtype=torch.float32
    )
    cell_size = 0.2
    # create occupancy tensor based on grid boundaries and cell size
    occupancy_shape = (
        int((grid_boundaries[1][0] - grid_boundaries[0][0]) / cell_size),
        int((grid_boundaries[1][1] - grid_boundaries[0][1]) / cell_size),
        int((grid_boundaries[1][2] - grid_boundaries[0][2]) / cell_size),
    )
    occupancy = torch.zeros(occupancy_shape, dtype=torch.float32)

    # assign the ground plane occupancy
    occupancy[:, :, 0] = 1.0

    # add an infinite height cylinder at the center (1.0, 2.0)
    cylinder_center = (
        int((1.0 - grid_boundaries[0][0]) / cell_size),
        int((2.0 - grid_boundaries[0][1]) / cell_size),
    )
    # generate a meshgrid of grid centers
    x = torch.arange(
        grid_boundaries[0][0] + cell_size / 2,
        grid_boundaries[1][0],
        cell_size,
        dtype=torch.float32,
    )
    y = torch.arange(
        grid_boundaries[0][1] + cell_size / 2,
        grid_boundaries[1][1],
        cell_size,
        dtype=torch.float32,
    )
    z = torch.arange(
        grid_boundaries[0][2] + cell_size / 2,
        grid_boundaries[1][2],
        cell_size,
        dtype=torch.float32,
    )
    grid_centers = torch.meshgrid(x, y, z, indexing="ij")
    grid_centers = torch.stack(grid_centers, dim=-1)
    

if __name__ == "__main__":
    test_cal_expected_return()
    test_out_of_grid_dist()
    print("All tests passed.")
