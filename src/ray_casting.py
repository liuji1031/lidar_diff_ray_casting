"""Differential ray casting to render point clouds from voxel grids."""

import torch

from src.fast_voxel_traversal import (
    fast_voxel_traversal,
    get_voxel_centers_3d,
)


def cal_expected_return(
    dist: torch.Tensor, occupancy_prob: torch.Tensor, out_of_grid_dist: float
) -> torch.Tensor:
    """Calculate the expected return for a ray from occupancy probability.
    Args:
        dist (torch.Tensor): Distances to the voxel centers along the ray, shape [N].
        occupancy_prob (torch.Tensor): Occupancy probabilities of the intersected voxels
        along the ray, shape [N].
        out_of_grid_dist (float): Distance to the nearest voxel outside the grid.
    Returns:
        torch.Tensor: Expected return for the ray.
    """
    # Compute the return probability for each voxel
    assert dist.shape == occupancy_prob.shape, (
        "dist and occupancy_prob must have the same shape"
    )
    expected_return = 0.0
    not_return_prev_prob: float | torch.Tensor = 1.0
    for d, p in zip(dist, occupancy_prob):
        # not return prob times the prob of returning at this voxel * d
        expected_return += not_return_prev_prob * p * d
        not_return_prev_prob *= 1 - p  # Update the not return probability

    # Add the contribution from the distance to the nearest voxel outside the grid
    expected_return += not_return_prev_prob * out_of_grid_dist
    return expected_return


def get_out_of_grid_dist(
    sensor_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    grid_boundaries: torch.Tensor,
) -> torch.Tensor:
    """Calculate the distance from the sensor origin to the nearest point outside the grid.
    Args:
        sensor_origin (torch.Tensor): Origin of the sensor rays, shape [N, 3].
        ray_direction (torch.Tensor): Direction of the rays, shape [N, 3].
        grid_boundaries (torch.Tensor): Boundaries of the voxel grid, shape [2, 3].
    Returns:
        torch.Tensor: Distance to the nearest voxel outside the grid, shape [N].
    """
    # check if the sensor origin is within the grid
    assert (
        sensor_origin[0] >= grid_boundaries[0][0]
        and sensor_origin[0] <= grid_boundaries[1][0]
    ), "Sensor origin x-coordinate is out of grid boundaries."
    assert (
        sensor_origin[1] >= grid_boundaries[0][1]
        and sensor_origin[1] <= grid_boundaries[1][1]
    ), "Sensor origin y-coordinate is out of grid boundaries."
    assert (
        sensor_origin[2] >= grid_boundaries[0][2]
        and sensor_origin[2] <= grid_boundaries[1][2]
    ), "Sensor origin z-coordinate is out of grid boundaries."
    # Calculate distances to each boundary along the ray direction
    t_min = (grid_boundaries[0] - sensor_origin) / ray_direction
    t_max = (grid_boundaries[1] - sensor_origin) / ray_direction

    tmp = torch.concat([t_min, t_max], dim=-1)
    # find the minimum among all the positive values in tmp
    t_min = torch.min(tmp[tmp > 0], dim=-1).values

    # Get the maximum distance to exit the grid
    out_of_grid_dist = torch.norm(ray_direction * t_min, dim=-1)
    
    return out_of_grid_dist


def diff_ray_casting_3d(
    sensor_origin: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: float,
    occupancy: torch.Tensor,
):
    """Perform differential ray casting in 3D to find voxel centers intersected by
    rays.
    Args:
        sensor_origin (torch.Tensor): Origin of the sensor rays, shape [N, 3].
        ray_directions (torch.Tensor): Directions of the rays, shape [N, 3].
        grid_boundaries (torch.Tensor): Boundaries of the voxel grid, shape [2, 3].
        cell_size (float): Size of each voxel cell.
        occupancy (torch.Tensor): Occupancy grid, shape [D1, D2, D3].
    Returns:

    """
    # Ensure inputs are on the same device
    device = sensor_origin.device
    ray_directions = ray_directions.to(device)
    grid_boundaries = grid_boundaries.to(device)

    # for each ray, get the indices of the voxels it intersects
    intersection_ind_list: list[torch.Tensor] = fast_voxel_traversal(
        sensor_origin, ray_directions, grid_boundaries, cell_size
    )
    if sensor_origin.dim() == 1:
        sensor_origin = sensor_origin.unsqueeze(0)
    expected_returns = torch.zeros(
        ray_directions.shape[0], device=device
    )  # Initialize expected returns
    for iray, (indices, ray_direction) in enumerate(
        zip(intersection_ind_list, ray_directions)
    ):
        voxel_centers = get_voxel_centers_3d(
            indices, grid_boundaries, cell_size
        )  # n by 3
        dist = torch.norm(
            voxel_centers - sensor_origin, dim=1, keepdim=False
        )  # n by 1
        # retrieve the occupancy probabilities for the intersected voxels
        occupancy_prob = occupancy[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ]  # n by 1

        # calculate the distance starting from the sensor origin along the ray to the
        # point where the ray exits the grid
        out_of_grid_dist = get_out_of_grid_dist(
            sensor_origin, ray_direction, grid_boundaries
        )

        # calculate the expected return for the ray
        expected_returns[iray] = cal_expected_return(
            dist, occupancy_prob, out_of_grid_dist
        )
    return expected_returns
