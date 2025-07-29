"""Differential ray casting to render point clouds from voxel grids."""

import torch
import time
import numpy as np
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


from lidar_diff_ray_casting.fast_voxel_traversal_numba import (
    fast_voxel_traversal_numba,
)


@torch.jit.script
def cal_expected_return_batch_padded(
    distances: torch.Tensor,
    occupancy_probs: torch.Tensor,
    out_of_grid_dists: torch.Tensor,
    ray_steps: torch.Tensor,
) -> torch.Tensor:
    """Batch calculation of expected return for multiple rays using padding.

    This function processes multiple rays simultaneously by padding all rays to the same
    length. Padded voxels use occupancy_prob=1.0 and dist=0.0, which mathematically
    contributes 0 to the expected return and terminates the cumulative product.

    This approach eliminates all Python loops and enables true batch processing with
    maximum GPU parallelization.

    Args:
        distances (torch.Tensor): Padded distances for all rays, shape [num_rays, max_length].
                                 Padded positions should have dist=0.0.
        occupancy_probs (torch.Tensor): Padded occupancy probabilities, shape [num_rays, max_length].
                                       Padded positions should have occupancy=1.0.
        out_of_grid_dists (torch.Tensor): Out-of-grid distances, shape [num_rays].
        ray_steps (torch.Tensor): Actual length of each ray, shape [num_rays].

    Returns:
        torch.Tensor: Expected returns for all rays, shape [num_rays].
    """
    assert distances.shape == occupancy_probs.shape, (
        "distances and occupancy_probs must have the same shape"
    )
    assert distances.shape[0] == out_of_grid_dists.shape[0], (
        "Number of rays must match between distances and out_of_grid_dists"
    )
    assert distances.shape[0] == ray_steps.shape[0], (
        "Number of rays must match between distances and ray_steps"
    )

    num_rays, max_length = distances.shape
    device = distances.device
    dtype = distances.dtype

    # Handle empty case
    if max_length == 0:
        return out_of_grid_dists.clone()

    # Calculate "not return" probabilities for all rays simultaneously
    not_return_probs = 1.0 - occupancy_probs  # [num_rays, max_length]

    # Calculate cumulative "not return" probabilities for all rays in parallel
    # Prepend 1.0 and shift to get correct cumulative product
    ones = torch.ones(num_rays, 1, device=device, dtype=dtype)
    shifted_probs = torch.cat(
        [ones, not_return_probs[:, :-1]], dim=1
    )  # [num_rays, max_length]
    cumulative_not_return = torch.cumprod(
        shifted_probs, dim=1
    )  # [num_rays, max_length]

    # Compute contributions for all voxels simultaneously
    contributions = (
        cumulative_not_return * occupancy_probs * distances
    )  # [num_rays, max_length]

    # Sum along voxel dimension to get expected return per ray
    expected_returns = torch.sum(contributions, dim=1)  # [num_rays]

    # Add out-of-grid contributions
    # Compute final not-return probability for each ray (only considering actual ray length)
    # Create a mask for actual voxels vs padding
    ray_indices = (
        torch.arange(max_length, device=device)
        .unsqueeze(0)
        .expand(num_rays, -1)
    )
    valid_mask = ray_indices < ray_steps.unsqueeze(1)  # [num_rays, max_length]

    # Set padded positions to have not_return_prob = 1.0 (no effect on product)
    # and actual positions keep their computed not_return_prob
    masked_not_return_probs = torch.where(
        valid_mask, not_return_probs, torch.ones_like(not_return_probs)
    )

    # Compute final not-return probability for each ray using vectorized product
    final_not_return_probs = torch.prod(
        masked_not_return_probs, dim=1
    )  # [num_rays]

    expected_returns += final_not_return_probs * out_of_grid_dists

    return expected_returns


@torch.jit.script
def cal_expected_return_batch_flat_legacy(
    ray_distances: torch.Tensor,
    ray_occupancy_probs: torch.Tensor,
    ray_starts: torch.Tensor,
    ray_lengths: torch.Tensor,
    out_of_grid_dists: torch.Tensor,
) -> torch.Tensor:
    """Calculate expected return for concatenated multiple rays, no padding.

    Args:
        ray_distances (torch.Tensor): A flat array of distances of each voxel to the
        sensor traversed by each ray, shape [total_voxels]. Concatenated from all rays.
        ray_occupancy_probs (torch.Tensor): A flat array of occupancy probabilities
        traversed by each ray, shape [total_voxels]. Concatenated from all rays.
        ray_starts (torch.Tensor): Start index of each ray, shape [num_rays].
        ray_lengths (torch.Tensor): Length of each ray, shape [num_rays].
        out_of_grid_dists (torch.Tensor): Out-of-grid distances, shape [num_rays].
    Returns:
        torch.Tensor: Expected returns for all rays, shape [num_rays].
    """

    not_return_at_curr_probs = 1.0 - ray_occupancy_probs
    # shift to the right by one
    not_return_at_prev_probs = torch.zeros_like(
        not_return_at_curr_probs
    )
    not_return_at_prev_probs[1:] = not_return_at_curr_probs[:-1]
    # set all value at ray_starts to 1.0
    not_return_at_prev_probs[ray_starts] = 1.0
    not_return_at_prev_probs = torch.clip(not_return_at_prev_probs, min=1.0e-16)
    # take log of not_return_at_prev_probs
    log_not_return_at_prev_probs = torch.log(not_return_at_prev_probs)
    # find the cumulative sum of log_not_return_at_prev_probs
    cumsum_log_not_return_at_prev_probs = torch.cumsum(
        log_not_return_at_prev_probs, dim=0
    )
    # now we need to calculate the cumulative sum within each ray. To do so, we need to
    # find the index of the ray_starts. Suppose the starting index is i, and the length
    # of the ray is l, then for each element in this ray (index from i to i+l-1), we can
    # calculate the cumsum at each index by subtracting the cumsum at index i from the
    # cumsum at index i+l-1 (the cumsum at index i should always be 0)
    start_indices_repeated = ray_starts.repeat_interleave(ray_lengths)

    # do the subtraction in a vectorized way
    log_not_return_at_prev_probs_by_ray = (
        cumsum_log_not_return_at_prev_probs
        - cumsum_log_not_return_at_prev_probs[start_indices_repeated]
    )

    # take exp to get the not return at prev probs within each ray
    not_return_at_prev_probs_by_ray = torch.exp(
        log_not_return_at_prev_probs_by_ray
    )

    not_return_at_prev_probs_by_ray = torch.clip(
        not_return_at_prev_probs_by_ray, max=1.0
    )

    # calculate return at out-of-grid distance prob
    ray_ends_actual = ray_starts + ray_lengths - 1
    out_of_grid_return_probs = (
        not_return_at_prev_probs_by_ray[ray_ends_actual]  # prob not return before the last voxel
        * not_return_at_curr_probs[ray_ends_actual]  # prob not return at the last voxel
    )

    # multiply with prob of return (occupancy) and distances
    tmp = ray_occupancy_probs * not_return_at_prev_probs_by_ray * ray_distances

    # now perform the actual cumsum within each ray
    expected_returns_cumsum = torch.cumsum(tmp, dim=0)
    # append a zero at the beginning
    expected_returns_cumsum = torch.cat(
        [torch.zeros(1, device=ray_distances.device), expected_returns_cumsum],
        dim=0,
    )
    # now we can calculate the expected return for each ray
    ray_ends = ray_starts + ray_lengths
    expected_returns = (
        expected_returns_cumsum[ray_ends] - expected_returns_cumsum[ray_starts]
    )
    return expected_returns + out_of_grid_return_probs * out_of_grid_dists


@torch.jit.script
def cal_expected_return_batch_flat(
    ray_distances: torch.Tensor,
    ray_occupancy_probs: torch.Tensor,
    ray_starts: torch.Tensor,
    ray_lengths: torch.Tensor,
    out_of_grid_dists: torch.Tensor,
) -> torch.Tensor:
    """Fixed version that should produce gradients more consistent with padded version.
    
    Key fixes:
    1. Remove torch.clip operations that affect gradients
    2. Use more numerically stable log-space computation
    3. Better handling of edge cases
    """
    
    not_return_at_curr_probs = 1.0 - ray_occupancy_probs
    
    # Create shifted array more carefully
    not_return_at_prev_probs = torch.zeros_like(not_return_at_curr_probs)
    not_return_at_prev_probs[1:] = not_return_at_curr_probs[:-1]
    not_return_at_prev_probs[ray_starts] = 1.0
    
    # Instead of clipping, use a small epsilon added directly to avoid log(0)
    # This preserves gradient flow better than torch.clip
    eps = 1.0e-16
    safe_not_return_probs = not_return_at_prev_probs + eps
    
    log_not_return_at_prev_probs = torch.log(safe_not_return_probs)
    cumsum_log_not_return_at_prev_probs = torch.cumsum(log_not_return_at_prev_probs, dim=0)
    
    start_indices_repeated = ray_starts.repeat_interleave(ray_lengths)
    
    log_not_return_at_prev_probs_by_ray = (
        cumsum_log_not_return_at_prev_probs
        - cumsum_log_not_return_at_prev_probs[start_indices_repeated]
    )
    
    not_return_at_prev_probs_by_ray = torch.exp(log_not_return_at_prev_probs_by_ray)
    
    # Remove the max=1.0 clipping that affects gradients
    # Instead, use a softer approach if needed
    # not_return_at_prev_probs_by_ray = torch.clamp(not_return_at_prev_probs_by_ray, max=1.0)
    
    # Calculate out-of-grid return probabilities
    ray_ends_actual = ray_starts + ray_lengths - 1
    out_of_grid_return_probs = (
        not_return_at_prev_probs_by_ray[ray_ends_actual] * not_return_at_curr_probs[ray_ends_actual]
    )
    
    # Calculate contributions
    tmp = ray_occupancy_probs * not_return_at_prev_probs_by_ray * ray_distances
    
    # Cumsum and calculate expected returns
    expected_returns_cumsum = torch.cumsum(tmp, dim=0)
    expected_returns_cumsum = torch.cat(
        [torch.zeros(1, device=ray_distances.device), expected_returns_cumsum], dim=0
    )
    
    ray_ends = ray_starts + ray_lengths
    expected_returns = (
        expected_returns_cumsum[ray_ends] - expected_returns_cumsum[ray_starts]
    )
    
    return expected_returns + out_of_grid_return_probs * out_of_grid_dists


def voxel_distance_to_point(
    voxel_centers: torch.Tensor,
    point: torch.Tensor,
) -> torch.Tensor:
    """Calculate the distance from each voxel center to a given point.

    Args:
        voxel_centers (torch.Tensor): Tensor of voxel centers, shape [D, H, W, N].
        point (torch.Tensor): Point to calculate distances to, shape [N].

    Returns:
        torch.Tensor: Distances from each voxel center to the point, shape [D, H, W].
    """
    assert voxel_centers.ndim == voxel_centers.shape[-1] + 1, (
        "dimensions of voxel_centers must be [D, H, W, N] and point must be [N]"
    )

    if point.ndim == 2:
        point = point.squeeze(0)  # shape: [N]

    ndim = voxel_centers.dim() - 1  # Number of dimensions of the point
    # Expand point to match voxel centers shape
    point_expanded = rearrange(point, "N -> " + "1 " * ndim + "N")

    # Calculate distances
    distances = torch.norm(
        voxel_centers - point_expanded, dim=-1, keepdim=False
    )

    return distances


def gather_ray_data_optimized(
    intersect_ind_list: list[np.ndarray],
    vox_dist_to_sensor: torch.Tensor,
    occupancy: torch.Tensor,
    device: torch.device,
    return_flat: bool = False,
) -> (
    tuple[list[torch.Tensor], list[torch.Tensor], None, None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Optimized gathering of ray data using vectorized operations.

    This function minimizes the number of indexing operations and leverages
    PyTorch's optimized tensor operations.
    """
    if not intersect_ind_list:
        raise ValueError("intersect_ind_list is empty")

    # find out the length of each ray
    ray_lengths = torch.tensor(
        [len(indices) for indices in intersect_ind_list],
        device=device,
        dtype=torch.long,
    )

    # Concatenate all indices into a single array
    all_indices = np.concatenate(intersect_ind_list, axis=0)
    all_indices_tensor = torch.tensor(
        all_indices, device=device, dtype=torch.long
    )

    # Single transpose operation for all indices
    transposed_indices = all_indices_tensor.T  # Shape: [3, total_voxels]

    # Batch gather all data in just two operations
    all_distances = vox_dist_to_sensor[
        tuple(transposed_indices)
    ]  # [total_voxels]
    all_occupancies = occupancy[tuple(transposed_indices)]  # [total_voxels]

    if return_flat:  # return a flat list of distances and occupancies
        # also get the start index of each ray
        ray_starts = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(ray_lengths[:-1], dim=0),
            ]
        )
        return all_distances, all_occupancies, ray_starts, ray_lengths

    # Split the concatenated results back into per-ray tensors
    ray_distances_list = []
    ray_occupancy_list = []
    start_idx = 0

    for indices in intersect_ind_list:
        length = len(indices)
        if length > 0:
            end_idx = start_idx + length
            ray_distances_list.append(all_distances[start_idx:end_idx])
            ray_occupancy_list.append(all_occupancies[start_idx:end_idx])
            start_idx = end_idx
        else:
            # Empty ray
            ray_distances_list.append(
                torch.empty(0, device=device, dtype=all_distances.dtype)
            )
            ray_occupancy_list.append(
                torch.empty(0, device=device, dtype=all_occupancies.dtype)
            )

    return ray_distances_list, ray_occupancy_list, None, None


def pad_ray_data_optimized(
    ray_distances_list: list[torch.Tensor],
    ray_occupancy_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized version that uses vectorized operations when possible.

    This version attempts to minimize Python loops and memory allocations.
    """
    if not ray_distances_list:
        device = torch.device("cpu")
        return (
            torch.empty(0, 0, device=device),
            torch.empty(0, 0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    device = ray_distances_list[0].device
    dtype = ray_distances_list[0].dtype
    num_rays = len(ray_distances_list)

    # Create ray_steps tensor directly from lengths
    ray_lengths = [len(ray) for ray in ray_distances_list]
    ray_steps = torch.tensor(ray_lengths, dtype=torch.long, device=device)
    max_length = int(torch.max(ray_steps).item()) if num_rays > 0 else 0

    if max_length == 0:
        return (
            torch.empty(num_rays, 0, device=device, dtype=dtype),
            torch.empty(num_rays, 0, device=device, dtype=dtype),
            ray_steps,
        )

    # Check if we can use the most efficient concatenation method
    if all(len(ray) > 0 for ray in ray_distances_list):
        # All rays are non-empty, use pad_sequence
        padded_distances = pad_sequence(
            ray_distances_list, batch_first=True, padding_value=0.0
        )
        padded_occupancy = pad_sequence(
            ray_occupancy_list, batch_first=True, padding_value=1.0
        )
    else:
        # Some rays might be empty, use manual approach but optimize it
        # Pre-allocate tensors
        padded_distances = torch.zeros(
            num_rays, max_length, device=device, dtype=dtype
        )
        padded_occupancy = torch.ones(
            num_rays, max_length, device=device, dtype=dtype
        )

        # Use advanced indexing to avoid Python loop where possible
        for i, (dists, occs) in enumerate(
            zip(ray_distances_list, ray_occupancy_list)
        ):
            length = len(dists)
            if length > 0:
                padded_distances[i, :length] = dists
                padded_occupancy[i, :length] = occs

    return padded_distances, padded_occupancy, ray_steps


def pad_ray_data_flat(
    ray_distances_list: list[torch.Tensor],
    ray_occupancy_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alternative approach using flat concatenation instead of padding.

    Returns flattened data with index mapping for batch processing.
    This can be more memory efficient when rays have very different lengths.

    Returns:
        - flat_distances: Concatenated distances [total_elements]
        - flat_occupancy: Concatenated occupancy [total_elements]
        - ray_starts: Start index for each ray [num_rays]
        - ray_steps: Length of each ray [num_rays]
    """
    if not ray_distances_list:
        device = torch.device("cpu")
        return (
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    device = ray_distances_list[0].device

    # Concatenate all rays
    flat_distances = torch.cat(ray_distances_list, dim=0)
    flat_occupancy = torch.cat(ray_occupancy_list, dim=0)

    # Create index mapping
    ray_lengths = [len(ray) for ray in ray_distances_list]
    ray_steps = torch.tensor(ray_lengths, dtype=torch.long, device=device)
    ray_starts = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(ray_steps[:-1], dim=0)]
    )

    return flat_distances, flat_occupancy, ray_starts, ray_steps


def pad_ray_data(
    ray_distances_list: list[torch.Tensor],
    ray_occupancy_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert list of variable-length rays to padded tensors for batch processing.

    Args:
        ray_distances_list: List of distance tensors, one per ray.
        ray_occupancy_list: List of occupancy tensors, one per ray.

    Returns:
        tuple containing:
        - padded_distances: Padded distance tensor [num_rays, max_length]
        - padded_occupancy: Padded occupancy tensor [num_rays, max_length]
        - ray_steps: Actual length of each ray [num_rays]
    """
    # Use the optimized version
    return pad_ray_data_optimized(ray_distances_list, ray_occupancy_list)


def get_out_of_grid_dist_batch(
    sensor_origin: np.ndarray,
    ray_directions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
) -> np.ndarray:
    """Calculate the distance from the sensor origin to the nearest point outside the
    grid for multiple rays.
    Args:
        sensor_origin (np.ndarray): Origin of the sensor rays, shape [3] or [1, 3].
        ray_directions (np.ndarray): Direction of the rays, shape [N, 3].
        grid_min (np.ndarray): Minimum boundaries of the voxel grid, shape [3].
        grid_max (np.ndarray): Maximum boundaries of the voxel grid, shape [3].
    Returns:
        np.ndarray: Distance to the nearest voxel outside the grid, shape [N].
    """
    # Ensure sensor_origin is broadcastable with ray_directions
    if sensor_origin.ndim == 1:
        sensor_origin = np.expand_dims(sensor_origin, axis=0)  # [1, 3]

    # Calculate distances to each boundary along the ray direction
    # t_min/t_max shape: [N, 3]
    t_min = (grid_min - sensor_origin) / (
        ray_directions + 1e-8
    )  # Avoid division by zero
    t_max = (grid_max - sensor_origin) / (ray_directions + 1e-8)

    # Concatenate along last dimension: [N, 6]
    tmp = np.concatenate([t_min, t_max], axis=-1)

    # Find minimum positive t for each ray
    # Create mask for positive values
    positive_mask = tmp > 0

    # Set negative values to infinity so they don't affect the minimum
    tmp_masked = np.where(positive_mask, tmp, np.inf)

    # Find minimum t for each ray
    t_exit = np.min(tmp_masked, axis=-1)  # [N]

    # Calculate exit distances
    out_of_grid_dist = np.linalg.norm(
        ray_directions * np.expand_dims(t_exit, axis=-1), axis=-1
    )

    return out_of_grid_dist


def get_voxel_centers_torch_tensor(
    grid_min: np.ndarray, grid_max: np.ndarray, voxel_size: np.ndarray
) -> torch.Tensor:
    """Calculate voxel centers as a torch tensor.

    Args:
        grid_min (np.ndarray): Minimum boundaries of the voxel grid, shape [3].
        grid_max (np.ndarray): Maximum boundaries of the voxel grid, shape [3].
        voxel_size (np.ndarray): Size of each voxel, shape [3].

    Returns:
        torch.Tensor: Tensor of voxel centers, shape [D, H, W, 3].
    """
    # Calculate number of voxels in each dimension
    num_voxels = np.floor((grid_max - grid_min) / voxel_size).astype(int)

    # Create a meshgrid for voxel indices
    z_indices = np.arange(num_voxels[2])
    y_indices = np.arange(num_voxels[1])
    x_indices = np.arange(num_voxels[0])

    x_grid, y_grid, z_grid = np.meshgrid(
        x_indices, y_indices, z_indices, indexing="ij"
    )

    # Calculate voxel centers
    centers_x = grid_min[0] + (x_grid + 0.5) * voxel_size[0]
    centers_y = grid_min[1] + (y_grid + 0.5) * voxel_size[1]
    centers_z = grid_min[2] + (z_grid + 0.5) * voxel_size[2]

    # Stack to create a tensor of shape [D, H, W, 3]
    voxel_centers = np.stack([centers_x, centers_y, centers_z], axis=-1)

    return torch.tensor(voxel_centers, dtype=torch.float32)


def diff_ray_casting_3d(
    sensor_origin: np.ndarray,
    ray_directions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    occupancy: torch.Tensor,
    vox_dist_to_sensor: torch.Tensor,
    out_of_grid_dists: torch.Tensor,
    pad_mode: str = "same",
) -> torch.Tensor:
    """Optimized differential ray casting in 3D using batch processing.

    This version processes multiple rays more efficiently by:
    1. Batching voxel center calculations
    2. Vectorizing distance computations
    3. Batch occupancy lookups
    4. Optional Numba acceleration for expected return calculation

    Args:
        sensor_origin (np.ndarray): Origin of the sensor rays, shape [3].
        ray_directions (np.ndarray): Direction of the rays, shape [N, 3].
        grid_min (np.ndarray): Minimum boundaries of the voxel grid, shape [3].
        grid_max (np.ndarray): Maximum boundaries of the voxel grid, shape [3].
        voxel_size (np.ndarray): Size of each voxel, shape [3].
        occupancy (torch.Tensor): Occupancy probabilities for each voxel,
                                  shape [D, H, W].
        vox_dist_to_sensor (torch.Tensor): Distances from sensor to voxel centers,
                                           shape [D, H, W].
        out_of_grid_dists (torch.Tensor): Distances to the nearest point outside the
                                          grid for each ray, shape [N].
        pad_mode (str): Mode for padding the ray data, "same", or "flat".
    Returns:
        torch.Tensor: Expected returns for each ray, shape [N].
    """
    assert occupancy.shape == vox_dist_to_sensor.shape[:3], (
        "occupancy must match the first three dimensions of vox_dist_to_sensor"
    )
    assert sensor_origin.shape == (3,), (
        "sensor_origin must be a 1D tensor with shape [3]"
    )
    assert out_of_grid_dists.shape[0] == ray_directions.shape[0], (
        "out_of_grid_dists must match the number of rays in ray_directions"
    )
    assert voxel_size.shape == (3,), (
        "voxel_size must be a 1D tensor with shape [3]"
    )
    assert pad_mode in ["same", "flat"], (
        "pad_mode must be either 'same', or 'flat'"
    )
    # Get voxel intersections for all rays
    _intersect_ind_list: list[np.ndarray] = fast_voxel_traversal_numba(
        sensor_origin,
        ray_directions,
        grid_min,
        grid_max,
        voxel_size,
    )

    # send to torch device of occupancy
    device = occupancy.device

    if pad_mode == "same":
        # Use optimized data gathering
        ray_distances_list, ray_occupancy_list, _, __ = (
            gather_ray_data_optimized(
                _intersect_ind_list,
                vox_dist_to_sensor,
                occupancy,
                device,
                return_flat=False,
            )
        )
        # get the padded tensors for batch processing
        padded_distances, padded_occupancy, ray_steps = pad_ray_data(
            ray_distances_list,  # type: ignore
            ray_occupancy_list,  # type: ignore
        )
        # Calculate expected returns for all rays in batch
        expected_returns = cal_expected_return_batch_padded(
            padded_distances,
            padded_occupancy,
            out_of_grid_dists,
            ray_steps,
        )
    elif pad_mode == "flat":
        ray_distances_flat, ray_occupancy_flat, ray_starts, ray_lengths = (
            gather_ray_data_optimized(
                _intersect_ind_list,
                vox_dist_to_sensor,
                occupancy,
                device,
                return_flat=True,
            )
        )
        expected_returns = cal_expected_return_batch_flat(
            ray_distances_flat,  # type: ignore
            ray_occupancy_flat,  # type: ignore
            ray_starts,  # type: ignore
            ray_lengths,  # type: ignore
            out_of_grid_dists,
        )
    else:
        raise ValueError(f"Invalid pad_mode: {pad_mode}")

    return expected_returns