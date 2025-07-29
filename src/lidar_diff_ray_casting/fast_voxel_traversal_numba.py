"""Fast voxel traversal algorithm for 2D and 3D grids using Numba-accelerated numpy code.

This module provides fast voxel traversal for 2D and 3D grids using the Amanatides-Woo algorithm,
accelerated with Numba for CPU workloads. All inputs and outputs are numpy arrays.

Limitations:
- Only works on CPU (Numba does not support GPU).
- All inputs must be numpy arrays.
"""

import numpy as np
from typing import List
from numba import njit

# =======================
# NUMBA-ACCELERATED CORE
# =======================

DEFAULT_NP_INT_DTYPE = np.int32

@njit
def fast_voxel_traversal_2d_numba_core(
    start_point: np.ndarray,
    ray_direction: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    num_voxels: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    """
    Numba-accelerated fast voxel traversal for a single 2D ray.
    Returns an array of traversed voxel indices (shape: [num_steps, 2]).

    Args:
        start_point: Starting position in the grid [2]
        ray_direction: Ray direction vector [2]
        grid_min: Minimum grid boundaries [2]
        grid_max: Maximum grid boundaries [2]
        voxel_size: Size of each voxel [2]
        num_voxels: Number of voxels in each dimension [2]
        max_steps: Maximum number of steps to take
    """
    grid_start = (start_point - grid_min) / voxel_size
    current_voxel = np.floor(grid_start).astype(DEFAULT_NP_INT_DTYPE)
    step = np.sign(ray_direction).astype(DEFAULT_NP_INT_DTYPE)
    delta_dist = np.abs(1.0 / (ray_direction + 1e-10))
    side_dist = np.zeros(2, dtype=np.float64)
    for i in range(2):
        if ray_direction[i] < 0:
            side_dist[i] = (
                float(grid_start[i]) - float(current_voxel[i])
            ) * float(delta_dist[i])
        else:
            side_dist[i] = (
                float(current_voxel[i]) + 1.0 - float(grid_start[i])
            ) * float(delta_dist[i])
    traversed = np.empty((max_steps, 2), dtype=DEFAULT_NP_INT_DTYPE)
    steps = 0
    while (
        0 <= current_voxel[0] < num_voxels[0]
        and 0 <= current_voxel[1] < num_voxels[1]
        and steps < max_steps
    ):
        traversed[steps, :] = current_voxel
        if side_dist[0] < side_dist[1]:
            side_dist[0] += delta_dist[0]
            current_voxel[0] += step[0]
        else:
            side_dist[1] += delta_dist[1]
            current_voxel[1] += step[1]
        steps += 1
    return traversed[:steps]


@njit
def fast_voxel_traversal_3d_numba_core(
    start_point: np.ndarray,
    ray_direction: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    num_voxel: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    """
    Numba-accelerated fast voxel traversal for a single 3D ray.
    Returns an array of traversed voxel indices (shape: [num_steps, 3]).

    Args:
        start_point: Starting position in the grid [3]
        ray_direction: Ray direction vector [3]
        grid_min: Minimum grid boundaries [3]
        grid_max: Maximum grid boundaries [3]
        voxel_size: Size of each voxel [3]
        num_voxel: Number of voxels in each dimension [3]
        max_steps: Maximum number of steps to take
    """
    grid_start = (start_point - grid_min) / voxel_size
    current_voxel = np.floor(grid_start).astype(DEFAULT_NP_INT_DTYPE)
    step = np.sign(ray_direction).astype(DEFAULT_NP_INT_DTYPE)
    delta_dist = np.abs(1.0 / (ray_direction + 1e-10))
    side_dist = np.zeros(3, dtype=np.float32)  # Use float32 to match PyTorch
    for i in range(3):
        if ray_direction[i] < 0:
            side_dist[i] = (grid_start[i] - current_voxel[i]) * delta_dist[i]
        else:
            side_dist[i] = (
                current_voxel[i] + 1.0 - grid_start[i]
            ) * delta_dist[i]
    traversed = np.empty((max_steps, 3), dtype=DEFAULT_NP_INT_DTYPE)
    steps = 0
    while (
        0 <= current_voxel[0] < num_voxel[0]
        and 0 <= current_voxel[1] < num_voxel[1]
        and 0 <= current_voxel[2] < num_voxel[2]
        and steps < max_steps
    ):
        traversed[steps, :] = current_voxel
        # Use the same tie-breaking logic as PyTorch by finding the minimum manually
        # to ensure consistent behavior across platforms
        min_val = side_dist[0]
        min_axis = 0
        for i in range(1, 3):
            if side_dist[i] < min_val:
                min_val = side_dist[i]
                min_axis = i
        side_dist[min_axis] += delta_dist[min_axis]
        current_voxel[min_axis] += step[min_axis]
        steps += 1
    return traversed[:steps]


# =======================
# POLAR GRID TRAVERSAL (NUMBA)
# =======================

@njit
def fast_voxel_traversal_polar_numba_core(
    origin_z: float,
    direction: np.ndarray,  # [3]
    r_max: float,
    z_min: float,
    z_max: float,
    num_azimuth: int,
    num_radius: int,
    num_z: int,
    max_steps: int | None = None,
) -> np.ndarray:
    """
    Improved Numba-accelerated fast voxel traversal for a single ray in a polar grid.
    Assumes the origin is at (0, 0, origin_z).
    Returns an array of traversed voxel indices (shape: [num_steps, 3]) as (azimuth, radius, z).
    
    Improvements:
    - Added numerical stability with epsilon values
    - Better edge case handling for vertical rays
    - More robust boundary condition checks
    - Clearer variable naming and structure

    Args:
        origin_z: Z coordinate of the origin
        direction: Ray direction vector [3]
        r_max: Maximum radius
        z_min: Minimum z coordinate
        z_max: Maximum z coordinate
        num_azimuth: Number of azimuth bins
        num_radius: Number of radius bins
        num_z: Number of z bins
        max_steps: The maximum number of steps to take. If None, it will be set to 
            num_radius + num_z.
    """
    # Constants for numerical stability
    EPS = 1e-10
    
    if max_steps is None:
        max_steps = num_radius + num_z

    # Extract ray direction components
    dx, dy, dz = direction[0], direction[1], direction[2]
    
    # Calculate radial component of direction (projection onto XY plane)
    dr = np.sqrt(dx * dx + dy * dy)
    
    # Handle purely vertical rays (no radial component)
    if dr < EPS:
        # Ray is vertical - only traverse in Z direction at r=0
        if abs(dz) < EPS:
            # Ray has no direction - return empty
            return np.empty((0, 3), dtype=DEFAULT_NP_INT_DTYPE)
        
        # Simple Z-only traversal
        z_bin_size = (z_max - z_min) / num_z
        z_idx = int(np.floor((origin_z - z_min) / z_bin_size))
        
        # Check if starting z position is within grid bounds
        if z_idx < 0 or z_idx >= num_z:
            return np.empty((0, 3), dtype=DEFAULT_NP_INT_DTYPE)
        
        step_z = 1 if dz > 0 else -1
        traversed = np.empty((max_steps, 3), dtype=DEFAULT_NP_INT_DTYPE)
        steps = 0
        
        while 0 <= z_idx < num_z and steps < max_steps:
            traversed[steps, 0] = 0  # azimuth bin 0 (arbitrary for r=0)
            traversed[steps, 1] = 0  # radius bin 0
            traversed[steps, 2] = z_idx
            z_idx += step_z
            steps += 1
            
        return traversed[:steps]
    
    # Calculate azimuth for non-vertical rays
    azimuth = np.arctan2(dy, dx)
    if azimuth < 0:
        azimuth += 2 * np.pi
    
    # Grid parameters
    r_bin_size = r_max / num_radius
    z_bin_size = (z_max - z_min) / num_z
    azimuth_bin_size = 2 * np.pi / num_azimuth
    
    # Calculate azimuth bin index with proper wrapping
    azimuth_idx = int(azimuth / azimuth_bin_size)
    azimuth_idx = min(azimuth_idx, num_azimuth - 1)  # Clamp to valid range
    
    # Initial grid indices
    r_idx = 0  # Always start at r=0 for origin-based rays
    z_idx = int(np.floor((origin_z - z_min) / z_bin_size))
    
    # Check if starting z position is within grid bounds
    if z_idx < 0 or z_idx >= num_z:
        return np.empty((0, 3), dtype=DEFAULT_NP_INT_DTYPE)
    
    # Step directions
    step_r = 1  # Radius always increases from origin
    step_z = 1 if dz > EPS else -1 if dz < -EPS else 0
    
    # Calculate parametric distances for radius traversal
    # For r = t * dr, next boundary at r = (r_idx + 1) * r_bin_size
    # So t = r_boundary / dr
    r_boundary = (r_idx + 1) * r_bin_size
    t_max_r = r_boundary / dr
    t_delta_r = r_bin_size / dr
    
    # Calculate parametric distances for z traversal  
    # For z = origin_z + t * dz
    if abs(dz) > EPS:
        if dz > 0:
            z_boundary = z_min + (z_idx + 1) * z_bin_size
            t_max_z = (z_boundary - origin_z) / dz
        else:
            z_boundary = z_min + z_idx * z_bin_size
            t_max_z = (z_boundary - origin_z) / dz
        t_delta_z = abs(z_bin_size / dz)
    else:
        t_max_z = np.inf
        t_delta_z = np.inf
    
    # Traversal array
    traversed = np.empty((max_steps, 3), dtype=DEFAULT_NP_INT_DTYPE)
    steps = 0
    
    # Main traversal loop using 2D Amanatides-Woo algorithm (r, z)
    while (0 <= r_idx < num_radius and 
           0 <= z_idx < num_z and 
           steps < max_steps):
        
        # Record current voxel
        traversed[steps, 0] = azimuth_idx
        traversed[steps, 1] = r_idx
        traversed[steps, 2] = z_idx
        steps += 1
        
        # Determine which boundary to cross next
        if t_max_r < t_max_z:
            # Cross radius boundary first
            r_idx += step_r
            t_max_r += t_delta_r
        else:
            # Cross z boundary first (or tie - choose z for consistency)
            z_idx += step_z
            t_max_z += t_delta_z
    
    return traversed[:steps]


# =======================
# WRAPPERS FOR BATCH INPUT
# =======================


def fast_voxel_traversal_2d_numba(
    start_point: np.ndarray,
    ray_directions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    num_voxels: np.ndarray | None = None,
    max_steps: int | None = None,
) -> List[np.ndarray]:
    """
    Fast voxel traversal for 2D rays using Numba-accelerated numpy code.

    Args:
        start_point: Starting position [2] or [1, 2]
        ray_directions: Ray directions [N, 2]
        grid_min: Minimum grid boundaries [2]
        grid_max: Maximum grid boundaries [2]
        voxel_size: Size of each voxel [2]
        num_voxels: Number of voxels in each dimension [2]
        max_steps: Maximum steps per ray

    Returns:
        List of numpy arrays (each [num_steps, 2]) for each ray.
    """
    if num_voxels is None:
        num_voxels = np.floor((grid_max - grid_min) / voxel_size).astype(
            DEFAULT_NP_INT_DTYPE
        )

    # Handle input shapes
    if ray_directions.ndim == 1:
        ray_directions = np.expand_dims(ray_directions, 0)

    num_rays = ray_directions.shape[0]

    # Handle start_point - extract single point if 2D array
    if start_point.ndim == 2:
        start_point = start_point[0]  # Extract [2] from [1, 2]

    results = []
    for i in range(num_rays):
        ray_dir = ray_directions[i]
        max_ray_steps = (
            max_steps
            if max_steps is not None
            else int(num_voxels[0] + num_voxels[1])
        )
        voxels = fast_voxel_traversal_2d_numba_core(
            start_point,
            ray_dir,
            grid_min,
            grid_max,
            voxel_size,
            num_voxels,
            max_ray_steps,
        )
        results.append(voxels)
    return results


def fast_voxel_traversal_3d_numba(
    start_point: np.ndarray,
    ray_directions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    num_voxels: np.ndarray | None = None,
    max_steps: int | None = None,
) -> List[np.ndarray]:
    """
    Fast voxel traversal for 3D rays using Numba-accelerated numpy code.

    Args:
        start_point: Starting position [3] or [1, 3]
        ray_directions: Ray directions [N, 3]
        grid_min: Minimum grid boundaries [3]
        grid_max: Maximum grid boundaries [3]
        voxel_size: Size of each voxel [3]
        num_voxels: Number of voxels in each dimension [3]
        max_steps: Maximum steps per ray
        
    Returns:
        List of numpy arrays (each [num_steps, 3]) for each ray.
    """
    if num_voxels is None:
        # Calculate number of voxels if not provided
        num_voxels = np.floor((grid_max - grid_min) / voxel_size).astype(
            DEFAULT_NP_INT_DTYPE
        )

    # Handle input shapes
    if ray_directions.ndim == 1:
        ray_directions = np.expand_dims(ray_directions, 0)

    num_rays = ray_directions.shape[0]

    # Handle start_point - extract single point if 2D array
    if start_point.ndim == 2:
        start_point = start_point[0]  # Extract [3] from [1, 3]

    results = []
    for i in range(num_rays):
        ray_dir = ray_directions[i]
        max_ray_steps = (
            max_steps
            if max_steps is not None
            else int(num_voxels[0] + num_voxels[1] + num_voxels[2])
        )
        voxels = fast_voxel_traversal_3d_numba_core(
            start_point,
            ray_dir,
            grid_min,
            grid_max,
            voxel_size,
            num_voxels,
            max_ray_steps,
        )
        results.append(voxels)
    return results


def fast_voxel_traversal_polar_numba(
    origin_z: float,
    ray_directions: np.ndarray,  # [N, 3]
    r_max: float,
    z_min: float,
    z_max: float,
    num_azimuth: int,
    num_radius: int,
    num_z: int,
    max_steps: int | None = None,
) -> list:
    """
    Batch wrapper for fast voxel traversal in polar grid.

    Args:
        origin_z: Z coordinate of the origin
        ray_directions: Ray direction vector [N, 3]
        r_max: Maximum radius
        z_min: Minimum z coordinate
        z_max: Maximum z coordinate
        num_azimuth: Number of azimuth bins
        num_radius: Number of radius bins
        num_z: Number of z bins
        max_steps: The maximum number of steps to take. If None, it will be set to 
            num_radius + num_z.
    Returns:
        List of numpy arrays (each [num_steps, 3]) for each ray. The indices are (azimuth, radius, z).
    """
    if ray_directions.ndim == 1:
        ray_directions = np.expand_dims(ray_directions, 0)
    num_rays = ray_directions.shape[0]
    if max_steps is None:
        max_steps = num_radius + num_z
    results = []
    for i in range(num_rays):
        voxels = fast_voxel_traversal_polar_numba_core(
            origin_z,
            ray_directions[i],
            r_max,
            z_min,
            z_max,
            num_azimuth,
            num_radius,
            num_z,
            max_steps,
        )
        results.append(voxels)
    return results


def fast_voxel_traversal_numba(
    start_point: np.ndarray,
    ray_directions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    max_steps: int | None = None,
) -> List[np.ndarray]:
    """
    Unified interface for fast voxel traversal (2D or 3D) using Numba-accelerated numpy code.

    Args:
        start_point: Starting position [D] or [1, D] where D=2 or 3
        ray_directions: Ray directions [N, D] where D=2 or 3
        grid_min: Grid minimum boundaries [D]
        grid_max: Grid maximum boundaries [D]
        voxel_size: Voxel size [D]
        max_steps: Maximum steps per ray

    Returns:
        List of numpy arrays (each [num_steps, D]) for each ray.
    """
    if ray_directions.shape[-1] == 2:
        return fast_voxel_traversal_2d_numba(
            start_point, ray_directions, grid_min, grid_max, voxel_size, max_steps=max_steps
        )
    elif ray_directions.shape[-1] == 3:
        return fast_voxel_traversal_3d_numba(
            start_point, ray_directions, grid_min, grid_max, voxel_size, max_steps=max_steps
        )
    else:
        raise ValueError(
            f"Unsupported dimension: {ray_directions.shape[-1]}. Only 2D and 3D are supported."
        )
