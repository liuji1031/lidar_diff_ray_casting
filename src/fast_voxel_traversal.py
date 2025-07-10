"""Fast voxel traversal algorithm for 2D and 3D grids using PyTorch tensors."""
import torch
from typing import Tuple, List, Union, Optional


def fast_voxel_traversal_2d(
    start_point: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor],
    max_steps: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Fast voxel traversal algorithm for 2D grids using the Amanatides-Woo algorithm.
    
    Args:
        start_point: Starting position of rays. Shape: [2] or [N, 2]
        ray_directions: Ray directions. Shape: [N, 2]
        grid_boundaries: Grid boundaries [[min_x, min_y], [max_x, max_y]]. Shape: [2, 2]
        cell_size: Size of each cell. Scalar or tensor of shape [2]
        max_steps: Maximum number of steps per ray (optional)
    
    Returns:
        List of tensors, each containing voxel indices for one ray. Shape: [num_steps, 2]
    """
    device = ray_directions.device
    dtype = ray_directions.dtype
    
    # Ensure inputs are on the same device and dtype
    if start_point.dim() == 1:
        start_point = start_point.unsqueeze(0).expand(ray_directions.shape[0], -1)
    
    grid_boundaries = grid_boundaries.to(device=device, dtype=dtype)
    
    if isinstance(cell_size, (int, float)):
        cell_size = torch.tensor([cell_size, cell_size], device=device, dtype=dtype)
    else:
        cell_size = cell_size.to(device=device, dtype=dtype)
        if cell_size.dim() == 0:
            cell_size = cell_size.expand(2)
    
    grid_min = grid_boundaries[0]  # [min_x, min_y]
    grid_max = grid_boundaries[1]  # [max_x, max_y]
    
    # Calculate grid dimensions
    grid_size = ((grid_max - grid_min) / cell_size).ceil().long()
    
    num_rays = ray_directions.shape[0]
    results = []
    
    for ray_idx in range(num_rays):
        ray_start = start_point[ray_idx]
        ray_dir = ray_directions[ray_idx]
        
        # Convert start point to grid coordinates
        grid_start = (ray_start - grid_min) / cell_size
        
        # Current voxel indices
        current_voxel = grid_start.floor().long()
        
        # Step direction for each axis
        step = torch.sign(ray_dir).long()
        
        # Distance to travel along ray to cross one voxel
        delta_dist = torch.abs(1.0 / (ray_dir + 1e-10))  # Add small epsilon to avoid division by zero
        
        # Distance from current position to next voxel boundary
        side_dist = torch.zeros(2, device=device, dtype=dtype)
        
        for i in range(2):
            if ray_dir[i] < 0:
                side_dist[i] = (grid_start[i] - current_voxel[i].float()) * delta_dist[i]
            else:
                side_dist[i] = (current_voxel[i].float() + 1.0 - grid_start[i]) * delta_dist[i]
        
        # Collect traversed voxels
        traversed_voxels = []
        steps = 0
        max_ray_steps = max_steps if max_steps is not None else (grid_size[0] + grid_size[1]).item()
        
        while (0 <= current_voxel[0] < grid_size[0] and 
               0 <= current_voxel[1] < grid_size[1] and 
               steps < max_ray_steps):
            
            traversed_voxels.append(current_voxel.clone())
            
            # Determine which axis to step along
            if side_dist[0] < side_dist[1]:
                side_dist[0] += delta_dist[0]
                current_voxel[0] += step[0]
            else:
                side_dist[1] += delta_dist[1]
                current_voxel[1] += step[1]
            
            steps += 1
        
        if traversed_voxels:
            results.append(torch.stack(traversed_voxels))
        else:
            results.append(torch.empty((0, 2), dtype=torch.long, device=device))
    
    return results


def fast_voxel_traversal_3d(
    start_point: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor],
    max_steps: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Fast voxel traversal algorithm for 3D grids using the Amanatides-Woo algorithm.
    
    Args:
        start_point: Starting position of rays. Shape: [3] or [N, 3]
        ray_directions: Ray directions. Shape: [N, 3]
        grid_boundaries: Grid boundaries [[min_x, min_y, min_z], [max_x, max_y, max_z]]. Shape: [2, 3]
        cell_size: Size of each cell. Scalar or tensor of shape [3]
        max_steps: Maximum number of steps per ray (optional)
    
    Returns:
        List of tensors, each containing voxel indices for one ray. Shape: [num_steps, 3]
    """
    device = ray_directions.device
    dtype = ray_directions.dtype
    
    # Ensure inputs are on the same device and dtype
    if start_point.dim() == 1:
        start_point = start_point.unsqueeze(0).expand(ray_directions.shape[0], -1)
    
    grid_boundaries = grid_boundaries.to(device=device, dtype=dtype)
    
    if isinstance(cell_size, (int, float)):
        cell_size = torch.tensor([cell_size, cell_size, cell_size], device=device, dtype=dtype)
    else:
        cell_size = cell_size.to(device=device, dtype=dtype)
        if cell_size.dim() == 0:
            cell_size = cell_size.expand(3)
    
    grid_min = grid_boundaries[0]  # [min_x, min_y, min_z]
    grid_max = grid_boundaries[1]  # [max_x, max_y, max_z]
    
    # Calculate grid dimensions
    grid_size = ((grid_max - grid_min) / cell_size).ceil().long()
    
    num_rays = ray_directions.shape[0]
    results = []
    
    for ray_idx in range(num_rays):
        ray_start = start_point[ray_idx]
        ray_dir = ray_directions[ray_idx]
        
        # Convert start point to grid coordinates
        grid_start = (ray_start - grid_min) / cell_size
        
        # Current voxel indices
        current_voxel = grid_start.floor().long()
        
        # Step direction for each axis
        step = torch.sign(ray_dir).long()
        
        # Distance to travel along ray to cross one voxel
        delta_dist = torch.abs(1.0 / (ray_dir + 1e-10))  # Add small epsilon to avoid division by zero
        
        # Distance from current position to next voxel boundary
        side_dist = torch.zeros(3, device=device, dtype=dtype)
        
        for i in range(3):
            if ray_dir[i] < 0:
                side_dist[i] = (grid_start[i] - current_voxel[i].float()) * delta_dist[i]
            else:
                side_dist[i] = (current_voxel[i].float() + 1.0 - grid_start[i]) * delta_dist[i]
        
        # Collect traversed voxels
        traversed_voxels = []
        steps = 0
        max_ray_steps = max_steps if max_steps is not None else (grid_size[0] + grid_size[1] + grid_size[2]).item()
        
        while (0 <= current_voxel[0] < grid_size[0] and 
               0 <= current_voxel[1] < grid_size[1] and 
               0 <= current_voxel[2] < grid_size[2] and 
               steps < max_ray_steps):
            
            traversed_voxels.append(current_voxel.clone())
            
            # Determine which axis to step along (choose the one with smallest side distance)
            min_axis = torch.argmin(side_dist)
            side_dist[min_axis] += delta_dist[min_axis]
            current_voxel[min_axis] += step[min_axis]
            
            steps += 1
        
        if traversed_voxels:
            results.append(torch.stack(traversed_voxels))
        else:
            results.append(torch.empty((0, 3), dtype=torch.long, device=device))
    
    return results


def fast_voxel_traversal(
    start_point: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor],
    max_steps: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Unified interface for fast voxel traversal in 2D or 3D.
    
    Args:
        start_point: Starting position of rays. Shape: [2/3] or [N, 2/3]
        ray_directions: Ray directions. Shape: [N, 2/3]
        grid_boundaries: Grid boundaries. Shape: [2, 2/3]
        cell_size: Size of each cell. Scalar or tensor (shape [2] or [3], for
        x, y, (z) directions)
        max_steps: Maximum number of steps per ray (optional)
    
    Returns:
        List of tensors, each containing voxel indices for one ray
    """
    if ray_directions.shape[-1] == 2:
        return fast_voxel_traversal_2d(start_point, ray_directions, grid_boundaries, cell_size, max_steps)
    elif ray_directions.shape[-1] == 3:
        return fast_voxel_traversal_3d(start_point, ray_directions, grid_boundaries, cell_size, max_steps)
    else:
        raise ValueError(f"Unsupported dimension: {ray_directions.shape[-1]}. Only 2D and 3D are supported.")


def get_voxel_centers_2d(
    voxel_indices: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Convert 2D voxel indices to world coordinates of voxel centers.
    
    Args:
        voxel_indices: Voxel indices. Shape: [N, 2]
        grid_boundaries: Grid boundaries [[min_x, min_y], [max_x, max_y]]. Shape: [2, 2]
        cell_size: Size of each cell. Scalar or tensor of shape [2]
    
    Returns:
        World coordinates of voxel centers. Shape: [N, 2]
    """
    device = voxel_indices.device
    
    grid_boundaries = grid_boundaries.to(device=device)
    
    if isinstance(cell_size, float):
        cell_size = torch.tensor([cell_size, cell_size], device=device)
    else:
        cell_size = cell_size.to(device=device)
        if cell_size.dim() == 0:
            cell_size = cell_size.expand(2)
    
    grid_min = grid_boundaries[0]
    # Convert indices to world coordinates (center of voxels)
    world_coords = grid_min + (voxel_indices.float() + 0.5) * cell_size
    
    return world_coords


def get_voxel_centers_3d(
    voxel_indices: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Convert 3D voxel indices to world coordinates of voxel centers.
    
    Args:
        voxel_indices: Voxel indices. Shape: [N, 3]
        grid_boundaries: Grid boundaries [[min_x, min_y, min_z], [max_x, max_y, max_z]]. Shape: [2, 3]
        cell_size: Size of each cell. Scalar or tensor of shape [3]
    
    Returns:
        World coordinates of voxel centers. Shape: [N, 3]
    """
    device = voxel_indices.device
    
    grid_boundaries = grid_boundaries.to(device=device)
    
    if isinstance(cell_size, (int, float)):
        cell_size = torch.tensor([cell_size, cell_size, cell_size], device=device)
    else:
        cell_size = cell_size.to(device=device)
        if cell_size.dim() == 0:
            cell_size = cell_size.expand(3)
    
    grid_min = grid_boundaries[0]
    
    # Convert indices to world coordinates (center of voxels)
    world_coords = grid_min + (voxel_indices.float() + 0.5) * cell_size
    
    return world_coords


def validate_inputs_2d(
    start_point: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor]
) -> None:
    """Validate inputs for 2D voxel traversal."""
    if start_point.shape[-1] != 2:
        raise ValueError(f"start_point must have 2 dimensions, got {start_point.shape[-1]}")
    
    if ray_directions.shape[-1] != 2:
        raise ValueError(f"ray_directions must have 2 dimensions, got {ray_directions.shape[-1]}")
    
    if grid_boundaries.shape != (2, 2):
        raise ValueError(f"grid_boundaries must have shape (2, 2), got {grid_boundaries.shape}")
    
    if isinstance(cell_size, torch.Tensor) and cell_size.numel() not in [1, 2]:
        raise ValueError(f"cell_size tensor must have 1 or 2 elements, got {cell_size.numel()}")


def validate_inputs_3d(
    start_point: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_boundaries: torch.Tensor,
    cell_size: Union[float, torch.Tensor]
) -> None:
    """Validate inputs for 3D voxel traversal."""
    if start_point.shape[-1] != 3:
        raise ValueError(f"start_point must have 3 dimensions, got {start_point.shape[-1]}")
    
    if ray_directions.shape[-1] != 3:
        raise ValueError(f"ray_directions must have 3 dimensions, got {ray_directions.shape[-1]}")
    
    if grid_boundaries.shape != (2, 3):
        raise ValueError(f"grid_boundaries must have shape (2, 3), got {grid_boundaries.shape}")
    
    if isinstance(cell_size, torch.Tensor) and cell_size.numel() not in [1, 3]:
        raise ValueError(f"cell_size tensor must have 1 or 3 elements, got {cell_size.numel()}")
