# lidar-diff-ray-casting

Differential ray casting to render point clouds from voxel grids, with fast voxel traversal and efficient batch processing. This package is designed for robotics, perception, and simulation applications where differentiable ray casting is needed for 3D occupancy grids.

## Features
- **Differentiable 3D ray casting**: Compute expected returns and gradients for rays traversing a voxel grid.
- **Fast voxel traversal**: Uses Numba-accelerated Amanatides-Woo algorithm for efficient CPU-based voxel intersection.
- **Batch processing**: Handles large numbers of rays in parallel, optimized for GPU with PyTorch.
- **Flexible padding modes**: Choose between memory-efficient flat mode or fully padded mode for batch operations.

## Installation

This package requires Python 3.12+ and the following dependencies:
- torch
- numpy
- einops
- numba
- scipy

Install with pip (from the project root):

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install torch numpy einops numba scipy
```

## Main API: `diff_ray_casting_3d`

The core function for differentiable ray casting is:

```python
diff_ray_casting_3d(
    sensor_origin: np.ndarray,
    ray_directions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    voxel_size: np.ndarray,
    occupancy: torch.Tensor,
    vox_dist_to_sensor: torch.Tensor,
    out_of_grid_dists: torch.Tensor,
    pad_mode: str = "same",
) -> torch.Tensor
```

### Arguments
- `sensor_origin`: (np.ndarray, shape [3]) The origin of all rays (e.g., sensor position).
- `ray_directions`: (np.ndarray, shape [N, 3]) Array of N ray direction vectors.
- `grid_min`, `grid_max`: (np.ndarray, shape [3]) Minimum and maximum coordinates of the voxel grid.
- `voxel_size`: (np.ndarray, shape [3]) Size of each voxel along each axis.
- `occupancy`: (torch.Tensor, shape [D, H, W]) Occupancy probability for each voxel (values in [0, 1]).
- `vox_dist_to_sensor`: (torch.Tensor, shape [D, H, W]) Precomputed distances from each voxel center to the sensor origin.
- `out_of_grid_dists`: (torch.Tensor, shape [N]) Distance from the sensor to the first point outside the grid for each ray.
- `pad_mode`: (str) Either 'same' (padded batch) or 'flat' (concatenated, memory-efficient).

### Returns
- `expected_returns`: (torch.Tensor, shape [N]) The expected return distance for each ray.

## Example Usage

Below is a minimal example based on the test suite. This demonstrates how to set up the grid, occupancy, rays, and call `diff_ray_casting_3d`:

```python
import numpy as np
import torch
from src import (
    diff_ray_casting_3d,
    get_voxel_centers_torch_tensor,
    voxel_distance_to_point,
    get_out_of_grid_dist_batch,
)

# Define grid and sensor
sensor_origin = np.array([0.0, 0.0, 0.9], dtype=np.float32)
grid_min = np.array([-50.0, -50.0, 0.0], dtype=np.float32)
grid_max = np.array([50.0, 50.0, 10.0], dtype=np.float32)
voxel_size = np.array([0.4, 0.4, 0.4], dtype=np.float32)

# Create occupancy grid (example: mostly free, ground plane occupied)
occupancy_shape = (
    int((grid_max[0] - grid_min[0]) / voxel_size[0]),
    int((grid_max[1] - grid_min[1]) / voxel_size[1]),
    int((grid_max[2] - grid_min[2]) / voxel_size[2]),
)
occupancy = torch.ones(occupancy_shape, dtype=torch.float32) * 1e-8
occupancy[:, :, 0] = 0.9  # ground plane

# Define rays (here: 128 azimuths, 3 elevations)
n_ray = 128
azimuth = np.arange(0, 2 * np.pi, 2 * np.pi / n_ray)
elevation = np.array([-np.pi / 20, -np.pi / 40, 0.0])
azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="ij")
ray_directions = np.stack([
    np.cos(azimuth_grid).reshape(-1) * np.cos(elevation_grid).reshape(-1),
    np.sin(azimuth_grid).reshape(-1) * np.cos(elevation_grid).reshape(-1),
    np.sin(elevation_grid).reshape(-1),
], axis=-1)

# Precompute voxel centers and distances
dist_to_sensor = voxel_distance_to_point(
    get_voxel_centers_torch_tensor(grid_min, grid_max, voxel_size),
    torch.tensor(sensor_origin, dtype=torch.float32),
)

# Compute out-of-grid distances
out_of_grid_dists = get_out_of_grid_dist_batch(
    sensor_origin, ray_directions, grid_min, grid_max
)
out_of_grid_dists = torch.tensor(out_of_grid_dists, dtype=torch.float32)

# Run differentiable ray casting
expected_return = diff_ray_casting_3d(
    sensor_origin=sensor_origin,
    ray_directions=ray_directions,
    grid_min=grid_min,
    grid_max=grid_max,
    voxel_size=voxel_size,
    occupancy=occupancy,
    vox_dist_to_sensor=dist_to_sensor,
    out_of_grid_dists=out_of_grid_dists,
    pad_mode="same",  # or "flat"
)
print(expected_return)
```

## Supporting Functions

- `get_voxel_centers_torch_tensor(grid_min, grid_max, voxel_size)`: Returns a tensor of voxel center coordinates.
- `voxel_distance_to_point(voxel_centers, point)`: Computes the distance from each voxel center to a given point.
- `get_out_of_grid_dist_batch(sensor_origin, ray_directions, grid_min, grid_max)`: Computes the distance from the sensor to the first point outside the grid for each ray.

## Fast Voxel Traversal (Numba)

The module `src/fast_voxel_traversal_numba.py` provides fast, Numba-accelerated routines for finding which voxels a ray traverses in 2D, 3D, or polar grids. It implements the Amanatides-Woo algorithm for efficient grid traversal, and is used internally by `diff_ray_casting_3d` for high performance.

- Only works on CPU (Numba does not support GPU).
- All inputs/outputs are numpy arrays.

### Supported Traversal Modes

- **2D Cartesian grid**: `fast_voxel_traversal_2d_numba`
- **3D Cartesian grid**: `fast_voxel_traversal_3d_numba`
- **Polar grid**: `fast_voxel_traversal_polar_numba`

#### Polar Grid Traversal

The function `fast_voxel_traversal_polar_numba` (and its core, `fast_voxel_traversal_polar_numba_core`) efficiently computes the sequence of (azimuth, radius, z) bins traversed by a ray starting from the z-axis (origin at (0, 0, origin_z)) in a cylindrical/polar grid. This is useful for simulating or analyzing sensors with polar or cylindrical geometry, such as rotating lidars.

- **Inputs:**
  - `origin_z`: Z coordinate of the origin
  - `ray_directions`: Ray direction(s), shape [N, 3]
  - `r_max`: Maximum radius
  - `z_min`, `z_max`: Z bounds
  - `num_azimuth`, `num_radius`, `num_z`: Number of bins in each dimension
  - `max_steps`: (Optional) Maximum number of steps per ray
- **Outputs:**
  - List of numpy arrays (each [num_steps, 3]) for each ray, with indices (azimuth, radius, z)

See the function docstrings for more details and usage examples.

## Running Tests

To run the test suite and generate example outputs:

```bash
python -m test.test_ray_casting
```

This will generate example images and print performance statistics.

## License

MIT License. See LICENSE file.
