from typing import List, Tuple
import numpy as np

from .container import Container

class SpatialGrid:
    """
    Spatial grid for efficient collision detection in a 3D container

    API
    build_grid(cell_size: float): called when the grid is created or when the cell size changes
    update(positions: np.ndarray): called when the positions of the particles change, to update the grid
    get_potential_collision_pairs(positions: np.ndarray) -> np.ndarray: called to detect potential collisions, returns an array of particle index pairs that are in the same or neighbouring cells

    """

    def __init__(self, container: Container, cell_size: float):
        self.container = container
        self.build_grid(cell_size)

    def build_grid(self, cell_size: float):
        container = self.container
        self._num_cells_x = int(np.ceil(container.lx / cell_size))
        self._num_cells_y = int(np.ceil(container.ly / cell_size))
        self._num_cells_z = int(np.ceil(container.lz / cell_size))
        self._total_cells = self._num_cells_x * self._num_cells_y * self._num_cells_z
        self._actual_cell_sizes =  container.dimensions / np.array([self._num_cells_x, self._num_cells_y, self._num_cells_z])
        self.cells: List[List[int]] = [[] for _ in range(self._total_cells)]

    @staticmethod
    def _neighbour_shifts() -> np.ndarray:
        shifts = [] # shifts to get all neighbouring cells (including the cell itself)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    shifts.append((dx, dy, dz))
        return np.array(shifts, dtype=np.int32)
    
    def _positions_to_cell_coords(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = (positions / self._actual_cell_sizes).astype(np.int32)
        coords[:, 0] = np.clip(coords[:, 0], 0, self._num_cells_x - 1) # clip in case some particles are outside the container
        coords[:, 1] = np.clip(coords[:, 1], 0, self._num_cells_y - 1)
        coords[:, 2] = np.clip(coords[:, 2], 0, self._num_cells_z - 1)
        return coords[:, 0], coords[:, 1], coords[:, 2]

    def _cell_coords_to_indices(self, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> np.ndarray:
        return cx + cy * self._num_cells_x + cz * self._num_cells_x * self._num_cells_y
    
    def _positions_to_cell_indices(self, positions: np.ndarray) -> np.ndarray:
        cx, cy, cz = self._positions_to_cell_coords(positions)
        return self._cell_coords_to_indices(cx, cy, cz)

    def update(self, positions: np.ndarray):
        for cell in self.cells:
            cell.clear()
        if len(positions) == 0:
            return
        cell_indices = self._positions_to_cell_indices(positions)
        for particle_idx, cell_idx in enumerate(cell_indices):
            self.cells[cell_idx].append(particle_idx)

    # def get_neighbouring_cells_indices(self, positions: np.ndarray) -> np.ndarray:
    #     # takes an array of positions and returns an array of indices of neighbouring cells for each position
    #     # includes the index of the cell the position is in
    #     # can contain duplicates because of clipping for particles near an edge
    #     cx, cy, cz = self._positions_to_cell_coords(positions)
    #     shifts = self._neighbour_shifts()
    #     neighbouring_cells = []
    #     for dx, dy, dz in shifts:
    #         ncx = np.clip(cx + dx, 0, self._num_cells_x - 1) # clip for particles near an edge
    #         ncy = np.clip(cy + dy, 0, self._num_cells_y - 1)
    #         ncz = np.clip(cz + dz, 0, self._num_cells_z - 1)
    #         neighbouring_cells.append(self._cell_coords_to_indices(ncx, ncy, ncz))
    #     return np.array(neighbouring_cells, dtype=np.int32).T  # shape (num_particles, num_neighbouring_cells)

    def _get_mask(self, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> np.ndarray:
        return (
            (0 <= cx) & (cx < self._num_cells_x) & # mask for valid cell coordinates
            (0 <= cy) & (cy < self._num_cells_y) &
            (0 <= cz) & (cz < self._num_cells_z)
        )
    
    def _get_neighbouring_cells_indices(self, positions: np.ndarray) -> List[List[int]]:
        cx, cy, cz = self._positions_to_cell_coords(positions)
        shifts = self._neighbour_shifts()
        neighbouring_cells: List[List[int]] = [[] for _ in range(positions.shape[0])]

        for dx, dy, dz in shifts:
            ncx = cx + dx
            ncy = cy + dy
            ncz = cz + dz
            # Note: this batch version improves only slightly the performance, because we compute +dx for all particles simultaneously
            # but because we want to avoid duplicates (and get lists of different lengths), we can't make a fully vectorized version

            valid_mask = self._get_mask(ncx, ncy, ncz)
            if not np.any(valid_mask):
                continue
            valid_particle_ids = np.where(valid_mask)[0]

            cell_indices = self._cell_coords_to_indices(ncx[valid_mask], ncy[valid_mask], ncz[valid_mask])
            for particle_id, cell_idx in zip(valid_particle_ids, cell_indices):
                neighbouring_cells[particle_id].append(cell_idx)
        return neighbouring_cells

    # def get_particles_in_cells(self, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> List[List[int]]:
    #     valid_mask = self._get_mask(cx, cy, cz)
    #     result: List[List[int]] = [[] for _ in range(cx.size)]
    #     if np.any(valid_mask):
    #         valid_indices = np.where(valid_mask)[0]
    #         cell_indices = self._cell_coords_to_indices(cx[valid_mask], cy[valid_mask], cz[valid_mask])
    #         for output_idx, cell_idx in zip(valid_indices, cell_indices):
    #             result[output_idx] = self.cells[cell_idx].copy()
    #     return result
    
    def _get_particles_in_cells(self, cells_indices: np.ndarray | List[int]) -> List[List[int]]:
        # same here, because the number of particles in each cell can vary,
        # we can't make a fully vectorized version, but we can still use numpy to filter out invalid cell indices
        # so the benefit is very small
        cells_indices = np.asarray(cells_indices, dtype=np.int32)
        valid_mask = (0 <= cells_indices) & (cells_indices < self._total_cells)
        return [
            self.cells[cell_idx].copy() if is_valid else []
            for cell_idx, is_valid in zip(cells_indices, valid_mask)
        ]
    
    def get_potential_collision_pairs(self, positions: np.ndarray) -> np.ndarray:
        """
        Returns an array of particle index pairs that are in the same or neighbouring cells.
        """
        neighbouring_cells = self._get_neighbouring_cells_indices(positions)
        pair_chunks = []
        for particle_id, cell_indices in enumerate(neighbouring_cells):
            particles_in_neighbouring_cells = self._get_particles_in_cells(cell_indices)

            candidate_ids = np.fromiter((
                    other_id
                    for particles_in_cell in particles_in_neighbouring_cells
                    for other_id in particles_in_cell # transforms list of lists into a flat list (like reshape -1)
                ), dtype=np.int32)
            if candidate_ids.size == 0:
                continue

            # Keeps only pairs (particle_id, other_id) with other_id > particle_id
            # to prevent (i,i) or duplicates like (i,j) and (j,i)
            candidate_ids = candidate_ids[candidate_ids > particle_id]
            if candidate_ids.size == 0:
                continue

            pairs = np.column_stack((
                np.full(candidate_ids.size, particle_id, dtype=np.int32), # first column is only the particle_id
                candidate_ids)) # second column is the other particle id (with higher index)
            pair_chunks.append(pairs)
        if len(pair_chunks) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.vstack(pair_chunks) # pair_chunks is a list of arrays for each particle