"""
Module to handle genetic distance measures between isolates.
"""
from dataclasses import dataclass, replace
import inspect
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components as sp_connected_components
from sklearn.manifold import MDS
from sklearn.neighbors import BallTree
from seroepi.constants import DistanceMetricType, DistanceFlavour


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DistancesBase(ABC):
    matrix: csr_array
    index: pd.Series
    metric_type: DistanceMetricType
    max_value: float = None  # Required if converting between absolute and relative

    def __post_init__(self):
        """Validates the consistency of the distance matrix and labels."""
        # Airtight check 1: Dimensions must match
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if len(self.index) != self.matrix.shape[0]:
            raise ValueError("Number of labels must match matrix dimensions.")
        self.index.name = 'sample_id'

    @abstractmethod
    def get_clusters(self, *args, **kwargs) -> pd.Series: ...

    def layout(self, random_state: int = 42, n_init: int = 1, max_iter: int = 100) -> np.ndarray:
        """
        Calculates a 2D layout for the distance matrix using Multi-Dimensional Scaling (MDS).

        Args:
            random_state: Seed for reproducibility. Defaults to 42.
            n_init: Number of initialization runs. Defaults to 1 for speed.
            max_iter: Maximum iterations. Defaults to 100 for speed.

        Returns:
            A numpy array of shape (n_samples, 2) containing the 2D coordinates.
        """
        dense_dist = self.matrix.toarray().astype(float)

        if self.metric_type in [DistanceMetricType.ABSOLUTE_SIMILARITY, DistanceMetricType.RELATIVE_SIMILARITY]:
            # Convert similarity to dissimilarity for MDS
            max_val = self.max_value if self.max_value is not None else 1.0
            dense_dist = max_val - dense_dist
            mask = (dense_dist == max_val) & (~np.eye(dense_dist.shape[0], dtype=bool))
            if mask.any():
                dense_dist[mask] = max_val * 2
        else:
            # If the matrix was sparse, 0s off the diagonal represent missing data. Fill with max distance.
            mask = (dense_dist == 0) & (~np.eye(dense_dist.shape[0], dtype=bool))
            if mask.any():
                dense_dist[mask] = dense_dist.max() * 2

        mds_kwargs = {
            'n_components': 2,
            'dissimilarity': 'precomputed',
            'random_state': random_state,
            'n_init': n_init,
            'max_iter': max_iter
        }
        
        # Dynamically suppress scikit-learn API FutureWarnings
        sig = inspect.signature(MDS.__init__)
        if 'normalized_stress' in sig.parameters: mds_kwargs['normalized_stress'] = 'auto'
        if 'init' in sig.parameters: mds_kwargs['init'] = 'random'

        return MDS(**mds_kwargs).fit_transform(dense_dist)


@dataclass(frozen=True, slots=True)
class GenomicDistances(DistancesBase):

    @classmethod
    def from_file(cls, filepath_or_buffer: Union[str, Path], flavour: Union[str, DistanceFlavour]) -> 'GenomicDistances':
        """
        Factory method to parse a distance matrix or tree from a file based on flavour.
        """
        flavour_val = flavour.value if isinstance(flavour, DistanceFlavour) else flavour
        if flavour_val == DistanceFlavour.PATHOGENWATCH.value:
            return cls.from_pathogenwatch(filepath_or_buffer)
        elif flavour_val == DistanceFlavour.SKA2.value:
            return cls.from_ska2(filepath_or_buffer)
        elif flavour_val == DistanceFlavour.NEWICK.value:
            return cls.from_newick(Path(filepath_or_buffer).read_text())
        else:
            raise ValueError(f"Unknown distance flavour: {flavour_val}")

    @classmethod
    def from_pairwise(cls, query_col: pd.Series, target_col: pd.Series, weight_col: pd.Series,
                      metric_type: DistanceMetricType = DistanceMetricType.ABSOLUTE_DISTANCE) -> 'GenomicDistances':
        """
        Creates a Distances instance from long-format pairwise data.

        Args:
            query_col: Series containing the first isolate IDs.
            target_col: Series containing the second isolate IDs.
            weight_col: Series containing the distances/similarities.
            metric_type: The type of metric provided. Defaults to ABSOLUTE_DISTANCE.

        Returns:
            A new Distances instance.
        """
        # 1. Drop to pure NumPy for speed
        q_vals = query_col.values
        t_vals = target_col.values
        
        # 2. Use pd.factorize for highly optimized O(N) string hashing 
        # (significantly faster than np.unique's O(N log N) sorting approach)
        codes, uids = pd.factorize(np.concatenate([q_vals, t_vals]))
        
        # 4. Split the mapped indices back into rows and columns
        half = len(q_vals)
        rows = codes[:half]
        cols = codes[half:]
        # 5. Build the matrix
        n = len(uids)
        M = coo_array((weight_col.values, (rows, cols)), shape=(n, n))
        # 6. Symmetrize and ensure it returns a CSR matrix
        return cls(M.maximum(M.T).tocsr(), pd.Series(uids), metric_type)

    @classmethod
    def from_ska2(cls, filepath_or_buffer) -> 'GenomicDistances':
        """
        Parses a pairwise distance matrix from SKA2 output.

        Args:
            filepath_or_buffer: Path to the SKA2 distance file.

        Returns:
            A new Distances instance.
        """
        df = pd.read_table(filepath_or_buffer, usecols=(0, 1, 2))
        return cls.from_pairwise(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])

    @classmethod
    def from_pathogenwatch(cls, filepath_or_buffer) -> 'GenomicDistances':
        """
        Parses a square distance matrix from Pathogenwatch.

        Args:
            filepath_or_buffer: Path to the Pathogenwatch CSV file.

        Returns:
            A new Distances instance.
        """
        df = pd.read_csv(filepath_or_buffer, index_col=0)
        M = coo_array(df.values)
        M = M.maximum(M.T)
        return cls(M.tocsr(), pd.Series(df.columns), DistanceMetricType.ABSOLUTE_DISTANCE)

    @classmethod
    def from_newick(cls, newick_string: str) -> 'GenomicDistances':
        """
        Parses a Newick string and calculates patristic distances.

        Requires Biopython (`pip install biopython`).

        Args:
            newick_string: The Newick tree string.

        Returns:
            A new Distances instance.

        Raises:
            ImportError: If biopython is not installed.
        """
        try:
            from Bio import Phylo
        except ImportError:
            raise ImportError("biopython is required to calculate patristics. Install with seroepi[dev]")

        # 1. Parse the tree
        tree = Phylo.read(StringIO(newick_string), "newick")

        # 2. Extract terminals (leaves)
        terminals = tree.get_terminals()
        labels = [leaf.name for leaf in terminals]
        n = len(terminals)

        # 3. Populate a dense NumPy array (symmetric distance calculation)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dist = tree.distance(terminals[i], terminals[j])
                matrix[i, j] = dist
                matrix[j, i] = dist

                # 4. Convert to CSR array and return as a Distances instance
        return cls(
            matrix=csr_array(matrix),
            index=pd.Series(labels),
            metric_type=DistanceMetricType.ABSOLUTE_DISTANCE
        )

    def get_clusters(self, threshold: int = 20) -> pd.Series:
        """
        Identifies clusters via connected components based on a distance threshold.

        Args:
            threshold: Maximum distance to consider isolates as connected.
                Defaults to 20 (e.g., 20 SNPs).

        Returns:
            A Series of cluster labels indexed by isolate IDs.
        """
        adj = self.matrix.copy()  # Make a copy of the CSR array to avoid mutating the frozen original
        # Convert to a binary adjacency array:
        # If distance <= threshold, make it a 1 (Valid Edge).
        # If distance > threshold, make it a 0 (Severed Edge).
        adj.data = (adj.data <= threshold).astype(np.int8)
        # Safely eliminate the 0s (which are now only the severed edges).
        # Identical clones are safe because their distance of 0 was turned into a 1!
        adj.eliminate_zeros()
        adj.setdiag(1)  # Ensure every sample is connected to itself on the diagonal
        _, labels = sp_connected_components(csgraph=adj, directed=False, return_labels=True)
        return pd.Series(labels, index=self.index, dtype='category', name=f"connected_components_{threshold=}").cat.as_ordered()

    def to_type(self, target_type: DistanceMetricType) -> 'GenomicDistances':
        """
        Converts the distances to a different metric type.

        Args:
            target_type: The desired target MetricType.

        Returns:
            A new Distances instance with the converted matrix.

        Raises:
            ValueError: If conversion requires `max_value` but it is not set.
        """
        if self.metric_type == target_type:
            return self

        # If crossing the Absolute <-> Relative boundary, we need max_value
        needs_max = {DistanceMetricType.ABSOLUTE_DISTANCE, DistanceMetricType.ABSOLUTE_SIMILARITY}
        targets_norm = {DistanceMetricType.RELATIVE_DISTANCE, DistanceMetricType.RELATIVE_SIMILARITY}

        if (self.metric_type in needs_max and target_type in targets_norm) or \
                (self.metric_type in targets_norm and target_type in needs_max):
            if self.max_value is None:
                raise ValueError(f"Cannot convert between Absolute and Relative without a max_value.")

        # Operate strictly on the internal 1D .data array to preserve sparsity
        # and prevent massive dense matrix bleed
        new_mat = self.matrix.copy()

        # 1. Standardize to Relative Distance first (as a base state)
        if self.metric_type == DistanceMetricType.ABSOLUTE_DISTANCE:
            new_mat.data = new_mat.data / self.max_value
        elif self.metric_type == DistanceMetricType.ABSOLUTE_SIMILARITY:
            new_mat.data = 1.0 - (new_mat.data / self.max_value)
        elif self.metric_type == DistanceMetricType.RELATIVE_SIMILARITY:
            new_mat.data = 1.0 - new_mat.data

        # 2. Convert from base state (Relative Distance) to Target
        if target_type == DistanceMetricType.RELATIVE_SIMILARITY:
            new_mat.data = 1.0 - new_mat.data
        elif target_type == DistanceMetricType.ABSOLUTE_DISTANCE:
            new_mat.data = new_mat.data * self.max_value
        elif target_type == DistanceMetricType.ABSOLUTE_SIMILARITY:
            new_mat.data = (1.0 - new_mat.data) * self.max_value
            
        return replace(self, matrix=new_mat, metric_type=target_type)


class TransmissionDistances(DistancesBase):

    @classmethod
    def from_spatiotemporal(
            cls,
            sample_ids: pd.Series,
            coords: np.ndarray,
            dates: np.ndarray,
            clones: np.ndarray,
            spatial_threshold_km: float = 10.0,
            temporal_threshold_days: int = 20,
    ) -> 'TransmissionDistances':
        """Builds a sparse transmission adjacency graph from spatiotemporal arrays."""
        n = len(sample_ids)
        global_rows = []
        global_cols = []

        # Filter out rows missing required spatial or temporal data
        valid_mask = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]) | np.isnan(dates))
        
        # Fast Pandas grouping avoids O(N^2) boolean array masking in loops
        valid_df = pd.DataFrame({'clone': clones, 'idx': np.arange(n)})[valid_mask]
        
        radius_radians = spatial_threshold_km / 6371.0

        for clone_val, group in valid_df.groupby('clone', observed=True):
            if pd.isna(clone_val):
                continue

            valid_idx = group['idx'].values
            if len(valid_idx) < 2:
                continue

            group_coords = coords[valid_idx]
            group_dates = dates[valid_idx]

            tree = BallTree(group_coords, metric='haversine')
            spatial_neighbors = tree.query_radius(group_coords, r=radius_radians)

            for i, neighbors in enumerate(spatial_neighbors):
                time_diffs = np.abs(group_dates[i] - group_dates[neighbors])
                valid_neighbors = neighbors[time_diffs <= temporal_threshold_days]

                global_rows.extend([valid_idx[i]] * len(valid_neighbors))
                global_cols.extend(valid_idx[valid_neighbors])

        if global_rows:
            adj = csr_array((np.ones(len(global_rows), dtype=np.int8), (global_rows, global_cols)), shape=(n, n))
            adj = adj.maximum(adj.T)  # Ensure undirected symmetry
        else:
            adj = csr_array((n, n), dtype=np.int8)

        return cls(matrix=adj, index=pd.Series(sample_ids.values, name='sample_id'),
                   metric_type=DistanceMetricType.ABSOLUTE_SIMILARITY, max_value=1.0)

    def get_clusters(self) -> pd.Series:
        """Extracts cluster labels directly from the pre-computed adjacency network."""
        _, labels = sp_connected_components(csgraph=self.matrix, directed=False, return_labels=True)
        return pd.Series(labels, index=self.index, dtype='category').cat.as_ordered()