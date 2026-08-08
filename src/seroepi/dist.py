"""Module to handle genetic distance measures between isolates."""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components as sp_connected_components
import networkx as nx
from sklearn.neighbors import BallTree

from seroepi.constants import DistanceFlavour, DistanceEpidemiologicalDomain


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DistancesBase(ABC):  # noqa: D101
    matrix: csr_array
    index: pl.Series
    metric_type: DistanceEpidemiologicalDomain
    max_value: float | None = None  # Required if converting between absolute and relative

    def __post_init__(self):  # noqa: ANN204
        """Validates the consistency of the distance matrix and labels."""
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if len(self.index) != self.matrix.shape[0]:
            raise ValueError("Number of labels must match matrix dimensions.")

    @abstractmethod
    def get_clusters(self, *args, **kwargs) -> pl.Series: ...  # noqa: ANN002, ANN003, D102

    def layout(self, random_state: int = 42, threshold: float | None = None, max_iter: int = 50) -> np.ndarray:
        """Calculates a 2D Force-Directed layout for the distance matrix."""
        dense_dist = self.matrix.toarray().astype(float)
        
        G = nx.Graph()
        G.add_nodes_from(range(dense_dist.shape[0]))
        
        if threshold is not None:
            adj = dense_dist <= threshold
            rows, cols = np.where(np.triu(adj, k=1))
            edges = zip(rows, cols)
            G.add_edges_from(edges)
        else:
            if self.metric_type in [DistanceEpidemiologicalDomain.ABSOLUTE_SIMILARITY, DistanceEpidemiologicalDomain.RELATIVE_SIMILARITY]:
                weights = dense_dist
            else:
                max_val = dense_dist.max() if dense_dist.max() > 0 else 1
                weights = 1.0 - (dense_dist / max_val)
            
            rows, cols = np.where(np.triu(np.ones_like(dense_dist, dtype=bool), k=1))
            edges = ((r, c, {'weight': weights[r, c]}) for r, c in zip(rows, cols))
            G.add_edges_from(edges)
            
        pos_dict = nx.spring_layout(G, seed=random_state, iterations=max_iter)
        return np.array([pos_dict[i] for i in range(dense_dist.shape[0])])


@dataclass(frozen=True, slots=True)
class GenomicDistances(DistancesBase):  # noqa: D101
    @classmethod
    def from_file(cls, filepath_or_buffer: str | Path, flavour: str | DistanceFlavour) -> "GenomicDistances":
        """Factory method to parse a distance matrix or tree from a file based on flavour."""
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
    def from_pairwise(
        cls,
        query_col: Any,  # noqa: ANN401
        target_col: Any,  # noqa: ANN401
        weight_col: Any,  # noqa: ANN401
        metric_type: DistanceEpidemiologicalDomain = DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE,
    ) -> "GenomicDistances":
        """Creates a Distances instance from long-format pairwise data."""
        q_vals = query_col.to_numpy() if isinstance(query_col, pl.Series) else np.asarray(query_col)
        t_vals = target_col.to_numpy() if isinstance(target_col, pl.Series) else np.asarray(target_col)
        w_vals = weight_col.to_numpy() if isinstance(weight_col, pl.Series) else np.asarray(weight_col)

        combined = np.concatenate([q_vals, t_vals])
        uids, codes = np.unique(combined, return_inverse=True)

        half = len(q_vals)
        rows = codes[:half]
        cols = codes[half:]

        n = len(uids)
        M = coo_array((w_vals, (rows, cols)), shape=(n, n))
        return cls(M.maximum(M.T).tocsr(), pl.Series("sample_id", uids), metric_type)

    @classmethod
    def from_ska2(cls, filepath_or_buffer: Any) -> "GenomicDistances":  # noqa: ANN401
        """Parses a pairwise distance matrix from SKA2 output."""
        df = pl.read_csv(filepath_or_buffer, separator="\t", has_header=False)
        cols = df.columns[:3]
        return cls.from_pairwise(df[cols[0]], df[cols[1]], df[cols[2]])

    @classmethod
    def from_pathogenwatch(cls, filepath_or_buffer: Any) -> "GenomicDistances":  # noqa: ANN401
        """Parses a square distance matrix from Pathogenwatch."""
        df = pl.read_csv(filepath_or_buffer)
        # First column is the sample_id header
        cols = df.columns[1:]
        matrix_data = df.select(cols).to_numpy()
        M = coo_array(matrix_data)
        M = M.maximum(M.T)
        return cls(M.tocsr(), pl.Series("sample_id", cols), DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE)

    @classmethod
    def from_newick(cls, newick_string: str) -> "GenomicDistances":
        """Parses a Newick string and calculates patristic distances."""
        try:
            from Bio import Phylo  # type: ignore
        except ImportError:
            raise ImportError("biopython is required to calculate patristics. Install with seroepi[dev]")

        tree = Phylo.read(StringIO(newick_string), "newick")
        terminals = tree.get_terminals()
        labels = [leaf.name for leaf in terminals]
        n = len(terminals)

        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dist = tree.distance(terminals[i], terminals[j])
                matrix[i, j] = dist
                matrix[j, i] = dist

        return cls(
            matrix=csr_array(matrix),
            index=pl.Series("sample_id", labels),
            metric_type=DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE,
        )

    def get_clusters(self, threshold: int = 20) -> pl.Series:
        """Identifies clusters via connected components based on a distance threshold."""
        adj = self.matrix.copy()
        adj.data = (adj.data <= threshold).astype(np.int8)
        adj.eliminate_zeros()
        adj.setdiag(1)
        _, labels = sp_connected_components(csgraph=adj, directed=False, return_labels=True)
        return pl.Series(f"connected_components_{threshold}", labels)

    def to_type(self, target_type: DistanceEpidemiologicalDomain) -> "GenomicDistances":  # noqa: D102
        if self.metric_type == target_type:
            return self

        needs_max = {DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE, DistanceEpidemiologicalDomain.ABSOLUTE_SIMILARITY}
        targets_norm = {DistanceEpidemiologicalDomain.RELATIVE_DISTANCE, DistanceEpidemiologicalDomain.RELATIVE_SIMILARITY}

        if (self.metric_type in needs_max and target_type in targets_norm) or (
            self.metric_type in targets_norm and target_type in needs_max
        ):
            if self.max_value is None:
                raise ValueError("Cannot convert between Absolute and Relative without a max_value.")

        new_mat = self.matrix.copy()

        if self.metric_type == DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE:
            new_mat.data = new_mat.data / self.max_value
        elif self.metric_type == DistanceEpidemiologicalDomain.ABSOLUTE_SIMILARITY:
            new_mat.data = 1.0 - (new_mat.data / self.max_value)
        elif self.metric_type == DistanceEpidemiologicalDomain.RELATIVE_SIMILARITY:
            new_mat.data = 1.0 - new_mat.data

        if target_type == DistanceEpidemiologicalDomain.RELATIVE_SIMILARITY:
            new_mat.data = 1.0 - new_mat.data
        elif target_type == DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE:
            new_mat.data = new_mat.data * self.max_value
        elif target_type == DistanceEpidemiologicalDomain.ABSOLUTE_SIMILARITY:
            new_mat.data = (1.0 - new_mat.data) * self.max_value

        return replace(self, matrix=new_mat, metric_type=target_type)


class TransmissionDistances(DistancesBase):  # noqa: D101
    @classmethod
    def from_spatiotemporal(
        cls,
        sample_ids: Any,  # noqa: ANN401
        coords: np.ndarray,
        dates: np.ndarray,
        clones: np.ndarray,
        spatial_threshold_km: float = 10.0,
        temporal_threshold_days: int = 20,
    ) -> "TransmissionDistances":
        """Builds a sparse transmission adjacency graph from spatiotemporal arrays."""
        n = len(sample_ids)
        global_rows = []
        global_cols = []

        valid_mask = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]) | np.isnan(dates))
        valid_df = pl.DataFrame({"clone": clones, "idx": np.arange(n)}).filter(
            pl.Series("mask", valid_mask) & ~pl.col("clone").is_null()
        )

        radius_radians = spatial_threshold_km / 6371.0

        if len(valid_df) > 0:
            for clone_val, group in valid_df.group_by("clone"):
                valid_idx = group["idx"].to_numpy()
                if len(valid_idx) < 2:
                    continue

                group_coords = coords[valid_idx]
                group_dates = dates[valid_idx]

                tree = BallTree(group_coords, metric="haversine")
                spatial_neighbors = tree.query_radius(group_coords, r=radius_radians)

                for i, neighbors in enumerate(spatial_neighbors):
                    time_diffs = np.abs(group_dates[i] - group_dates[neighbors])
                    valid_neighbors = neighbors[time_diffs <= temporal_threshold_days]

                    global_rows.extend([valid_idx[i]] * len(valid_neighbors))
                    global_cols.extend(valid_idx[valid_neighbors])

        if global_rows:
            adj = csr_array((np.ones(len(global_rows), dtype=np.int8), (global_rows, global_cols)), shape=(n, n))
            adj = adj.maximum(adj.T)
        else:
            adj = csr_array((n, n), dtype=np.int8)

        index_vals = sample_ids.to_list() if isinstance(sample_ids, pl.Series) else list(sample_ids)
        return cls(
            matrix=adj,
            index=pl.Series("sample_id", index_vals),
            metric_type=DistanceEpidemiologicalDomain.ABSOLUTE_SIMILARITY,
            max_value=1.0,
        )

    def get_clusters(self) -> pl.Series:
        """Extracts cluster labels directly from the pre-computed adjacency network."""
        _, labels = sp_connected_components(csgraph=self.matrix, directed=False, return_labels=True)
        return pl.Series("transmission_clusters", labels)
