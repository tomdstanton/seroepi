from dataclasses import dataclass
from typing import Generic, TypeVar

import polars as pl

from seroepi.estimators.base import PrevalenceEstimates

T_Result = TypeVar("T_Result")


@dataclass
class ComparisonResult(Generic[T_Result]):
    """Stores the merged comparison of two estimator results."""

    left_name: str
    right_name: str
    left_result: T_Result
    right_result: T_Result
    df: pl.DataFrame


class BaseComparator(Generic[T_Result]):
    """Base class for comparing two estimator results."""

    def compare(
        self, left_name: str, left_result: T_Result, right_name: str, right_result: T_Result
    ) -> ComparisonResult[T_Result]:
        """Compares two estimator results and returns a ComparisonResult."""
        raise NotImplementedError


class PrevalenceComparator(BaseComparator[PrevalenceEstimates]):
    """Comparator for PrevalenceEstimates objects."""

    def compare(
        self, left_name: str, left_result: PrevalenceEstimates, right_name: str, right_result: PrevalenceEstimates
    ) -> ComparisonResult[PrevalenceEstimates]:
        """Performs an outer join (union) of two PrevalenceEstimates dataframes."""
        df_left = left_result.df
        df_right = right_result.df

        # Rename metrics to prevent collisions
        df_left = df_left.rename(
            {
                "estimate": "estimate_left",
                "lower": "lower_left",
                "upper": "upper_left",
            }
        )
        if "n" in df_left.columns:
            df_left = df_left.rename({"n": "n_left", "event": "event_left"})

        df_right = df_right.rename(
            {
                "estimate": "estimate_right",
                "lower": "lower_right",
                "upper": "upper_right",
            }
        )
        if "n" in df_right.columns:
            df_right = df_right.rename({"n": "n_right", "event": "event_right"})

        join_cols = [
            c
            for c in df_left.columns
            if c in df_right.columns
            and c
            not in [
                "estimate_left",
                "lower_left",
                "upper_left",
                "n_left",
                "event_left",
                "estimate_right",
                "lower_right",
                "upper_right",
                "n_right",
                "event_right",
            ]
        ]

        # Use full join to keep all traits
        df_merged = df_left.join(df_right, on=join_cols, how="outer_coalesce")

        # Fill missing values with 0
        df_merged = df_merged.with_columns(
            [
                pl.col("estimate_left").fill_null(0.0),
                pl.col("lower_left").fill_null(0.0),
                pl.col("upper_left").fill_null(0.0),
                pl.col("estimate_right").fill_null(0.0),
                pl.col("lower_right").fill_null(0.0),
                pl.col("upper_right").fill_null(0.0),
            ]
        )

        return ComparisonResult(
            left_name=left_name,
            right_name=right_name,
            left_result=left_result,
            right_result=right_result,
            df=df_merged,
        )
