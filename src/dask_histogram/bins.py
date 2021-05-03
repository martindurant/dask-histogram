"""Help determining bin definitions."""

from enum import Enum
from typing import Any, Tuple

import numpy as np


class BinsStyle(Enum):
    """Styles for the bins argument in histogramming functions."""

    Undetermined = 0
    SingleScalar = 1
    MultiScalar = 2
    SingleSequence = 3
    MultiSequence = 4


class RangeStyle(Enum):
    """Styles for the range argument in histogramming functions."""

    Undetermined = 0
    IsNone = 1
    SinglePair = 2
    MultiPair = 3


def bins_style(D: int, bins: Any) -> BinsStyle:
    """Determine the style of the bins argument."""
    if isinstance(bins, int):
        return BinsStyle.SingleScalar
    elif isinstance(bins, (tuple, list)):
        # all integers in the tuple of list
        if all(isinstance(b, int) for b in bins):
            if len(bins) != D and D != 1:
                raise ValueError(
                    "Total number of bins definitions must be equal to the "
                    "dimensionality of the histogram."
                )
            if D == 1:
                return BinsStyle.SingleSequence
            return BinsStyle.MultiScalar
        # sequence of sequences
        else:
            if len(bins) != D:
                raise ValueError(
                    "Total number of bins definitions must be equal to the "
                    "dimensionality of the histogram."
                )
            return BinsStyle.MultiSequence
    elif isinstance(bins, np.ndarray):
        if bins.ndim == 1:
            return BinsStyle.SingleSequence

    raise ValueError(f"Could not determine bin style from bins={bins}")


def bins_range_styles(D: int, bins: Any, range: Any) -> Tuple[BinsStyle, RangeStyle]:
    """Determine the style of the bins and range arguments.

    Parameters
    ----------
    D : int
        The dimensionality of the histogram to be created by the bin
        and range defintions.
    bins : int, sequence if ints, array, or sequence of arrays
        Definition of the bins either by total number of bins in each
        dimension, or by the bin edges in each dimension.
    range : sequence of pairs, optional
        If bins are defined by the total number in each dimension, a
        range must be defined representing the left- and right-most
        edges of the axis. For a multidimensional histogram a single
        pair will represent a (min, max) in each dimension. For
        multiple pairs, the total number must be equal to the
        dimensionality of the histogram.

    Returns
    -------
    BinsStyle
        The style of the bins argument
    RangeStyle
        The style of the range argument

    """
    b_style = bins_style(D, bins)
    r_style = RangeStyle.Undetermined

    # If range is None we can return or raise if the bins are defined by scalars.
    if range is None:
        r_style = RangeStyle.IsNone
        if b_style in [BinsStyle.SingleSequence, BinsStyle.MultiSequence]:
            return b_style, r_style
        else:
            raise ValueError(
                "range cannot be None when bins argument is a scalar or sequence of scalars."
            )

    if b_style == BinsStyle.SingleScalar:
        if len(range) != 2:
            raise ValueError(
                "For a single scalar bin definition, one range tuple must be defined."
            )
        if not isinstance(range[0], (int, float)) or not isinstance(
            range[1], (int, float)
        ):
            raise ValueError(
                "For a single scalar bin definition, one range tuple must be defined."
            )
        r_style = RangeStyle.SinglePair

    elif b_style == BinsStyle.MultiScalar:
        if len(range) != D:
            ValueError(
                "Total number of range pairs must be equal to the dimensionality of the histogram."
            )
        for entry in range:
            if len(entry) != 2:
                raise ValueError("Each range definition must be a pair of numbers.")
        r_style = RangeStyle.MultiPair

    return b_style, r_style