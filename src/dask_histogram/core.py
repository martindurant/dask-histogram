"""Dask Histogram core High Level Graph API."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Iterable, List, Tuple, Union

import boost_histogram as bh
import dask.array as da
import dask.bag as db
from dask.bag.core import empty_safe_aggregate, partition_all
from dask.base import DaskMethodsMixin, is_dask_collection, tokenize
from dask.dataframe.core import partitionwise_graph as partitionwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as tget
from dask.utils import is_dataframe_like, key_split

if TYPE_CHECKING:
    import numpy as np

    from .typing import DaskCollection
else:
    DaskCollection = object

__all__ = (
    "AggHistogram",
    "PartitionedHistogram",
    "clone",
    "factory",
    "to_dask_array",
)


def clone(histref: bh.Histogram = None) -> bh.Histogram:
    """Create a Histogram object based on another.

    The axes and storage of the `histref` will be used to create a new
    Histogram object.

    Parameters
    ----------
    histref : bh.Histogram
        The reference Histogram.

    Returns
    -------
    bh.Histogram
        New Histogram with identical axes and storage.

    """
    if histref is None:
        return bh.Histogram()
    return bh.Histogram(*histref.axes, storage=histref._storage_type())


def _blocked_sa_w(
    sample: Any,
    weights: Any,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted."""
    if sample.ndim == 1:
        return clone(histref).fill(sample, weight=weights)
    elif sample.ndim == 2:
        return clone(histref).fill(*(sample.T), weight=weights)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_sa(
    sample: Any,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; unweighted."""
    if sample.ndim == 1:
        return clone(histref).fill(sample, weight=None)
    elif sample.ndim == 2:
        return clone(histref).fill(*(sample.T), weight=None)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_ma_w(
    *sample: Any,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; unweighted."""
    weights = sample[-1]
    sample = sample[:-1]
    return clone(histref).fill(*sample, weight=weights)


def _blocked_ma(
    *sample: Any,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; unweighted."""
    return clone(histref).fill(*sample, weight=None)


def _blocked_df_w(
    sample: Any,
    weights: Any,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted."""
    return clone(histref).fill(*(sample[c] for c in sample.columns), weight=weights)


def _blocked_df(
    sample: Any,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    return clone(histref).fill(*(sample[c] for c in sample.columns), weight=None)


class AggHistogram(db.Item):
    """Aggregated Histogram collection.

    The class constructor is typically used internally; for users
    :py:func:`dask_histogram.core.histogram` is recommended (along
    with the `dask_histogram.routines` module).

    See Also
    --------
    dask_histogram.core.histogram

    Parameters
    ----------
    dsk : dask.highlevelgraph.HighLevelGraph
        High level graph providing the computation.
    key : str
        Unique identifier for the Dask graph.
    histref : boost_histogram.Histogram
        Reference histogram providing axes, storage, and metadata.

    """

    def __init__(self, dsk: HighLevelGraph, key: str, histref: bh.Histogram) -> None:
        self._dsk = dsk
        self._key = key
        self._histref: bh.Histogram = histref

    @property
    def dask(self) -> HighLevelGraph:
        """High level graph object."""
        return self._dsk

    @property
    def key(self) -> str:
        """Key in a Dask graph."""
        return self._key

    @property
    def name(self) -> str:
        """Duplicate of `key`."""
        return self._key

    @property
    def histref(self) -> bh.Histogram:
        """Empty reference boost-histogram object."""
        return self._histref

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        return self.histref.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the histogram as an array."""
        return self.histref.shape

    @property
    def size(self) -> int:
        """Size of the histogram."""
        return self.histref.size

    def __str__(self) -> str:
        return f"dask_histogram.AggHistogram<{key_split(self.name)}>"

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(tget)

    def to_dask_array(
        self, flow: bool = False, dd: bool = False
    ) -> Union[Tuple[da.Array, ...], Tuple[da.Array, Tuple[da.Array, ...]]]:
        """Convert histogram object to dask.array form.

        Parameters
        ----------
        flow : bool
            Include the flow bins.
        dd : bool
            Use the histogramdd return syntax, where the edges are in a tuple.
            Otherwise, this is the histogram/histogram2d return style.

        Returns
        -------
        contents : dask.array.Array
            The bin contents
        *edges : dask.array.Array
            The edges for each dimension

        """
        return to_dask_array(self, flow=flow, dd=dd)

    def to_boost(self) -> bh.Histogram:
        return self.compute()

    def __array__(self) -> np.ndarray:
        return self.compute().__array__()

    def __iadd__(self, other) -> AggHistogram:
        return _iadd(self, other)

    def __add__(self, other: Any) -> AggHistogram:
        return self.__iadd__(other)

    def __radd__(self, other: Any) -> AggHistogram:
        return self.__iadd__(other)

    def __itruediv__(self, other: Any) -> AggHistogram:
        return _itruediv(self, other)

    def __truediv__(self, other: Any) -> AggHistogram:
        return self.__itruediv__(other)

    def __idiv__(self, other: Any) -> AggHistogram:
        return self.__itruediv__(other)

    def __div__(self, other: Any) -> AggHistogram:
        return self.__idiv__(other)

    def __imul__(self, other: Any) -> AggHistogram:
        return _imul(self, other)

    def __mul__(self, other: Any) -> AggHistogram:
        return self.__imul__(other)

    def __rmul__(self, other: Any) -> AggHistogram:
        return self.__mul__(other)


def _finalize_partitioned_histogram(results: Any) -> Any:
    return results


class PartitionedHistogram(DaskMethodsMixin):
    def __init__(
        self, dsk: HighLevelGraph, name: str, npartitions: int, histref: bh.Histogram
    ) -> None:
        self.dask: HighLevelGraph = dsk
        self.name: str = name
        self.npartitions: int = npartitions
        self._histref: bh.Histogram = histref

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> List[Tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> Tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return _finalize_partitioned_histogram, ()

    def _rebuild(self, dsk: Any, *, rename: Any = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.npartitions, self.histref)

    def __str__(self) -> str:
        return "dask_histogram.PartitionedHistogram,<%s, npartitions=%d>" % (
            key_split(self.name),
            self.npartitions,
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(tget)

    @property
    def histref(self) -> bh.Histogram:
        """boost_histogram.Histogram: reference histogram."""
        return self._histref

    def reduced(self, split_every: int = None) -> AggHistogram:
        """FIXME: Short description.

        FIXME: Long description.

        Parameters
        ----------
        split_every : int
            FIXME: Add docs.

        Returns
        -------
        AggHistogram
            FIXME: Add docs.

        Examples
        --------
        FIXME: Add docs.

        """
        return _reduction(self, split_every=split_every)


def _reduction(ph: PartitionedHistogram, split_every: int = None) -> AggHistogram:
    if split_every is None:
        split_every = 4
    if split_every is False:
        split_every = ph.npartitions

    token = tokenize(ph, sum, split_every)
    fmt = f"hist-aggregate-{token}"
    k = ph.npartitions
    b = ph.name
    d = 0
    dsk = {}
    while k > split_every:
        c = f"{fmt}{d}"
        for i, inds in enumerate(partition_all(split_every, range(k))):
            dsk[(c, i)] = (
                empty_safe_aggregate,
                sum,
                [(b, j) for j in inds],
                False,
            )
        k = i + 1
        b = c
        d += 1
    dsk[(fmt, 0)] = (
        empty_safe_aggregate,
        sum,
        [(b, j) for j in range(k)],
        True,
    )

    dsk[fmt] = dsk.pop((fmt, 0))  # type: ignore
    g = HighLevelGraph.from_collections(fmt, dsk, dependencies=[ph])
    return AggHistogram(g, fmt, histref=ph.histref)


def _dependencies(
    *args: DaskCollection,
    weights: DaskCollection = None,
) -> Tuple[DaskCollection, ...]:
    if weights is not None:
        return (*args, weights)
    return args


def _reduced_histogram(
    *data: DaskCollection,
    histref: bh.Histogram,
    weights: DaskCollection = None,
    split_every: int = None,
) -> AggHistogram:
    name = "hist-on-block-{}".format(tokenize(data, histref, weights))
    if len(data) == 1 and not is_dataframe_like(data[0]):
        x = data[0]
        if weights is not None:
            g = partitionwise(_blocked_sa_w, name, x, weights, histref=histref)
        else:
            g = partitionwise(_blocked_sa, name, x, histref=histref)
    elif len(data) == 1 and is_dataframe_like(data[0]):
        x = data[0]
        if weights is not None:
            g = partitionwise(_blocked_df_w, name, x, weights, histref=histref)
        else:
            g = partitionwise(_blocked_df, name, x, histref=histref)
    else:
        if weights is not None:
            g = partitionwise(_blocked_ma_w, name, *data, weights, histref=histref)
        else:
            g = partitionwise(_blocked_ma, name, *data, histref=histref)

    dependencies = _dependencies(*data, weights=weights)
    hlg = HighLevelGraph.from_collections(name, g, dependencies=dependencies)
    ph = PartitionedHistogram(hlg, name, data[0].npartitions, histref=histref)
    return ph.reduced(split_every=split_every)


def to_dask_array(
    agghist: AggHistogram,
    flow: bool = False,
    dd: bool = False,
) -> Union[Tuple[DaskCollection, List[DaskCollection]], Tuple[DaskCollection, ...]]:
    """FIXME: Short description.

    FIXME: Long description.

    Parameters
    ----------
    agghist : AggHistogram
        FIXME: Add docs.
    flow : bool
        FIXME: Add docs.
    dd : bool
        FIXME: Add docs.

    Returns
    -------
    Union[Tuple[DaskCollection, List[DaskCollection]], Tuple[DaskCollection, ...]]
        FIXME: Add docs.

    Examples
    --------
    FIXME: Add docs.

    """
    name = "to-dask-array-{}".format(tokenize(agghist))
    zeros = (0,) * agghist.histref.ndim
    dsk = {(name, *zeros): (lambda x, f: x.to_numpy(flow=f)[0], agghist.key, flow)}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(agghist,))
    shape = agghist.histref.shape
    if flow:
        shape = tuple(i + 2 for i in shape)
    int_storage = agghist.histref._storage_type in (
        bh.storage.Int64,
        bh.storage.AtomicInt64,
    )
    dt = int if int_storage else float
    c = da.Array(graph, name=name, shape=shape, chunks=shape, dtype=dt)
    axes = agghist.histref.axes
    edges = (da.asarray(ax.edges) for ax in axes)
    if dd:
        return (c, list(edges))
    return (c, *(tuple(edges)))


class BinaryOp:
    def __init__(self, func, name=None):
        self.func = func
        if name is None:
            self.__name__ = func.__name__
        else:
            self.__name__ = name

    def __call__(self, a, b):
        name = "{}-hist-{}".format(self.__name__, tokenize(a, b))
        deps = []
        if is_dask_collection(a):
            deps.append(a)
            k1 = a.name
        else:
            k1 = a
        if is_dask_collection(b):
            deps.append(b)
            k2 = b.name
        else:
            k2 = b
        k1 = a.__dask_tokenize__() if is_dask_collection(a) else a
        k2 = b.__dask_tokenize__() if is_dask_collection(b) else b
        llg = {name: (self.func, k1, k2)}
        g = HighLevelGraph.from_collections(name, llg, dependencies=deps)
        try:
            ref = a.histref
        except AttributeError:
            ref = b.histref
        return AggHistogram(g, name, histref=ref)


_iadd = BinaryOp(operator.iadd, name="add")
_imul = BinaryOp(operator.imul, name="mul")
_itruediv = BinaryOp(operator.itruediv, name="div")


def factory(
    *data: DaskCollection,
    histref: bh.Histogram = None,
    axes: Iterable[bh.axis.Axis] = None,
    storage: bh.storage.Storage = None,
    weights: DaskCollection = None,
    split_every: int = None,
) -> AggHistogram:
    """FIXME: Short description.

    FIXME: Long description.

    Parameters
    ----------
    *data : DaskCollection
        FIXME: Add docs.
    histref : bh.Histogram
        FIXME: Add docs.
    axes : Tuple[bh.axis.Axis, ...]
        FIXME: Add docs.
    storage : bh.storage.Storage
        FIXME: Add docs.
    weights : DaskCollection
        FIXME: Add docs.
    split_every : int
        FIXME: Add docs.

    Returns
    -------
    AggHistogram
        FIXME: Add docs.

    Raises
    ------
    ValueError
        FIXME: Add docs.

    Examples
    --------
    FIXME: Add docs.

    """
    if histref is None and axes is None:
        raise ValueError("Either histref or axes must be defined.")
    elif histref is None:
        if storage is None:
            storage = bh.storage.Double()
        histref = bh.Histogram(*axes, storage=storage)  # type: ignore
    return _reduced_histogram(
        *data, histref=histref, weights=weights, split_every=split_every
    )
