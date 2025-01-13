import boost_histogram as bh
import boost_histogram.numpy as bhnp
import dask.array as da
import numpy as np
import pytest

import dask_histogram.boost as dhb
import dask_histogram.core as dhc


def test_empty():
    h = dhb.Histogram(
        dhb.axis.StrCategory([], growth=True),
        dhb.axis.IntCategory([], growth=True),
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(7, -3.3, 3.3),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=dhb.storage.Weight(),
    )
    control = bh.Histogram(*h.axes, storage=h.storage_type())
    computed = h.compute()

    assert type(computed) is type(control)
    assert np.allclose(computed.values(), control.values())


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_1D(use_weights):
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(dhb.axis.Regular(12, -3, 3), storage=storage)
    h.fill(x, weight=weights)
    h = h.compute()

    control = bh.Histogram(*h.axes, storage=h.storage_type())
    if use_weights:
        control.fill(x.compute(), weight=weights.compute())
    else:
        control.fill(x.compute())

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_2D(use_weights):
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    y = da.random.standard_normal(size=(2000,), chunks=(400,))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(
        dhb.axis.Regular(9, -3.5, 3.5),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=storage,
    )
    h.fill(x, y, weight=weights)
    h = h.compute()

    control = bh.Histogram(*h.axes, storage=h.storage_type())
    if use_weights:
        control.fill(x.compute(), y.compute(), weight=weights.compute())
    else:
        control.fill(x.compute(), y.compute())

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_3D_rectangular(use_weights):
    x = da.random.standard_normal(size=(2000, 3), chunks=(400, 3))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape[0], chunks=x.chunksize[0])
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(7, -3.3, 3.3),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=storage,
    )
    h.fill(x, weight=weights)
    h = h.compute()

    control = bh.Histogram(*h.axes, storage=h.storage_type())
    if use_weights:
        control.fill(*(x.compute().T), weight=weights.compute())
    else:
        control.fill(*(x.compute().T))

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_5D_strcat_intcat_rectangular(use_weights):
    x = da.random.standard_normal(size=(2000, 3), chunks=(400, 3))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape[0], chunks=x.chunksize[0])
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(
        dhb.axis.StrCategory([], growth=True),
        dhb.axis.IntCategory([], growth=True),
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(7, -3.3, 3.3),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=storage,
    )
    for i in range(25):
        h.fill(f"testcat{i + 1}", i + 1, *(x.T), weight=weights)
    h = h.compute()

    control = bh.Histogram(*h.axes, storage=h.storage_type())
    if use_weights:
        for i in range(25):
            control.fill(
                f"testcat{i + 1}", i + 1, *(x.compute().T), weight=weights.compute()
            )
    else:
        for i in range(25):
            control.fill(f"testcat{i + 1}", i + 1, *(x.compute().T))

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())

    assert len(h.axes[0]) == 25 and len(control.axes[0]) == 25
    assert all(cx == hx for cx, hx in zip(control.axes[0], h.axes[0]))

    assert len(h.axes[1]) == 25 and len(control.axes[1]) == 25
    assert all(cx == hx for cx, hx in zip(control.axes[1], h.axes[1]))


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_5D_strcat_intcat_rectangular_dak(use_weights):
    dak = pytest.importorskip("dask_awkward")

    x = dak.from_dask_array(da.random.standard_normal(size=2000, chunks=400))
    y = dak.from_dask_array(da.random.standard_normal(size=2000, chunks=400))
    z = dak.from_dask_array(da.random.standard_normal(size=2000, chunks=400))
    weights = []
    if use_weights:
        for i in range(25):
            weights.append(
                dak.from_dask_array(da.random.uniform(0.5, 0.75, size=2000, chunks=400))
            )
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(
        dhb.axis.StrCategory([], growth=True),
        dhb.axis.IntCategory([], growth=True),
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(7, -3.3, 3.3),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=storage,
    )

    # check that we are using the correct optimizer
    assert h.__dask_optimize__ == dak.lib.optimize.all_optimizations

    for i in range(25):
        h.fill(
            f"testcat{i + 1}", i + 1, x, y, z, weight=weights[i] if weights else None
        )
    h = h.compute()

    control = bh.Histogram(*h.axes, storage=h.storage_type())
    x_c, y_c, z_c = x.compute(), y.compute(), z.compute()
    if use_weights:
        for i in range(25):
            control.fill(
                f"testcat{i + 1}", i + 1, x_c, y_c, z_c, weight=weights[i].compute()
            )
    else:
        for i in range(25):
            control.fill(f"testcat{i + 1}", i + 1, x_c, y_c, z_c)

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())

    assert len(h.axes[0]) == 25 and len(control.axes[0]) == 25
    assert all(cx == hx for cx, hx in zip(control.axes[0], h.axes[0]))

    assert len(h.axes[1]) == 25 and len(control.axes[1]) == 25
    assert all(cx == hx for cx, hx in zip(control.axes[1], h.axes[1]))


def test_histogramdd():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    z = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5), (-2, 2))
    h1 = dhb.histogramdd(
        (x, y, z),
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
    )
    h2 = bhnp.histogramdd(
        (x.compute(), y.compute(), z.compute()),
        bins=bins,
        range=range,
        weights=w,
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_multicolumn_input():
    x = da.random.standard_normal(size=(3_000, 3), chunks=(500, 3))
    bins = [[-3, -2, 2, 3], [-4, -1, 1, 4], [-2, -1, 1, 2]]
    h1 = dhb.histogramdd(x, bins=bins, histogram=dhb.Histogram)
    h2 = bhnp.histogramdd(x.compute(), bins=bins, histogram=bh.Histogram)
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view(), h2.view())


def test_histogramdd_series():
    pytest.importorskip("pandas")

    x = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    y = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    w = da.random.uniform(size=(1000,), chunks=(200,)).to_dask_dataframe()
    bins = (12, 12)
    range = ((-3, 3), (-2.6, 2.6))
    h1 = dhb.histogramdd(
        (x, y),
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
    )
    h1 = h1.compute()
    h2 = bhnp.histogramdd(
        (x.compute(), y.compute()),
        bins=bins,
        range=range,
        weights=w.compute(),
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_arrays_and_series():
    pytest.importorskip("pandas")

    x = da.random.standard_normal(size=(1000,), chunks=(200,))
    y = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    w = da.random.uniform(size=(1000,), chunks=(200,))
    bins = (12, 12)
    range = ((-3, 3), (-2.6, 2.6))
    h1 = dhb.histogramdd(
        (x, y),
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
    )
    h1 = h1.compute()
    h2 = bhnp.histogramdd(
        (x.compute(), y.compute()),
        bins=bins,
        range=range,
        weights=w.compute(),
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_dataframe():
    pytest.importorskip("pandas")
    x = da.random.standard_normal(size=(1000, 3), chunks=(200, 3))
    df = x.to_dask_dataframe(columns=["a", "b", "c"])
    w = da.random.uniform(size=(1000,), chunks=(200,))
    bins = [[-3, -2, 1, 2, 3], [-4, -1, 1, 2, 3, 4], [-3, -2, 0, 2]]
    h1 = dhb.histogramdd(
        df,
        bins=bins,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
    )
    h1 = h1.compute()
    h2 = bhnp.histogramdd(
        df.compute().to_numpy(),
        bins=bins,
        weights=w.compute(),
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogram2d():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 6)
    range = ((-2.5, 2.5), (-2, 2))
    h1 = dhb.histogram2d(
        x,
        y,
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
    )
    h2 = bhnp.histogram2d(
        x.compute(),
        y.compute(),
        bins=bins,
        range=range,
        weights=w,
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogram():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = 7
    range = (-2.5, 2.5)
    h1 = dhb.histogram(
        x,
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
    )
    h2 = bhnp.histogram(
        x.compute(),
        bins=bins,
        range=range,
        weights=w,
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    z = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5), (-2, 2))
    h1, edges1 = dhb.histogramdd((x, y, z), bins=bins, range=range, weights=w)
    h2, edges2 = bhnp.histogramdd(
        (x.compute(), y.compute(), z.compute()), bins=bins, range=range, weights=w
    )
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    for e1, e2 in zip(edges1, edges2):
        np.testing.assert_array_almost_equal(e1.compute(), e2)


def test_histogram2d_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    bins = (5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5))
    h1, edgesx1, edgesy1 = dhb.histogram2d(x, y, bins=bins, range=range)
    h2, edgesx2, edgesy2 = bhnp.histogram2d(
        x.compute(), y.compute(), bins=bins, range=range
    )
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edgesx1.compute(), edgesx2)
    np.testing.assert_array_almost_equal(edgesy1.compute(), edgesy2)


def test_histogram_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    bins = np.array([-3, -2.2, 0, 1.1, 2.2, 3.3])
    h1, edges1 = dhb.histogram(x, bins=bins)
    h2, edges2 = bhnp.histogram(x.compute(), bins=bins)
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edges1.compute(), edges2)

    x = da.random.standard_normal(size=(3_000, 3), chunks=(500, 3))
    bins = (5, 6, 7)
    range = ((-3, 3),) * 3
    h1, edges1 = dhb.histogramdd(x, bins=bins, range=range)
    h2, edges2 = bhnp.histogramdd(x.compute(), bins=bins, range=range)
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    for e1, e2 in zip(edges1, edges2):
        np.testing.assert_array_almost_equal(e1.compute(), e2)


@pytest.mark.parametrize("flow", [True, False])
def test_to_da(flow):
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    h1 = dhb.Histogram(dhb.axis.Regular(10, -3, 3))
    h2 = bh.Histogram(bh.axis.Regular(10, -3, 3))
    h1.fill(x)
    h2.fill(x.compute())
    h1c, h1e = h1.to_dask_array(dd=True, flow=flow)
    h2c, h2e = h2.to_numpy(dd=True, flow=flow)
    np.testing.assert_array_almost_equal(h1c.compute(), h2c)
    np.testing.assert_array_almost_equal(h1e[0].compute(), h2e[0])


def test_to_delayed():
    x = da.random.standard_normal(size=(2_000, 3), chunks=(500, 3))
    bins = [
        [-3, -2.2, 0, 1.1, 2.2, 3.3],
        [-4, -1.1, 0, 2.2, 3.3, 4.4],
        [-2, -0.9, 0, 0.3, 2.2],
    ]
    dh = dhb.Histogram(
        dhb.axis.Variable(bins[0]),
        dhb.axis.Variable(bins[1]),
        dhb.axis.Variable(bins[2]),
    )
    dh.fill(x)
    dh.fill(x * 0.8)

    ch = dhc.clone(dh)
    ch.fill(*(x.compute().T))
    ch.fill(*((x.compute() * 0.8).T))
    np.testing.assert_array_almost_equal(
        dh.to_delayed().compute().to_numpy()[0], ch.to_numpy()[0]
    )


@pytest.mark.parametrize("use_weights", [True, False])
def test_add(use_weights):
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    y = da.random.standard_normal(size=(1000,), chunks=(400,))
    if use_weights:
        xweights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
        yweights = da.random.uniform(0.25, 0.5, size=y.shape, chunks=y.chunks)
        store = dhb.storage.Weight
    else:
        xweights = None
        yweights = None
        store = dhb.storage.Double

    h1 = dhb.Histogram(dhb.axis.Regular(12, -3, 3), storage=store())
    h1.fill(x, weight=xweights)
    h2 = dhb.Histogram(dhb.axis.Regular(12, -3, 3), storage=store())
    h2.fill(y, weight=yweights)

    with pytest.raises(NotImplementedError):
        h3 = h1 + h2

    h3 = h1.compute() + h2.compute()

    h4 = dhb.Histogram(dhb.axis.Regular(12, -3, 3), storage=store())
    h4.fill(x, weight=xweights)

    with pytest.raises(NotImplementedError):
        h4 += h2

    h4 = h4.compute()
    h4 += h2.compute()

    controlx = bh.Histogram(*h1.axes, storage=h1.storage_type())
    controly = bh.Histogram(*h2.axes, storage=h2.storage_type())
    if use_weights:
        controlx.fill(x.compute(), weight=xweights.compute())
        controly.fill(y.compute(), weight=yweights.compute())
    else:
        controlx.fill(x.compute())
        controlx.fill(y.compute())

    c3 = controlx + controly

    assert np.allclose(h3.counts(), c3.counts())
    assert np.allclose(h4.counts(), c3.counts())
    if use_weights:
        assert np.allclose(h3.variances(), c3.variances())
        assert np.allclose(h4.variances(), c3.variances())


def test_name_assignment():
    import dask.array as da

    pytest.importorskip("hist")
    import hist.dask

    x = da.random.normal(size=100)
    h1 = hist.dask.Hist(hist.axis.Regular(10, -2, 2, name="ax1"))
    h2 = h1.copy()
    h1.fill(x)
    h2.axes.name = ("ax2",)
    h2.fill(x)

    assert h1.axes.name == ("ax1",)
    assert h2.axes.name == ("ax2",)

    h1c = h1.compute()
    h2c = h2.compute()

    assert h1c.axes.name == ("ax1",)
    assert h2c.axes.name == ("ax2",)


def test_histref_pickle():
    import pickle

    import dask.array as da

    pytest.importorskip("hist")
    import hist.dask

    x = da.random.normal(size=100)
    h1 = hist.dask.Hist(hist.axis.Regular(10, -2, 2, name="ax1"))
    h1.fill(x)  # forces the internal state histref update

    pickle.dumps(h1._histref)


def test_boost_output_pickles():
    import pickle

    import boost_histogram
    import dask

    import dask_histogram.boost

    h = dask_histogram.boost.Histogram(boost_histogram.axis.Regular(10, 0, 1))

    o = dask.compute(h)
    pickle.dumps(o)


def test_155_boost_factory():
    import boost_histogram as bh

    dak = pytest.importorskip("dask_awkward")

    import dask_histogram as dh

    arr = dak.from_lists([list(range(10))] * 3)
    axis = bh.axis.Regular(10, 0.0, 10.0)
    hist = dh.factory(
        arr,
        axes=(axis,),
    ).compute()
    assert np.all(hist.values() == [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])


def test_155_2():
    import boost_histogram as bh

    import dask_histogram as dh

    dak = pytest.importorskip("dask_awkward")

    arr = dak.from_lists([list(range(10))] * 3)
    axis = bh.axis.Regular(10, 0.0, 10.0)
    hist = dh.factory(
        arr,
        axes=(axis,),
        weights=arr,
    ).compute()
    assert np.all(
        hist.values() == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0]
    )


def test_155_3_2d():
    import boost_histogram as bh

    dak = pytest.importorskip("dask_awkward")

    import dask_histogram as dh

    arr1 = dak.from_lists([list(range(10))] * 3)
    arr2 = dak.from_lists([list(reversed(range(10)))] * 3)
    axis1 = bh.axis.Regular(10, 0.0, 10.0)
    axis2 = bh.axis.Regular(10, 0.0, 10.0)
    hist = dh.factory(
        arr1,
        arr2,
        axes=(axis1, axis2),
        weights=arr1,
    ).compute()
    should_be = (
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    assert np.all(hist.values() == should_be)
