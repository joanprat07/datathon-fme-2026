"""trend2 / trend0: the weight of the newest information is opt-in and leaves every other arm untouched."""
import numpy as np

from twin.variants import LiveLayer, Variant, get_variant


class _St:
    cap, SCAP, NC, P, now_ms = 6, 3, 2, 6, 10**12
    views_7d = np.full(6, 700.0)
    purch_7d = np.zeros(6)
    sub = np.array([0, 0, 1, 1, 2, 2])
    cat = np.array([0, 0, 0, 1, 1, 1])


def test_variants_resolve():
    base = get_variant("v5-hybrid+m2-intent+m3-live")
    t2 = get_variant("v5t=v5-hybrid+m2-intent+m3-live+trend2")
    assert base.trend_power == 1.0 and Variant().trend_power == 1.0
    assert t2.name == "v5t" and t2.trend_power == 2.0 and t2.model == base.model and t2.live_layer
    assert get_variant("v5-hybrid+m2-intent+m3-live+trend0").trend_power == 0.0


def test_multiplier_power():
    st = _St()
    layer = LiveLayer(st)
    layer._trend = np.array([0.8, 1.0, 1.4, 1.2, 1.0, 0.9], np.float32)
    cand = np.arange(6)
    m1, _ = layer.multipliers(st, cand, np.random.default_rng(0), st.now_ms)
    m1b, _ = layer.multipliers(st, cand, np.random.default_rng(0), st.now_ms, trend_power=1.0)
    m2, _ = layer.multipliers(st, cand, np.random.default_rng(0), st.now_ms, trend_power=2.0)
    m0, _ = layer.multipliers(st, cand, np.random.default_rng(0), st.now_ms, trend_power=0.0)
    assert np.array_equal(m1, m1b)                                  # default path is unchanged
    assert np.allclose(m2 / m1, layer._trend, rtol=1e-5)            # squared trend
    assert np.allclose(m0 * layer._trend, m1, rtol=1e-5)            # trend switched off


def test_seeded_trend_new_product_not_max_boost():
    """A product that appears after the loop started: with the plain trend its first views give it the maximum
    boost (counters start at 0: ~+17% after an hour at an average rate, the 1.4 cap above it, squared by trend2); with trend_src="seed" its inherited 7-day rate is the prior, so the same views
    at its normal rate leave it close to 1. Products that already existed get the same value in both."""
    class St(_St):
        cap, P = 8, 6
        views_7d = np.array([672.0] * 6 + [0.0, 0.0])   # 1 view / 15 min; slots 6-7 not registered yet
        purch_7d = np.zeros(8)
        sub = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        cat = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        idx = {f"p{i}": i for i in range(8)}
    st = St()
    layer = LiveLayer(st)
    st.P = 7                                      # product 6 registered after the layer was built...
    st.views_7d = st.views_7d.copy()
    st.views_7d[6] = 672.0                        # ...inheriting its neighbours' rate (State._inherit)
    t = st.now_ms
    for step in range(1, 13):                     # one hour, every product viewed at its normal rate
        t += 300_000
        rows = [{"t": t, "p": f"p{i}", "et": "view" if step % 3 == 0 else "add_to_cart"} for i in range(7)]
        layer.events(st, rows)
    assert layer._trend[6] > layer._trend[0] + 0.1          # the bug: a boost only for being new
    assert abs(layer._trend_seed[6] - layer._trend_seed[0]) < 0.05
    assert np.allclose(layer._trend_seed[:6], layer._trend[:6])
    cand = np.arange(7)
    m_seed, _ = layer.multipliers(st, cand, np.random.default_rng(0), t, trend_src="seed")
    m_live, _ = layer.multipliers(st, cand, np.random.default_rng(0), t)
    assert m_seed[6] < m_live[6] and np.allclose(m_seed[:6], m_live[:6])
    assert get_variant("v5-hybrid+m2-intent+m3-live+trend2+seedtrend").trend_src == "seed"
