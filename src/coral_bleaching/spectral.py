import math


def normalized_difference(band_a, band_b):
    if band_a is None or band_b is None:
        return None
    if not all(math.isfinite(v) for v in (band_a, band_b)) or band_a + band_b == 0:
        return None
    return (band_a - band_b) / (band_a + band_b)


def rolling_index_shift(history, window=3, max_gap_days=14):
    """Return (shift, earliest contributing date) from dated valid observations."""
    if window < 1 or max_gap_days < 1:
        raise ValueError("Window and max gap must be positive")
    if not history:
        return None, None
    if any(a[0] >= b[0] for a, b in zip(history, history[1:])):
        raise ValueError("History must contain unique chronological dates")
    recent = history[-2 * window :]
    if len(recent) < 2 * window or any(
        (b[0] - a[0]).days > max_gap_days for a, b in zip(recent, recent[1:])
    ):
        return None, history[-1][0]
    if not all(math.isfinite(v) for _, v in recent):
        raise ValueError("History must contain finite values")
    shift = (
        sum(v for _, v in recent[window:]) / window - sum(v for _, v in recent[:window]) / window
    )
    return shift, recent[0][0]
