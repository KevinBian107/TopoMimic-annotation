"""Auto-directionality: infer Left / Right / Straight segments from qpos.

Independent of the macro-behavior pipeline. Produces a continuous track —
every frame carries exactly one of {Left, Right, Straight}. Segments below
a minimum duration are merged into their neighbor with the larger summed
yaw magnitude, so the direction track "holds for longer durations" instead
of flickering.

Per-frame classification:
    yaw_rate  > +thresh  -> Left
    yaw_rate  < -thresh  -> Right
    otherwise            -> Straight

Here `yaw_rate` is the (signed) instantaneous rate of the unwrapped yaw,
smoothed by averaging over a sliding window of width `window` (same as the
auto-labeler).

Re-uses qpos IO + yaw helpers from `auto_labeler` rather than duplicating.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from auto_labeler import (
    DEFAULT_CONFIG,
    _quat_to_yaw,
    _unwrap,
    load_config,
    read_qpos,
    resolve_qpos_path,  # noqa: F401  (re-exported for symmetry)
)

DIR_MIN_DURATION = 1.0  # seconds — "hold for longer"


def _signed_yaw_rate(yaw_unwrapped: List[float], window: int) -> List[float]:
    """Signed, smoothed yaw rate per frame.

    Rate is computed as the centered difference averaged over a window of
    width `window` centered on the frame.
    """
    n = len(yaw_unwrapped)
    if n < 2:
        return [0.0] * n
    half = max(1, window // 2)
    out: List[float] = []
    for i in range(n):
        s = max(0, i - half)
        e = min(n - 1, i + half)
        if e == s:
            out.append(0.0)
        else:
            out.append((yaw_unwrapped[e] - yaw_unwrapped[s]) / (e - s))
    return out


def _classify_frame(signed_rate: float, thresh: float) -> str:
    if signed_rate > thresh:
        return "Left"
    if signed_rate < -thresh:
        return "Right"
    return "Straight"


def _group_and_smooth(
    labels: List[str], signed_rate: List[float], cfg: Dict[str, float]
) -> List[Tuple[str, int, int]]:
    """Group contiguous same-label frames, then merge short segments.

    A segment shorter than `dir_min_duration` is merged into whichever
    neighbor has the larger summed |yaw_rate| across its span (i.e. into
    the more "decisive" neighbor).
    """
    if not labels:
        return []
    fps = float(cfg.get("fps", 30.0))
    min_dur = float(cfg.get("dir_min_duration", DIR_MIN_DURATION))

    raw: List[Tuple[str, int, int]] = []
    cur = labels[0]
    cur_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            raw.append((cur, cur_start, i))
            cur = labels[i]
            cur_start = i
    raw.append((cur, cur_start, len(labels)))

    def span_magnitude(s: int, e: int) -> float:
        return sum(abs(signed_rate[k]) for k in range(s, e))

    # Iteratively merge the shortest under-threshold segment into its
    # dominant neighbor until all remaining segments are >= min_dur.
    segs = list(raw)
    while True:
        shortest_idx = -1
        shortest_dur = float("inf")
        for i, (_, s, e) in enumerate(segs):
            d = (e - s) / fps
            if d < min_dur and d < shortest_dur:
                shortest_dur = d
                shortest_idx = i
        if shortest_idx < 0:
            break

        i = shortest_idx
        _, s, e = segs[i]
        left_mag = span_magnitude(segs[i - 1][1], segs[i - 1][2]) if i > 0 else -1.0
        right_mag = (
            span_magnitude(segs[i + 1][1], segs[i + 1][2])
            if i + 1 < len(segs)
            else -1.0
        )

        if left_mag < 0 and right_mag < 0:
            break

        if right_mag > left_mag:
            lbl, ls, _ = segs[i + 1]
            segs[i + 1] = (lbl, s, segs[i + 1][2])
            segs.pop(i)
        else:
            lbl, ls, _ = segs[i - 1]
            segs[i - 1] = (lbl, ls, e)
            segs.pop(i)

        # Coalesce newly-adjacent same-label runs
        coalesced: List[Tuple[str, int, int]] = []
        for seg in segs:
            if coalesced and coalesced[-1][0] == seg[0] and coalesced[-1][2] == seg[1]:
                lbl, ps, _ = coalesced[-1]
                coalesced[-1] = (lbl, ps, seg[2])
            else:
                coalesced.append(seg)
        segs = coalesced

    return segs


def auto_direction_clip(
    qpos_path: str, cfg: Optional[Dict[str, float]] = None
) -> List[Dict[str, object]]:
    """Infer direction-track segments for a clip.

    Returns a list of {name, start_time, end_time, start_frame, end_frame}
    covering every frame of the clip.
    """
    if cfg is None:
        cfg = load_config()
    thresh = float(cfg.get("dir_yaw_thresh", DEFAULT_CONFIG["dir_yaw_thresh"]))
    window = int(cfg.get("window", DEFAULT_CONFIG["window"]))
    fps = float(cfg.get("fps", DEFAULT_CONFIG["fps"]))

    rows = read_qpos(qpos_path)
    if not rows:
        return []

    raw_yaw = [_quat_to_yaw(r[3], r[4], r[5], r[6]) for r in rows]
    yaw_unwrapped = _unwrap(raw_yaw)
    signed_rate = _signed_yaw_rate(yaw_unwrapped, window)

    labels = [_classify_frame(r, thresh) for r in signed_rate]
    groups = _group_and_smooth(labels, signed_rate, cfg)

    out: List[Dict[str, object]] = []
    for lbl, s, e in groups:
        out.append(
            {
                "name": lbl,
                "start_time": s / fps,
                "end_time": e / fps,
                "start_frame": s,
                "end_frame": e,
            }
        )
    return out
