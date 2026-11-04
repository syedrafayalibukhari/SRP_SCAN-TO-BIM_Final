"""Corner clean-up of linear elements.

Elements are dicts with axis d, normal n, interval [t0, t1] along d, offset o along n, thickness T and kind.
  1. Junctions between non-parallel elements, closest first, each element end used once:
     L (the centre lines cross near an end of both): the higher-priority element runs to the outer face,
       the other stops at its inner face, which gives a butt corner without gap or overshoot.
     T (near an end of one, inside the other): the end moves to the near face of the through element.
  2. Collinear gaps: two ends of same-kind, collinear elements closer than GAP_COLLINEAR meet at the midpoint.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

TOLERANCE = 0.35        # m, largest extension or trim of an end
GAP_COLLINEAR = 0.15    # m, collinear gaps closed
MIN_ANGLE = 30.0        # deg, junction only between elements at least this far from parallel


def centre_line_crossing(a: dict, b: dict) -> Tuple[float, float, float]:
    """Crossing of the centre lines of a and b: position along a, position along b, sine of the angle."""
    sine = a["d"][0] * b["d"][1] - a["d"][1] * b["d"][0]
    pa, pb = a["n"] * a["o"], b["n"] * b["o"]
    ta = ((pb - pa)[0] * b["d"][1] - (pb - pa)[1] * b["d"][0]) / sine
    crossing = pa + ta * a["d"]
    return float(crossing @ a["d"]), float(crossing @ b["d"]), abs(sine)


def find_candidates(elements: List[dict], index: List[int]) -> List[tuple]:
    candidates = []
    for position, i in enumerate(index):
        a = elements[i]
        for j in index[position + 1:]:
            b = elements[j]
            if abs(a["d"] @ b["d"]) > np.cos(np.radians(MIN_ANGLE)):
                continue
            ta, tb, sine = centre_line_crossing(a, b)
            end_a = min((abs(ta - a["t0"]), 0), (abs(ta - a["t1"]), 1))
            end_b = min((abs(tb - b["t0"]), 0), (abs(tb - b["t1"]), 1))
            inside_a = a["t0"] + TOLERANCE < ta < a["t1"] - TOLERANCE
            inside_b = b["t0"] + TOLERANCE < tb < b["t1"] - TOLERANCE
            if end_a[0] <= TOLERANCE and end_b[0] <= TOLERANCE:
                candidates.append((end_a[0] + end_b[0], "L", i, end_a[1], j, end_b[1], ta, tb, sine))
            elif end_a[0] <= TOLERANCE and inside_b:
                candidates.append((end_a[0], "T", i, end_a[1], j, None, ta, tb, sine))
            elif end_b[0] <= TOLERANCE and inside_a:
                candidates.append((end_b[0], "T", j, end_b[1], i, None, tb, ta, sine))
    return candidates


def close_collinear_gaps(elements: List[dict], index: List[int], used: set) -> int:
    closed = 0
    for position, i in enumerate(index):
        a = elements[i]
        for j in index[position + 1:]:
            b = elements[j]
            if a["kind"] != b["kind"] or abs(a["d"] @ b["d"]) < np.cos(np.radians(2)):
                continue
            sign = 1 if a["d"] @ b["d"] > 0 else -1
            if abs(a["o"] - sign * b["o"]) > 0.03:
                continue
            span = sorted((sign * b["t0"], sign * b["t1"]))
            for end_a, end_b in ((1, 0), (0, 1)):
                ta, tb = (a["t1"] if end_a else a["t0"]), span[end_b]
                gap = (tb - ta) if end_a else (ta - tb)
                if 0 < gap <= GAP_COLLINEAR and (i, end_a) not in used:
                    middle = (ta + tb) / 2
                    a["t1" if end_a else "t0"] = middle
                    if sign > 0:
                        b["t0" if end_b == 0 else "t1"] = middle
                    else:
                        b["t1" if end_b == 0 else "t0"] = -middle
                    closed += 1
    return closed


def join(elements: List[dict], priority: Dict[str, int], log: Callable[[str], None]) -> None:
    index = [i for i, x in enumerate(elements) if x["kind"] != "column"]
    used = set()
    corners = tees = 0
    for _, kind, i, end_i, j, end_j, ti, tj, sine in sorted(find_candidates(elements, index), key=lambda c: c[0]):
        if (i, end_i) in used or (kind == "L" and (j, end_j) in used):
            continue
        a, b = elements[i], elements[j]
        side_a = 1 if end_i == 1 else -1
        if kind == "T":
            new = ti - side_a * (b["T"] / 2) / sine             # stop at the near face of b
            length = a["t1"] - a["t0"]
            if abs(new - (a["t1"] if end_i else a["t0"])) > TOLERANCE or length < 2 * TOLERANCE and abs(new - ti) > length:
                continue
            a["t1" if end_i else "t0"] = new
            used.add((i, end_i))
            tees += 1
        else:
            side_b = 1 if end_j == 1 else -1
            a_leads = priority[a["kind"]] < priority[b["kind"]] or (
                priority[a["kind"]] == priority[b["kind"]] and a["t1"] - a["t0"] >= b["t1"] - b["t0"])
            lead, side_l, t_l, end_l = (a, side_a, ti, end_i) if a_leads else (b, side_b, tj, end_j)
            rest, side_r, t_r, end_r = (b, side_b, tj, end_j) if a_leads else (a, side_a, ti, end_i)
            lead["t1" if end_l else "t0"] = t_l + side_l * (rest["T"] / 2) / sine    # to the outer face of the other
            rest["t1" if end_r else "t0"] = t_r - side_r * (lead["T"] / 2) / sine    # to the inner face of the lead
            used |= {(i, end_i), (j, end_j)}
            corners += 1
    closed = close_collinear_gaps(elements, index, used)
    log(f"junctions: {corners} L-corners, {tees} T-junctions, {closed} collinear gaps closed (tolerance {TOLERANCE} m)")


def remove_stubs(elements: List[dict], log: Callable[[str], None], touch: float = 0.003) -> List[dict]:
    """Delete overshoot remnants: a piece shorter than TOLERANCE, split off the same source element, with
    one end face touching another element and the other end free."""
    outline = [Polygon([x["d"] * t + x["n"] * (x["o"] + side * x["T"] / 2) for t, side in
                        ((x["t0"], -1), (x["t1"], -1), (x["t1"], 1), (x["t0"], 1))]) for x in elements]

    def end_face(x: dict, t: float) -> LineString:
        return LineString([x["d"] * t + x["n"] * (x["o"] - x["T"] / 2), x["d"] * t + x["n"] * (x["o"] + x["T"] / 2)])

    pieces: Dict[tuple, int] = {}
    for x in elements:
        pieces[x["src"]] = pieces.get(x["src"], 0) + 1
    dropped = set()
    for i, x in enumerate(elements):
        if x["kind"] == "column" or x["t1"] - x["t0"] >= TOLERANCE or pieces[x["src"]] < 2:
            continue
        touching = [any(outline[j].distance(end_face(x, t)) < touch for j in range(len(elements)) if j != i and j not in dropped)
                    for t in (x["t0"], x["t1"])]
        if touching[0] != touching[1]:
            dropped.add(i)
    log(f"overshoot stubs removed: {len(dropped)}")
    return [x for i, x in enumerate(elements) if i not in dropped]
