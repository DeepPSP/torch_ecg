# -*- coding: utf-8 -*-
"""Remarks: commonly used functions related to intervals.

NOTE
----
1. `Interval` refers to interval of the form ``[a,b]``
2. `GeneralizedInterval` refers to some (finite) union of `Interval`

TODO
----
1. Unify `Interval` and `GeneralizedInterval`, by letting `Interval` be of the form ``[[a,b]]``.
2. Distinguish openness and closedness.

"""

import time
import warnings
from copy import deepcopy
from numbers import Real
from typing import Any, List, Sequence, Tuple, Union

import numpy as np


__all__ = [
    "overlaps",
    "validate_interval",
    "in_interval",
    "in_generalized_interval",
    "intervals_union",
    "generalized_intervals_union",
    "intervals_intersection",
    "generalized_intervals_intersection",
    "generalized_interval_complement",
    "get_optimal_covering",
    "find_max_cont_len",
    "interval_len",
    "generalized_interval_len",
    "find_extrema",
    "is_intersect",
    "max_disjoint_covering",
]


EMPTY_SET = []
Interval = Union[Sequence[Real], type(EMPTY_SET)]
GeneralizedInterval = Union[Sequence[Interval], type(EMPTY_SET)]


def overlaps(interval: Interval, another: Interval) -> int:
    """Find the overlap between two intervals.

    The amount of overlap, in bp between interval and anohter, is returned.

        - If > 0, the number of bp of overlap
        - If 0,  they are book-ended
        - If < 0, the distance in bp between them

    Parameters
    ----------
    interval, another : Interval
        The two intervals to compute their overlap.

    Returns
    -------
    int
        Overlap length of two intervals;
        if < 0, the distance of two intervals.

    Examples
    --------
    >>> overlaps([1,2], [2,3])
    0
    >>> overlaps([1,2], [3,4])
    -1
    >>> overlaps([1,2], [0,3])
    1

    """
    # in case a or b is not in ascending order
    interval.sort()
    another.sort()
    return min(interval[-1], another[-1]) - max(interval[0], another[0])


def validate_interval(
    interval: Union[Interval, GeneralizedInterval], join_book_endeds: bool = True
) -> Tuple[bool, Union[Interval, GeneralizedInterval]]:
    """Check whether `interval` is an `Interval` or a `GeneralizedInterval`.

    If true, return True, and validated (of the form [a, b] with a <= b) interval,
    return ``False, []`` otherwise.

    NOTE: if `interval` is empty, return ``False, []``.

    Parameters
    ----------
    interval : Interval or GeneralizedInterval
        The interval to be validated.
    join_book_endeds : bool, default True
        If True, two book-ended intervals will be joined into one.

    Returns
    -------
    tuple
        2-tuple consisting of
            - bool: indicating whether `interval` is a valid interval
            - an interval (can be empty)

    Examples
    --------
    >>> validate_interval([1, 2, 3])
    (False, [])
    >>> validate_interval([2, 1])
    (True, [1, 2])
    >>> validate_interval([[1, 4], [4, 8]])
    (True, [[1, 8]])
    >>> validate_interval([[1, 4], [4, 8]], join_book_endeds=False)
    (True, [[1, 4], [4, 8]])
    >>> validate_interval([])
    (False, [])

    """
    if (not isinstance(interval, (list, tuple))) or (len(interval) == 0):
        return False, []
    if isinstance(interval[0], (list, tuple)):
        info = [validate_interval(itv, join_book_endeds) for itv in interval]
        if all([item[0] for item in info]):
            return True, intervals_union(interval, join_book_endeds)
        else:
            return False, []

    if len(interval) == 2:
        return True, [min(interval), max(interval)]
    else:
        return False, []


def in_interval(
    val: Real, interval: Interval, left_closed: bool = True, right_closed: bool = False
) -> bool:
    """Check whether val is inside interval or not.

    Parameters
    ----------
    val: numbers.Real,
        the value to be checked
    interval: Interval,
        the interval to be checked
    left_closed: bool, default True,
        whether the left end of `interval` is closed
    right_closed: bool, default False,
        whether the right end of `interval` is closed

    Returns
    -------
    bool, whether `val` is inside `generalized_interval` or not

    Examples
    --------
    >>> in_interval(-1.3, [0, 2])
    False
    >>> in_interval(1.5, [1, 2])
    True
    >>> in_interval(1, [1, 2])
    True
    >>> in_interval(1, [1, 2], left_closed=False)
    False
    >>> in_interval(2, [1, 2])
    False
    >>> in_interval(2, [1, 2], right_closed=True)
    True

    """
    itv = sorted(interval)
    if left_closed:
        is_in = itv[0] <= val
    else:
        is_in = itv[0] < val
    if right_closed:
        is_in = is_in and (val <= itv[-1])
    else:
        is_in = is_in and (val < itv[-1])
    return is_in


def in_generalized_interval(
    val: Real,
    generalized_interval: GeneralizedInterval,
    left_closed: bool = True,
    right_closed: bool = False,
) -> bool:
    """Check whether val is inside generalized_interval or not.

    Parameters
    ----------
    val : numbers.Real
        The value to be checked whether
        it is inside `generalized_interval` or not.
    generalized_interval : GeneralizedInterval
        The interval to be checked.
    left_closed : bool, default True
        Whether the left end of `generalized_interval` is closed or not.
    right_closed : bool, default False
        Whether the right end of `generalized_interval` is closed or not.

    Returns
    -------
    bool
        Whether `val` is inside `generalized_interval` or not.

    Examples
    --------
    >>> in_generalized_interval(1.5, [[1, 2], [3, 4]])
    True
    >>> in_generalized_interval(2.5, [[1, 3], [2, 4]])
    True
    >>> in_generalized_interval(3.45, [[1, 3], [4, 6.9]])
    False
    >>> in_generalized_interval(0, [[0, 1], [3, 4]])
    True
    >>> in_generalized_interval(0, [[0, 1], [3, 4]], left_closed=False)
    False
    >>> in_generalized_interval(1, [[0, 1], [3, 4]])
    False
    >>> in_generalized_interval(1, [[0, 1], [3, 4]], right_closed=True)
    True

    """
    is_in = False
    for interval in generalized_interval:
        if in_interval(val, interval, left_closed, right_closed):
            is_in = True
            break
    return is_in


def intervals_union(
    interval_list: GeneralizedInterval, join_book_endeds: bool = True
) -> GeneralizedInterval:
    """
    find the union (ordered and non-intersecting) of all the intervals in `interval_list`,
    which is a list of intervals in the form [a,b], where a,b need not be ordered

    Parameters
    ----------
    interval_list: GeneralizedInterval,
        the list of intervals to calculate their union
    join_book_endeds: bool, default True,
        join the book-ended intervals into one (e.g. [[1,2],[2,3]] into [1,3]) or not

    Returns
    -------
    processed: GeneralizedInterval,
        the union of the intervals in `interval_list`

    Examples
    --------
    >>> intervals_union([[1, 2], [3, 4]])
    [[1, 2], [3, 4]]
    >>> intervals_union([[1, 2], [2, 3]])
    [[1, 3]]
    >>> intervals_union([[1, 2], [2, 3]], join_book_endeds=False)
    [[1, 2], [2, 3]]
    >>> intervals_union([[1, 2.1], [1.6, 4], [3.1, 10.9]])
    [[1, 10.9]]

    """
    # list_add = lambda list1, list2: list1+list2
    processed = [item for item in interval_list if len(item) > 0]
    for item in processed:
        item.sort()
    processed.sort(key=lambda i: i[0])
    # end_points = reduce(list_add, processed)
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(processed) == 1:
            return processed
        for idx, interval in enumerate(processed[:-1]):
            this_start, this_end = interval
            next_start, next_end = processed[idx + 1]
            # it is certain that this_start <= next_start
            if this_end < next_start:
                # the case where two consecutive intervals are disjoint
                new_intervals.append([this_start, this_end])
                if idx == len(processed) - 2:
                    new_intervals.append([next_start, next_end])
            elif this_end == next_start:
                # the case where two consecutive intervals are book-ended
                # concatenate if `join_book_endeds` is True,
                # or one interval degenerates (to a single point)
                if (
                    this_start == this_end or next_start == next_end
                ) or join_book_endeds:
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += processed[idx + 2 :]
                    merge_flag = True
                    processed = new_intervals
                    break
                else:
                    new_intervals.append([this_start, this_end])
                    if idx == len(processed) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += processed[idx + 2 :]
                merge_flag = True
                processed = new_intervals
                break
        processed = new_intervals
    return processed


def generalized_intervals_union(
    interval_list: Union[List[GeneralizedInterval], Tuple[GeneralizedInterval]],
    join_book_endeds: bool = True,
) -> GeneralizedInterval:
    """Calculate the union of a list (or tuple) of ``GeneralizedInterval``.

    Parameters
    ----------
    interval_list : list or tuple
        A list (or tuple) of `GeneralizedInterval`.
    join_book_endeds : bool, default True
        Whether join the book-ended intervals into one
        (e.g. [[1,2],[2,3]] into [1,3]) or not

    Returns
    -------
    GeneralizedInterval
        The union of `interval_list`.

    Examples
    --------
    >>> generalized_intervals_union(([[1, 2], [3, 7]], [[40,90], [-30, -10]]))
    [[-30, -10], [1, 2], [3, 7], [40, 90]]
    >>> generalized_intervals_union(([[1, 2], [3, 7]], [[4,9], [-3, 1]]))
    [[-3, 2], [3, 9]]
    >>> generalized_intervals_union(([[1, 2], [3, 7]], [[4,9], [-3, 1]]), join_book_endeds=False)
    [[-3, 1], [1, 2], [3, 9]]

    """
    all_intervals = [itv for gnr_itv in interval_list for itv in gnr_itv]
    iu = intervals_union(interval_list=all_intervals, join_book_endeds=join_book_endeds)
    return iu


def intervals_intersection(
    interval_list: GeneralizedInterval, drop_degenerate: bool = True
) -> Interval:
    """Calculate the intersection of all intervals in `interval_list`.

    Parameters
    ----------
    interval_list : GeneralizedInterval
        The list of intervals to yield intersection.
    drop_degenerate : bool, default True
        Whether or not drop the degenerate intervals,
        i.e. intervals with length 0.

    Returns
    -------
    Interval
        The intersection of all intervals in `interval_list`.

    Examples
    --------
    >>> intervals_intersection([[1, 2], [3, 4]])
    []
    >>> intervals_intersection([[1, 2], [2, 3]])
    []
    >>> intervals_intersection([[1, 2], [2, 3]], drop_degenerate=False)
    [[2, 2]]
    >>> intervals_intersection([[1, 2.1], [1.6, 4], [3.1, 10.9]])
    []
    >>> intervals_intersection([[1, 2.1], [1.6, 4], [0.7, 1.9]])
    [1.6, 1.9]

    """
    if [] in interval_list:
        return []
    for item in interval_list:
        item.sort()
    potential_start = max([item[0] for item in interval_list])
    potential_end = min([item[-1] for item in interval_list])
    if (potential_end > potential_start) or (
        potential_end == potential_start and not drop_degenerate
    ):
        its = [potential_start, potential_end]
    else:
        its = []
    return its


def generalized_intervals_intersection(
    generalized_interval: GeneralizedInterval,
    another_generalized_interval: GeneralizedInterval,
    drop_degenerate: bool = True,
) -> GeneralizedInterval:
    """calculate the intersection of intervals.

    Parameters
    ----------
    generalized_interval, another_generalized_interval : GeneralizedInterval
        The 2 `GeneralizedInterval` to yield intersection.
    drop_degenerate : bool, default True
        Whether or not drop the degenerate intervals,
        i.e. intervals with length 0.

    Returns
    -------
    GeneralizedInterval
        The intersection of `generalized_interval` and `another_generalized_interval`.

    Examples
    --------
    >>> generalized_intervals_intersection([[1, 2], [3, 7]], [[40,90], [-30, -10]])
    []
    >>> generalized_intervals_intersection([[1, 5], [12, 33]], [[4, 9], [-3, 3], [33, 99]])
    [[1, 3], [4, 5]]
    >>> generalized_intervals_intersection([[1, 5], [12, 33]], [[4, 9], [-3, 3], [33, 99]], drop_degenerate=False)
    [[1, 3], [4, 5], [33, 33]]

    """
    this = intervals_union(generalized_interval)
    another = intervals_union(another_generalized_interval)
    # NOTE: from now on, `this`, `another` are in ascending ordering
    # and are disjoint unions of intervals
    its = []
    # TODO: optimize the following process
    cut_idx = 0
    for item in this:
        another = another[cut_idx:]
        intersected_indices = []
        for idx, item_prime in enumerate(another):
            tmp = intervals_intersection(
                [item, item_prime], drop_degenerate=drop_degenerate
            )
            if len(tmp) > 0:
                its.append(tmp)
                intersected_indices.append(idx)
        if len(intersected_indices) > 0:
            cut_idx = intersected_indices[-1]
    return its


def generalized_interval_complement(
    total_interval: Interval, generalized_interval: GeneralizedInterval
) -> GeneralizedInterval:
    """
    calculate the complement of `generalized_interval` in `total_interval`

    Parameters
    ----------
    total_interval : Interval
        The total interval.
    generalized_interval : GeneralizedInterval
        The interval to be complemented.

    Returns
    -------
    cpl : GeneralizedInterval
        The complement of `generalized_interval` in `total_interval`

    TODO: the case `total_interval` is a `GeneralizedInterval`.

    Examples
    --------
    >>> generalized_interval_complement([1, 100], [[5, 33], [40, 50], [60, 140]])
    [[1, 5], [33, 40], [50, 60]]
    >>> generalized_interval_complement([1, 10], [[40, 66], [111, 300]])
    [[1, 10]]
    >>> generalized_interval_complement([150, 200], [[40, 66], [111, 300]])
    []

    """
    rearranged_intervals = intervals_union(generalized_interval)
    total_interval.sort()
    tot_start, tot_end = total_interval[0], total_interval[-1]
    rearranged_intervals = [
        [max(tot_start, item[0]), min(tot_end, item[-1])]
        for item in rearranged_intervals
        if overlaps(item, total_interval) > 0
    ]
    slice_points = [tot_start]
    for item in rearranged_intervals:
        slice_points += item
    slice_points.append(tot_end)
    cpl = []
    for i in range(len(slice_points) // 2):
        if slice_points[2 * i + 1] - slice_points[2 * i] > 0:
            cpl.append([slice_points[2 * i], slice_points[2 * i + 1]])
    return cpl


def get_optimal_covering(
    total_interval: Interval,
    to_cover: List[Union[Real, Interval]],
    min_len: Real,
    split_threshold: Real,
    isolated_point_dist_threshold: Real = 0,
    traceback: bool = False,
    **kwargs: Any,
) -> Union[GeneralizedInterval, Tuple[GeneralizedInterval, list]]:
    """
    compute an optimal covering (disjoint union of intervals) that covers `to_cover` such that
    each interval in the covering is of length at least `min_len`,
    and any two intervals in the covering have distance at least `split_threshold`

    Parameters
    ----------
    total_interval: Interval,
        the total interval that the covering is picked from
    to_cover: list,
        a list of intervals to cover
    min_len: numbers.Real,
        minimun length (positive) of the intervals of the covering
    split_threshold: numbers.Real,
        minumun distance (positive) of intervals of the covering
    isolated_point_dist_threshold: numbers.Real, default 0.0,
        the minimum distance (non-negative) of isolated points in `to_cover`
        to the interval boundaries of the interval containing the point in the covering.
        If one wants the isolated points to be centered in the interval containing the point,
        set `isolated_point_dist_threshold` to be `min_len / 2`
    traceback: bool, default False,
        if True, a list containing the list of indices of the intervals in the original `to_cover`,
        that each interval in the covering covers

    Raises
    ------
    if any of the intervals in `to_cover` exceeds the range of `total_interval`,
    ValueError will be raised

    Returns
    -------
    covering or (covering, ret_traceback),
        covering: GeneralizedInterval,
            the covering that satisfies the given conditions
        ret_traceback: list, optional,
            contains the list of indices of the intervals in the original `to_cover`,
            that each interval in the covering covers.
            If `traceback` is False, this will not be returned

    TODO
    ----
    make positions of isolated points in the final covering as close as possible
    to the center of the interval that contains the point

    Examples
    --------
    >>> total_interval = [0, 100]
    >>> to_cover = [[7,33], 66, [82, 89]]
    >>> get_optimal_covering(total_interval, to_cover, 10, 5)
    [[7, 33], [56, 66], [82, 92]]
    >>> get_optimal_covering(total_interval, to_cover, 10, 5, traceback=True)
    ([[7, 33], [56, 66], [82, 92]], [[0], [1], [2]])
    >>> get_optimal_covering(total_interval, to_cover, 20, 5, traceback=True)
    ([[7, 33], [46, 66], [80, 100]], [[0], [1], [2]])
    >>> get_optimal_covering(total_interval, to_cover, 20, 13, traceback=True)
    ([[7, 33], [46, 66], [80, 100]], [[0], [1], [2]])
    >>> get_optimal_covering(total_interval, to_cover, 20, 14, traceback=True)
    ([[7, 33], [66, 89]], [[0], [1, 2]])
    >>> get_optimal_covering(total_interval, to_cover, 20, 13, isolated_point_dist_threshold=1)
    [[7, 33], [47, 67], [80, 100]]
    >>> get_optimal_covering(total_interval, to_cover, 20, 13, isolated_point_dist_threshold=2)
    [[7, 33], [64, 89]]
    >>> get_optimal_covering(total_interval, to_cover, 30, 3)
    [[3, 33], [36, 66], [70, 100]]
    >>> get_optimal_covering(total_interval, to_cover, 30, 4)
    [[3, 33], [59, 89]]
    >>> get_optimal_covering(total_interval, to_cover, 40, 5)
    [[0, 40], [60, 100]]
    >>> get_optimal_covering(total_interval, to_cover, 1000, 1, traceback=True)
    ([[0, 100]], [[0, 1, 2]])

    """
    assert validate_interval(total_interval)[
        0
    ], "`total_interval` must be a valid interval (a sequence of two real numbers)"
    assert min_len > 0, "`min_len` must be positive"
    assert split_threshold > 0, "`split_threshold` must be positive"
    assert (
        isolated_point_dist_threshold >= 0
    ), "`isolated_point_dist_threshold` must be non-negative"

    if len(to_cover) == 0:
        return [] if not traceback else ([], [])

    start_time = time.time()
    verbose = kwargs.get("verbose", 0)
    tmp = sorted(total_interval)
    tot_start, tot_end = tmp[0], tmp[-1]

    if (
        tot_start
        > min([item if isinstance(item, Real) else item[0] for item in to_cover])
    ) or (
        tot_end
        < max([item if isinstance(item, Real) else item[-1] for item in to_cover])
    ):
        raise ValueError(
            "some of the elements in `to_cover` exceeds the range of `total_interval`"
        )

    if verbose >= 1:
        print(f"total_interval = {total_interval}, with_length = {tot_end-tot_start}")

    if tot_end - tot_start < min_len:
        covering = [[tot_start, tot_end]]
        ret_traceback = [list(range(len(to_cover)))] if traceback else []
        return (covering, ret_traceback) if traceback else covering
    to_cover_intervals = []
    isolated_points = []
    if isolated_point_dist_threshold > min_len / 2:
        isolated_point_dist_threshold = min_len / 2
        warnings.warn(
            "`isolated_point_dist_threshold` should be smaller than `min_len`/2, "
            f"hence is set to {isolated_point_dist_threshold}",
            RuntimeWarning,
        )
    for item in to_cover:
        if isinstance(item, list):
            to_cover_intervals.append(item.copy())
        else:
            to_cover_intervals.append(
                [
                    max(tot_start, item - isolated_point_dist_threshold),
                    min(tot_end, item + isolated_point_dist_threshold),
                ]
            )
            isolated_points.append(item)
    if traceback:
        replica_for_traceback = deepcopy(to_cover_intervals)

    if verbose >= 2:
        print(
            f"to_cover_intervals after all converted to intervals = {to_cover_intervals}"
        )

    for interval in to_cover_intervals:
        interval.sort()

    to_cover_intervals.sort(key=lambda i: i[0])

    to_cover_intervals = intervals_union(to_cover_intervals)

    if verbose >= 2:
        print(f"to_cover_intervals after sorted = {to_cover_intervals}")

    # if to_cover_intervals[0][0] < tot_start or to_cover_intervals[-1][-1] > tot_end:
    #     raise IndexError("some item in to_cover list exceeds the range of total_interval")
    # these cases now seen normal, and treated as follows:
    for item in to_cover_intervals:
        item[0] = max(item[0], tot_start)
        item[-1] = min(item[-1], tot_end)
    # to_cover_intervals = [item for item in to_cover_intervals if item[-1] > item[0]]

    # ensure that the distance from the first interval to `tot_start` is at least `min_len`
    to_cover_intervals[0][-1] = max(to_cover_intervals[0][-1], tot_start + min_len)
    # ensure that the distance from the last interval to `tot_end` is at least `min_len`
    to_cover_intervals[-1][0] = min(to_cover_intervals[-1][0], tot_end - min_len)

    to_cover_intervals = intervals_union(to_cover_intervals)

    if verbose >= 2:
        print(f"`to_cover_intervals` after two tails adjusted to {to_cover_intervals}")

    # merge intervals whose distances (might be negative) are less than `split_threshold`
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(to_cover_intervals) == 1:
            break
        for idx, item in enumerate(to_cover_intervals[:-1]):
            this_start, this_end = item
            next_start, next_end = to_cover_intervals[idx + 1]
            if next_start - this_end >= split_threshold:
                if (
                    split_threshold == (next_start - next_end) == 0
                    or split_threshold == (this_start - this_end) == 0
                ):
                    # the case where split_threshold ==0 and the degenerate case should be dealth with separately
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += to_cover_intervals[idx + 2 :]
                    merge_flag = True
                    to_cover_intervals = new_intervals
                    break
                else:
                    new_intervals.append([this_start, this_end])
                    if idx == len(to_cover_intervals) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += to_cover_intervals[idx + 2 :]
                merge_flag = True
                to_cover_intervals = new_intervals
                break
    if verbose >= 2:
        print(
            f"`to_cover_intervals` after merging intervals whose gaps < split_threshold are {to_cover_intervals}"
        )

    # currently, distance between any two intervals in `to_cover_intervals` are larger than `split_threshold`
    # but any interval except the head and tail might has length less than `min_len`
    covering = []
    if len(to_cover_intervals) == 1:
        # NOTE: here, there's only one `to_cover_intervals`,
        # whose length should be at least `min_len`
        mid_pt = (to_cover_intervals[0][0] + to_cover_intervals[0][-1]) // 2
        half_len = min_len // 2
        if mid_pt - tot_start < half_len:
            ret_start = tot_start
            ret_end = min(tot_end, max(tot_start + min_len, to_cover_intervals[0][-1]))
            covering = [[ret_start, ret_end]]
        else:
            ret_start = max(tot_start, min(to_cover_intervals[0][0], mid_pt - half_len))
            ret_end = min(
                tot_end, max(mid_pt - half_len + min_len, to_cover_intervals[0][-1])
            )
            covering = [[ret_start, ret_end]]

    start = min(to_cover_intervals[0][0], to_cover_intervals[0][-1] - min_len)
    start = [start, start]

    for idx, item in enumerate(to_cover_intervals[:-1]):
        this_start, this_end = item
        next_start, next_end = to_cover_intervals[idx + 1]
        potential_end = max(this_end, start[0] + min_len)
        # if distance from `potential_end` to `next_start` is not enough
        # and has not reached the end of `to_cover_intervals`
        # continue to the next loop
        if next_start - potential_end < split_threshold:
            if idx < len(to_cover_intervals) - 2:
                continue
            else:
                # now, idx equals len(to_cover_intervals)-2
                # distance from `next_start` (hence `start`) to `tot_end` is at least `min_len`
                potential_end = max(start[0] + min_len, next_end)
                covering.append(
                    [
                        max(start[0], min(start[1], potential_end - min_len)),
                        potential_end,
                    ]
                )
        else:
            covering.append(
                [max(start[0], min(start[1], potential_end - min_len)), potential_end]
            )
            start = [
                max(
                    potential_end + split_threshold, min(next_start, next_end - min_len)
                ),
                next_start,
            ]
            if idx == len(to_cover_intervals) - 2:
                covering.append([next_start, max(next_start + min_len, next_end)])

    ret_traceback = []
    if traceback:
        for item in covering:
            record = []
            for idx, item_prime in enumerate(replica_for_traceback):
                itc = intervals_intersection([item, item_prime], drop_degenerate=False)
                len_itc = itc[-1] - itc[0] if len(itc) > 0 else -1
                if len_itc > 0 or (
                    len_itc == 0 and item_prime[-1] - item_prime[0] == 0
                ):
                    record.append(idx)
            ret_traceback.append(record)

    if verbose >= 1:
        print(
            f"the final result of `get_optimal_covering` is {covering}, "
            f"ret_traceback = {ret_traceback}, "
            f"the whole process used {time.time()-start_time} second(s)."
        )

    if traceback:
        return covering, ret_traceback
    return covering


def find_max_cont_len(sublist: Interval, tot_rng: Real) -> dict:
    """
    find the maximum length of continuous (consecutive) sublists of `sublist`,
    whose element are integers within the range from 0 to `tot_rng`,
    along with the position of this sublist and the sublist itself.

    Parameters
    ----------
    sublist: Interval,
        a sublist
    tot_rng: numbers.Real,
        the total range

    Returns
    -------
    ret: dict, with items
        - "max_cont_len"
        - "max_cont_sublist_start"
        - "max_cont_sublist"

    Examples
    --------
    >>> tot_rng = 10
    >>> sublist = [0, 2, 3, 4, 7, 9]
    >>> find_max_cont_len(sublist, tot_rng)
    {'max_cont_len': 3, 'max_cont_sublist_start': 1, 'max_cont_sublist': [2, 3, 4]}

    """
    complementary_sublist = (
        [-1] + [i for i in range(tot_rng) if i not in sublist] + [tot_rng]
    )
    diff_list = np.diff(np.array(complementary_sublist))
    max_cont_len = np.max(diff_list) - 1
    max_cont_sublist_start = np.argmax(diff_list)
    max_cont_sublist = sublist[
        max_cont_sublist_start : max_cont_sublist_start + max_cont_len
    ]
    ret = {
        "max_cont_len": max_cont_len,
        "max_cont_sublist_start": max_cont_sublist_start,
        "max_cont_sublist": max_cont_sublist,
    }
    return ret


def interval_len(interval: Interval) -> Real:
    """
    compute the length of an interval. 0 for the empty interval []

    Parameters
    ----------
    interval: Interval

    Returns
    -------
    itv_len: numbers.Real,
        the `length` of `interval`

    Examples
    --------
    >>> interval_len([0, 10])
    10
    >>> interval_len([10, 10])
    0
    >>> interval_len([10, 0])
    10
    >>> interval_len([])
    0

    """
    interval.sort()
    itv_len = interval[-1] - interval[0] if len(interval) > 0 else 0
    return itv_len


def generalized_interval_len(generalized_interval: GeneralizedInterval) -> Real:
    """
    compute the length of a generalized interval. 0 for the empty interval []

    Parameters
    ----------
    generalized_interval: GeneralizedInterval

    Returns
    -------
    gi_len: numbers.Real,
        the `length` of `generalized_interval`

    Examples
    --------
    >>> generalized_interval_len([[0, 10], [20, 30]])
    20
    >>> generalized_interval_len([[10, 10], [20, 30]])
    10
    >>> generalized_interval_len([[10, 0], [20, 30]])
    20
    >>> generalized_interval_len([[0, 20], [10, 30]])
    30
    >>> generalized_interval_len([])
    0

    """
    gi_len = sum([interval_len(item) for item in intervals_union(generalized_interval)])
    return gi_len


def find_extrema(signal: Union[np.ndarray, Sequence], mode: str = "both") -> np.ndarray:
    """
    Locate local extrema points in a 1D signal. Based on Fermat's Theorem

    Parameters
    ----------
    signal: ndarray,
        1D input signal.
    mode: str, default "both",
        whether to find maxima ("max"), minima ("min"), or both ("both"),
        case insensitive.

    Returns
    -------
    extrema : ndarray
        indices of the extrama points.

    Examples
    --------
    >>> x = np.linspace(0, 2 * np.pi, 100)
    >>> y = np.sin(x)
    >>> find_extrema(y, mode="max")
    array([25])
    >>> find_extrema(y, mode="min")
    array([74])
    >>> find_extrema(y, mode="both")
    array([25, 74])

    """
    # check inputs
    if np.ndim(signal) != 1:
        raise ValueError(f"`signal` must be 1D, but got {np.ndim(signal)}D")

    mode = mode.lower()
    if mode not in ["max", "min", "both"]:
        raise ValueError(f"Unknwon `{mode}`, must be one of `max`, `min`, `both`")

    aux = np.diff(np.sign(np.diff(signal)))

    if mode == "both":
        aux = np.abs(aux)
        extrema = np.nonzero(aux > 0)[0] + 1
    elif mode == "max":
        extrema = np.nonzero(aux < 0)[0] + 1
    elif mode == "min":
        extrema = np.nonzero(aux > 0)[0] + 1

    return extrema


def is_intersect(
    interval: Union[GeneralizedInterval, Interval],
    another_interval: Union[GeneralizedInterval, Interval],
) -> bool:
    """
    determines if two (generalized) intervals intersect or not

    Parameters
    ----------
    interval, another_interval: GeneralizedInterval or Interval

    Returns
    -------
    bool,
        True if `interval` intersects with another_interval, False otherwise

    Examples
    --------
    >>> is_intersect([0, 10], [5, 15])
    True
    >>> is_intersect([0, 10], [10, 15])
    False
    >>> is_intersect([0, 10], [])
    False
    >>> is_intersect([0, 10], [[5, 20], [25, 30]])
    True

    """
    if (
        interval is None
        or another_interval is None
        or len(interval) * len(another_interval) == 0
    ):
        # the case of empty set
        return False

    # check if is GeneralizedInterval
    is_generalized = isinstance(interval[0], (list, tuple))
    is_another_generalized = isinstance(another_interval[0], (list, tuple))

    if is_generalized and is_another_generalized:
        return any([is_intersect(interval, itv) for itv in another_interval])
    elif not is_generalized and is_another_generalized:
        return is_intersect(another_interval, interval)
    elif is_generalized:  # and not is_another_generalized
        return any([is_intersect(itv, another_interval) for itv in interval])
    else:  # not is_generalized and not is_another_generalized
        return any([overlaps(interval, another_interval) > 0])


def max_disjoint_covering(
    intervals: GeneralizedInterval,
    allow_book_endeds: bool = True,
    traceback: bool = True,
    verbose: int = 0,
) -> Tuple[GeneralizedInterval, List[int]]:
    """Find the largest (the largest interval length) covering
    of a sequence of intervals.

    Parameters
    ----------
    intervals : GeneralizedInterval
        A sequence of intervals.
    allow_book_endeds : bool, default True
        If True, book-ended intervals will be considered valid (disjoint).
    traceback : bool, default True
        If True, the indices of the intervals in the input `intervals`
        of the output covering will also be returned.

    Returns
    -------
    covering : GeneralizedInterval
        The maximum non-overlapping (disjoint) subset of `intervals`.
    covering_inds : List[int]
        Indices in `intervals` of the intervals of `covering_inds`.

    Examples
    --------
    >>> max_disjoint_covering([])
    ([], [])
    >>> max_disjoint_covering([[0, 10]])
    ([[0, 10]], [0])
    >>> max_disjoint_covering([[1, 4], [2, 3], [4, 6], [8, 9]])
    ([[1, 4], [4, 6], [8, 9]], [0, 2, 3])
    >>> max_disjoint_covering([[1, 4], [2, 3], [4, 6], [8, 9]], allow_book_endeds=False, traceback=False)
    ([[2, 3], [4, 6], [8, 9]], [])

    NOTE
    ----
    1. The problem seems slightly different from the problem discussed in reference [1]_ and [2]_.
    2. Intervals with non-positive length will be ignored

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Maximum_disjoint_set
    .. [2] https://www.geeksforgeeks.org/maximal-disjoint-intervals/

    """
    if len(intervals) <= 1:
        covering = deepcopy(intervals)
        return covering, list(range(len(covering)))

    l_itv = [sorted(itv) for itv in intervals]
    ordering = np.argsort([itv[-1] for itv in l_itv])
    l_itv = [l_itv[idx] for idx in ordering]

    if verbose >= 1:
        print(
            f"the sorted intervals are {l_itv}, "
            f"whose indices in the original input `intervals` are {ordering}"
        )

    if allow_book_endeds:
        candidates_inds = [
            [idx] for idx, itv in enumerate(l_itv) if overlaps(itv, l_itv[0]) > 0
        ]
    else:
        candidates_inds = [
            [idx] for idx, itv in enumerate(l_itv) if overlaps(itv, l_itv[0]) >= 0
        ]
    candidates = [[l_itv[inds[0]]] for inds in candidates_inds]

    if verbose >= 1:
        print(
            f"candidates heads = {candidates}, with corresponding indices "
            f"in the sorted list of input intervals = {candidates_inds}"
        )

    for c_idx, (cl, ci) in enumerate(zip(candidates, candidates_inds)):
        if interval_len(cl[0]) == 0:
            continue
        if allow_book_endeds:
            tmp_inds = [
                idx
                for idx, itv in enumerate(l_itv)
                if itv[0] >= cl[0][-1] and interval_len(itv) > 0
            ]
        else:
            tmp_inds = [
                idx
                for idx, itv in enumerate(l_itv)
                if itv[0] > cl[0][-1] and interval_len(itv) > 0
            ]
        if verbose >= 2:
            print(f"for the {c_idx}-th candidate, tmp_inds = {tmp_inds}")
        if len(tmp_inds) > 0:
            tmp = [l_itv[idx] for idx in tmp_inds]
            tmp_candidates, tmp_candidates_inds = max_disjoint_covering(
                intervals=tmp,
                allow_book_endeds=allow_book_endeds,
                traceback=traceback,
                # verbose=verbose,
            )
            candidates[c_idx] = cl + tmp_candidates
            candidates_inds[c_idx] = ci + [tmp_inds[i] for i in tmp_candidates_inds]

    if verbose >= 1:
        print(
            f"the processed candidates are {candidates}, with corresponding indices "
            f"in the sorted list of input intervals = {candidates_inds}"
        )

    # covering = max(candidates, key=generalized_interval_len)
    max_idx = np.argmax([generalized_interval_len(c) for c in candidates])
    covering = candidates[max_idx]
    if traceback:
        covering_inds = candidates_inds[max_idx]
        covering_inds = [
            ordering[i] for i in covering_inds
        ]  # map to the original indices
    else:
        covering_inds = []
    return covering, covering_inds
