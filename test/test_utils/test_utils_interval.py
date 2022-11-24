"""
"""

import numpy as np
import pytest

from torch_ecg.utils.utils_interval import (
    get_optimal_covering,
    overlaps,
    validate_interval,
    in_interval,
    in_generalized_interval,
    intervals_union,
    generalized_intervals_union,
    intervals_intersection,
    generalized_intervals_intersection,
    generalized_interval_complement,
    find_max_cont_len,
    interval_len,
    generalized_interval_len,
    find_extrema,
    is_intersect,
    max_disjoint_covering,
)


def test_overlaps():
    assert overlaps([1, 2], [2, 3]) == 0
    assert overlaps([1, 2], [3, 4]) == -1
    assert overlaps([1, 2], [0, 3]) == 1


def test_validate_interval():
    assert validate_interval([1, 2, 3]) == (False, [])
    assert validate_interval([2, 1]) == (True, [1, 2])
    assert validate_interval([[1, 4], [4, 8]]) == (True, [[1, 8]])
    assert validate_interval([[1, 4], [4, 8]], join_book_endeds=False) == (
        True,
        [[1, 4], [4, 8]],
    )
    assert validate_interval([]) == (False, [])


def test_in_interval():
    assert in_interval(1.5, [1, 2]) is True
    assert in_interval(-1.3, [0, 2]) is False
    assert in_interval(10, [0, 2]) is False
    assert in_interval(1, [1, 2]) is True
    assert in_interval(1, [1, 2], left_closed=False) is False
    assert in_interval(2, [1, 2]) is False
    assert in_interval(2, [1, 2], right_closed=True) is True


def test_in_generalized_interval():
    assert in_generalized_interval(1.5, [[1, 2], [3, 4]]) is True
    assert in_generalized_interval(2.5, [[1, 3], [2, 4]]) is True
    assert in_generalized_interval(3.45, [[1, 3], [4, 6.9]]) is False
    assert in_generalized_interval(0, [[0, 1], [3, 4]]) is True
    assert in_generalized_interval(0, [[0, 1], [3, 4]], left_closed=False) is False
    assert in_generalized_interval(1, [[0, 1], [3, 4]]) is False
    assert in_generalized_interval(1, [[0, 1], [3, 4]], right_closed=True) is True


def test_intervals_union():
    assert intervals_union([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    assert intervals_union([[1, 2], [2, 3]]) == [[1, 3]]
    assert intervals_union([[1, 2], [2, 3]], join_book_endeds=False) == [[1, 2], [2, 3]]
    assert intervals_union([[1, 2.1], [1.6, 4], [3.1, 10.9]]) == [[1, 10.9]]


def test_generalized_intervals_union():
    assert generalized_intervals_union(([[1, 2], [3, 7]], [[40, 90], [-30, -10]])) == [
        [-30, -10],
        [1, 2],
        [3, 7],
        [40, 90],
    ]
    assert generalized_intervals_union(([[1, 2], [3, 7]], [[4, 9], [-3, 1]])) == [
        [-3, 2],
        [3, 9],
    ]
    assert generalized_intervals_union(
        ([[1, 2], [3, 7]], [[4, 9], [-3, 1]]), join_book_endeds=False
    ) == [[-3, 1], [1, 2], [3, 9]]


def test_intervals_intersection():
    assert intervals_intersection([[1, 2], [3, 4]]) == []
    assert intervals_intersection([[1, 2], [2, 3]]) == []
    assert intervals_intersection([[1, 2], [2, 3]], drop_degenerate=False) == [2, 2]
    assert intervals_intersection([[1, 2.1], [1.6, 4], [3.1, 10.9]]) == []
    assert intervals_intersection([[1, 2.1], [1.6, 4], [0.7, 1.9]]) == [1.6, 1.9]


def test_generalized_intervals_intersection():
    assert (
        generalized_intervals_intersection([[1, 2], [3, 7]], [[40, 90], [-30, -10]])
        == []
    )
    assert generalized_intervals_intersection(
        [[1, 5], [12, 33]], [[4, 9], [-3, 3], [33, 99]]
    ) == [[1, 3], [4, 5]]
    assert generalized_intervals_intersection(
        [[1, 5], [12, 33]], [[4, 9], [-3, 3], [33, 99]], drop_degenerate=False
    ) == [[1, 3], [4, 5], [33, 33]]


def test_generalized_interval_complement():
    assert generalized_interval_complement(
        [1, 100], [[5, 33], [40, 50], [60, 140]]
    ) == [[1, 5], [33, 40], [50, 60]]
    assert generalized_interval_complement([1, 10], [[40, 66], [111, 300]]) == [[1, 10]]
    assert generalized_interval_complement([150, 200], [[40, 66], [111, 300]]) == []


def test_get_optimal_covering():
    total_interval = [0, 100]
    to_cover = [[7, 33], 66, [82, 89]]
    covering_1 = get_optimal_covering(
        total_interval, to_cover, min_len=10, split_threshold=5, verbose=3
    )
    assert len(covering_1) == 3
    covering, traceback = get_optimal_covering(
        total_interval, to_cover, min_len=10, split_threshold=5, traceback=True
    )
    assert covering_1 == covering
    assert traceback == [[0], [1], [2]]
    covering, traceback = get_optimal_covering(
        total_interval, to_cover, min_len=20, split_threshold=5, traceback=True
    )
    assert len(covering) == 3
    assert traceback == [[0], [1], [2]]
    covering, traceback = get_optimal_covering(
        total_interval, to_cover, min_len=20, split_threshold=13, traceback=True
    )
    assert len(covering) == 3
    assert traceback == [[0], [1], [2]]
    covering, traceback = get_optimal_covering(
        total_interval, to_cover, min_len=20, split_threshold=14, traceback=True
    )
    assert len(covering) == 2
    assert traceback == [[0], [1, 2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=20,
        split_threshold=13,
        isolated_point_dist_threshold=1,
        traceback=True,
    )
    assert len(covering) == 3
    assert traceback == [[0], [1], [2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=20,
        split_threshold=13,
        isolated_point_dist_threshold=2,
        traceback=True,
    )
    assert len(covering) == 2
    assert traceback == [[0], [1, 2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=30,
        split_threshold=3,
        traceback=True,
    )
    assert len(covering) == 3
    assert traceback == [[0], [1], [2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=30,
        split_threshold=4,
        traceback=True,
    )
    assert len(covering) == 2
    assert traceback == [[0], [1, 2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=40,
        split_threshold=20,
        traceback=True,
    )
    assert len(covering) == 2
    assert traceback == [[0], [1, 2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=40,
        split_threshold=21,
        traceback=True,
    )
    assert len(covering) == 1
    assert traceback == [[0, 1, 2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=50,
        split_threshold=1,
        traceback=True,
    )
    assert len(covering) == 1
    assert traceback == [[0, 1, 2]]
    covering, traceback = get_optimal_covering(
        total_interval,
        to_cover,
        min_len=1000,
        split_threshold=1,
        traceback=True,
    )
    assert len(covering) == 1
    assert traceback == [[0, 1, 2]]

    with pytest.raises(
        ValueError,
        match="some of the elements in `to_cover` exceeds the range of `total_interval`",
    ):
        get_optimal_covering([10, 100], to_cover, min_len=10, split_threshold=5)
    with pytest.raises(
        ValueError,
        match="some of the elements in `to_cover` exceeds the range of `total_interval`",
    ):
        get_optimal_covering([0, 70], to_cover, min_len=10, split_threshold=5)
    with pytest.raises(
        AssertionError,
        match="`total_interval` must be a valid interval \\(a sequence of two real numbers\\)",
    ):
        get_optimal_covering([0, 100, 200], to_cover, min_len=10, split_threshold=5)
    with pytest.raises(AssertionError, match="`to_cover` must be non-empty"):
        get_optimal_covering(total_interval, [], min_len=10, split_threshold=5)
    with pytest.raises(
        AssertionError,
        match="`total_interval` must be a valid interval \\(a sequence of two real numbers\\)",
    ):
        get_optimal_covering([0], to_cover, min_len=10, split_threshold=5)
    with pytest.raises(
        AssertionError,
        match="`total_interval` must be a valid interval \\(a sequence of two real numbers\\)",
    ):
        get_optimal_covering(0, to_cover, min_len=10, split_threshold=5)
    with pytest.raises(AssertionError, match="`min_len` must be positive"):
        get_optimal_covering(total_interval, to_cover, min_len=0, split_threshold=5)
    with pytest.raises(AssertionError, match="`split_threshold` must be positive"):
        get_optimal_covering(total_interval, to_cover, min_len=10, split_threshold=0)
    with pytest.raises(
        AssertionError, match="`isolated_point_dist_threshold` must be non-negative"
    ):
        get_optimal_covering(
            total_interval,
            to_cover,
            min_len=10,
            split_threshold=5,
            isolated_point_dist_threshold=-1,
        )
    with pytest.warns(
        RuntimeWarning,
        match="isolated_point_dist_threshold` should be smaller than `min_len`/2",
    ):
        get_optimal_covering(
            total_interval,
            to_cover,
            min_len=10,
            split_threshold=5,
            isolated_point_dist_threshold=6,
        )


def test_find_max_cont_len():
    tot_rng = 10
    sublist = [0, 2, 3, 4, 7, 9]
    assert find_max_cont_len(sublist, tot_rng) == {
        "max_cont_len": 3,
        "max_cont_sublist_start": 1,
        "max_cont_sublist": [2, 3, 4],
    }


def test_interval_len():
    assert interval_len([0, 10]) == 10
    assert interval_len([10, 10]) == 0
    assert interval_len([10, 0]) == 10
    assert interval_len([]) == 0


def test_generalized_interval_len():
    assert generalized_interval_len([[0, 10], [20, 30]]) == 20
    assert generalized_interval_len([[10, 10], [20, 30]]) == 10
    assert generalized_interval_len([[10, 0], [20, 30]]) == 20
    assert generalized_interval_len([[0, 20], [10, 30]]) == 30
    assert generalized_interval_len([]) == 0


def test_is_intersect():
    assert is_intersect([0, 10], [5, 15]) is True
    assert is_intersect([0, 10], [10, 15]) is False
    assert is_intersect([0, 10], []) is False
    assert is_intersect([0, 10], [[5, 20], [25, 30]]) is True


def test_find_extrema():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    assert find_extrema(y, mode="max").tolist() == [25]
    assert find_extrema(y, mode="min").tolist() == [74]
    assert find_extrema(y, mode="both").tolist() == [25, 74]

    with pytest.raises(ValueError, match="`signal` must be 1D, but got 2D"):
        find_extrema(y.reshape(1, -1), mode="max")
    with pytest.raises(
        ValueError, match="Unknwon `invalid`, must be one of `max`, `min`, `both`"
    ):
        find_extrema(y, mode="invalid")


def test_max_disjoint_covering():
    assert max_disjoint_covering([]) == ([], [])
    assert max_disjoint_covering([[0, 10]]) == ([[0, 10]], [0])
    assert max_disjoint_covering([[1, 4], [2, 3], [4, 6], [8, 9]], verbose=2) == (
        [[1, 4], [4, 6], [8, 9]],
        [0, 2, 3],
    )
    assert max_disjoint_covering(
        [[1, 4], [2, 3], [4, 6], [8, 9]], allow_book_endeds=False, traceback=False
    ) == ([[2, 3], [4, 6], [8, 9]], [])
