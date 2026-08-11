import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from .mlx_progressive_tome_ab import (
    TOME_SCHEDULE,
    TOME_SCHEDULES,
    _cell_layout,
    tome_bipartite_merge,
    tome_bipartite_merge_numpy,
)


def _inputs(cells=2, width=4):
    values = np.arange(cells * 9 * width, dtype=np.float32).reshape(1, cells, 9, width) + 1
    sizes = np.ones((1, cells, 9), dtype=np.float32)
    positions = np.stack(np.indices((cells, 9)), axis=-1)[None].astype(np.int32)
    return values, sizes, positions


def test_schedule_finishes_after_the_sixteenth_block():
    assert TOME_SCHEDULE[-1] == (16, 1)
    assert all(schedule[-1] == (16, 1) for schedule in TOME_SCHEDULES.values())


def test_final_block_schedule_only_reduces_block_sixteen_input():
    assert TOME_SCHEDULES["final-block"] == ((15, 6), (16, 3), (16, 1))


@pytest.mark.parametrize("target", (6, 3, 1))
def test_numpy_merge_shape_and_finite(target):
    hidden, sizes, positions = _inputs()
    current = 9
    for next_target in (6, 3, 1):
        hidden, sizes, positions = tome_bipartite_merge_numpy(hidden, sizes, positions, next_target)
        current = next_target
        assert hidden.shape == (1, 2, current, 4)
        assert sizes.shape == (1, 2, current)
        assert positions.shape == (1, 2, current, 2)
        assert np.isfinite(hidden).all()
        if next_target == target:
            break


def test_progressive_weighted_conservation_and_final_mean():
    hidden, sizes, positions = _inputs(cells=3, width=5)
    original_mass = (hidden * sizes[..., None]).sum(axis=2)
    original_sizes = sizes.sum(axis=2)
    for target in (6, 3, 1):
        hidden, sizes, positions = tome_bipartite_merge_numpy(hidden, sizes, positions, target)
        np.testing.assert_allclose((hidden * sizes[..., None]).sum(axis=2), original_mass, rtol=1e-6)
        np.testing.assert_allclose(sizes.sum(axis=2), original_sizes)
    np.testing.assert_allclose(hidden[:, :, 0], original_mass / original_sizes[..., None], rtol=1e-6)
    np.testing.assert_array_equal(sizes[:, :, 0], 9)


def test_merge_is_cell_local_and_retains_destination_coordinates():
    hidden, sizes, positions = _inputs(cells=2)
    hidden[:, 1] += 10_000
    # Exercise the required schedule rather than an unsupported direct 9->1 merge.
    for target in (6, 3, 1):
        if target == 6:
            merged, merged_sizes, merged_positions = tome_bipartite_merge_numpy(hidden, sizes, positions, target)
        else:
            merged, merged_sizes, merged_positions = tome_bipartite_merge_numpy(merged, merged_sizes, merged_positions, target)
    assert merged[0, 0, 0].max() < 1_000
    assert merged[0, 1, 0].min() > 9_000
    assert merged_positions[0, 0, 0, 0] == 0
    assert merged_positions[0, 1, 0, 0] == 1


def test_cell_layout_is_row_major_without_crossing_cells():
    grid = mx.array(np.arange(36).reshape(1, 36, 1))
    cells = np.array(_cell_layout(grid, 6, 6), copy=True).reshape(4, 9)
    np.testing.assert_array_equal(cells[0], [0, 1, 2, 6, 7, 8, 12, 13, 14])
    np.testing.assert_array_equal(cells[1], [3, 4, 5, 9, 10, 11, 15, 16, 17])
    np.testing.assert_array_equal(cells[2], [18, 19, 20, 24, 25, 26, 30, 31, 32])


def test_mlx_matches_numpy_and_stays_finite():
    hidden, sizes, positions = _inputs(cells=2, width=8)
    expected = tome_bipartite_merge_numpy(hidden, sizes, positions, 6)
    actual = tome_bipartite_merge(mx.array(hidden), mx.array(sizes), mx.array(positions), 6)
    mx.eval(*actual)
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(np.array(actual_value, copy=True), expected_value, rtol=1e-5)
    assert np.isfinite(np.array(actual[0], copy=True)).all()


def test_centroid_mode_retains_a_cluster_member_coordinate():
    hidden, sizes, positions = _inputs(cells=2, width=8)
    output = tome_bipartite_merge(
        mx.array(hidden),
        mx.array(sizes),
        mx.array(positions),
        6,
        position_mode="centroid",
    )
    mx.eval(*output)
    output_positions = np.array(output[2], copy=True)
    for cell in range(2):
        members = {tuple(value) for value in positions[0, cell]}
        assert all(tuple(value) in members for value in output_positions[0, cell])


def test_mlx_progressive_final_value_is_cell_mean_and_zero_input_is_finite():
    hidden, sizes, positions = _inputs(cells=2, width=3)
    expected_mean = hidden.mean(axis=2)
    values = (mx.array(hidden), mx.array(sizes), mx.array(positions))
    for target in (6, 3, 1):
        values = tome_bipartite_merge(*values, target)
    mx.eval(*values)
    np.testing.assert_allclose(
        np.array(values[0][:, :, 0], copy=True), expected_mean, rtol=1e-5
    )
    np.testing.assert_array_equal(np.array(values[1][:, :, 0], copy=True), 9)

    zeros = mx.zeros((1, 1, 9, 3), dtype=mx.float32)
    zero_result = tome_bipartite_merge(
        zeros,
        mx.ones((1, 1, 9)),
        mx.array(positions[:, :1]),
        6,
    )
    mx.eval(*zero_result)
    assert np.isfinite(np.array(zero_result[0], copy=True)).all()


@pytest.mark.parametrize(
    "mutation,target,message",
    (
        ("hidden", 6, "hidden must"),
        ("sizes", 6, "sizes must"),
        ("positions", 6, "positions must"),
        (None, 0, "target_tokens"),
        (None, 9, "target_tokens"),
    ),
)
def test_merge_validation(mutation, target, message):
    hidden, sizes, positions = _inputs(cells=1)
    if mutation == "hidden":
        hidden = hidden[0]
    elif mutation == "sizes":
        sizes = sizes[..., :-1]
    elif mutation == "positions":
        positions = positions[..., :1]
    with pytest.raises(ValueError, match=message):
        tome_bipartite_merge_numpy(hidden, sizes, positions, target)
