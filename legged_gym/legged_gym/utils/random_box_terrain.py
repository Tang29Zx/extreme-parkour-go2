"""Seeded random multi-box terrain generation for the Go2 task."""

import numpy as np


def _aligned_sample(rng, value_range, scale):
    lower, upper = map(float, value_range)
    lower_cell = int(np.ceil(lower / scale - 1e-8))
    upper_cell = int(np.floor(upper / scale + 1e-8))
    if lower_cell > upper_cell:
        raise ValueError("The configured range contains no grid-aligned value.")
    return float(rng.integers(lower_cell, upper_cell + 1) * scale)


def _aligned_fixed(value, scale):
    cell = int(round(float(value) / scale))
    aligned = float(cell * scale)
    if not np.isclose(aligned, float(value), atol=1e-8):
        raise ValueError("A fixed override must align with the terrain grid.")
    return aligned


def _balanced_box_counts(seed, layout_count, minimum, maximum):
    choices = np.arange(int(minimum), int(maximum) + 1, dtype=np.int64)
    if layout_count < len(choices):
        raise ValueError("num_unique_layouts must cover every configured box count.")
    counts = np.resize(choices, int(layout_count))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7919]))
    rng.shuffle(counts)
    return counts


def _sample_gap(rng, distributions, scale):
    weights = np.asarray(
        [float(distribution["weight"]) for distribution in distributions],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Gap distribution weights must be non-negative and non-zero.")
    weights /= weights.sum()
    selected = int(rng.choice(len(distributions), p=weights))
    return _aligned_sample(rng, distributions[selected]["range"], scale), selected


def _get_indexed_override(overrides, index):
    if not overrides:
        return None
    return overrides.get(index, overrides.get(str(index)))


def resolve_random_box_layout(kwargs, layout_index):
    """Resolve an optional logical layout to another seeded source layout."""

    logical_layout_count = int(kwargs["num_unique_layouts"])
    if logical_layout_count <= 0:
        raise ValueError("num_unique_layouts must be positive.")
    logical_layout_index = int(layout_index) % logical_layout_count
    preset = _get_indexed_override(
        kwargs.get("layout_presets"), logical_layout_index
    )
    if preset is None:
        return kwargs, logical_layout_index

    resolved_kwargs = dict(kwargs)
    resolved_kwargs.pop("layout_presets", None)
    preset = dict(preset)
    source_layout_index = int(
        preset.pop("source_layout_index", logical_layout_index)
    )
    resolved_kwargs.update(preset)

    source_layout_count = int(resolved_kwargs["num_unique_layouts"])
    if source_layout_count <= 0:
        raise ValueError("A layout preset must have a positive layout count.")
    return resolved_kwargs, source_layout_index % source_layout_count


def select_roughness_range(seed, layout_count, distributions, layout_index):
    """Select a reproducible roughness bucket while preserving global weights."""

    weights = np.asarray(
        [float(distribution["weight"]) for distribution in distributions],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Roughness weights must be non-negative and non-zero.")
    weights /= weights.sum()
    expected_counts = weights * int(layout_count)
    counts = np.floor(expected_counts).astype(np.int64)
    remainder = int(layout_count) - int(counts.sum())
    if remainder:
        fractional_order = np.argsort(-(expected_counts - counts))
        counts[fractional_order[:remainder]] += 1

    classes = np.repeat(np.arange(len(distributions), dtype=np.int64), counts)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 65537]))
    rng.shuffle(classes)
    selected = int(classes[int(layout_index) % int(layout_count)])
    return tuple(map(float, distributions[selected]["range"])), selected


def build_random_box_terrain(terrain, cfg, layout_index):
    """Fill one SubTerrain with one deterministic random multi-box layout."""

    kwargs, layout_index = resolve_random_box_layout(
        cfg.random_box_kwargs, layout_index
    )
    horizontal_scale = float(terrain.horizontal_scale)
    vertical_scale = float(terrain.vertical_scale)
    track_length = terrain.width * horizontal_scale
    track_width = terrain.length * horizontal_scale

    layout_count = int(kwargs["num_unique_layouts"])
    layout_index = int(layout_index) % layout_count
    rng = np.random.default_rng(
        np.random.SeedSequence([int(kwargs["seed"]), 104729, layout_index])
    )
    box_counts = _balanced_box_counts(
        kwargs["seed"],
        layout_count,
        kwargs["box_count_range"][0],
        kwargs["box_count_range"][1],
    )
    box_count = int(box_counts[layout_index])
    if int(cfg.num_goals) < int(kwargs["box_count_range"][1]) + 1:
        raise ValueError(
            "num_goals must contain the maximum box count plus one exit."
        )

    spawn_x = float(kwargs["spawn_margin"])
    center_y = track_width / 2.0
    terrain.height_field_raw.fill(0)
    goals = np.zeros((cfg.num_goals, 2), dtype=np.float32)
    box_specs = []

    cursor_x = spawn_x + _aligned_sample(
        rng, kwargs["first_runup_range"], horizontal_scale
    )
    for box_index in range(box_count):
        if box_index > 0:
            sampled_gap, gap_class = _sample_gap(
                rng, kwargs["gap_distributions"], horizontal_scale
            )
            gap_override = _get_indexed_override(
                kwargs.get("gap_overrides"), box_index
            )
            gap = (
                _aligned_fixed(gap_override, horizontal_scale)
                if gap_override is not None
                else sampled_gap
            )
            if gap_override is not None:
                gap_class = -2
            cursor_x += gap
        else:
            gap = cursor_x - spawn_x
            gap_class = -1

        length = _aligned_sample(
            rng, kwargs["length_range"], horizontal_scale
        )
        width = _aligned_sample(
            rng, kwargs["width_range"], horizontal_scale
        )
        sampled_height = _aligned_sample(
            rng, kwargs["height_range"], vertical_scale
        )
        height_override = _get_indexed_override(
            kwargs.get("height_overrides"), box_index
        )
        height = (
            _aligned_fixed(height_override, vertical_scale)
            if height_override is not None
            else sampled_height
        )
        sampled_lateral_offset = _aligned_sample(
            rng, kwargs["lateral_offset_range"], horizontal_scale
        )
        lateral_offset_override = _get_indexed_override(
            kwargs.get("lateral_offset_overrides"), box_index
        )
        lateral_offset = (
            _aligned_fixed(lateral_offset_override, horizontal_scale)
            if lateral_offset_override is not None
            else sampled_lateral_offset
        )

        front_x = cursor_x
        rear_x = front_x + length
        y_min = center_y + lateral_offset - width / 2.0
        y_max = y_min + width
        if rear_x > track_length or y_min < 0.0 or y_max > track_width:
            raise ValueError("A sampled random box extends beyond the terrain bounds.")

        x0 = int(round(front_x / horizontal_scale))
        x1 = int(round(rear_x / horizontal_scale))
        y0 = int(round(y_min / horizontal_scale))
        y1 = int(round(y_max / horizontal_scale))
        terrain.height_field_raw[x0:x1, y0:y1] = int(
            round(height / vertical_scale)
        )

        goals[box_index] = [(front_x + rear_x) / 2.0, (y_min + y_max) / 2.0]
        box_specs.append(
            {
                "index": box_index,
                "gap": gap,
                "gap_class": gap_class,
                "front_x": front_x,
                "rear_x": rear_x,
                "length": length,
                "width": width,
                "height": height,
                "lateral_offset": lateral_offset,
                "y_min": y_min,
                "y_max": y_max,
            }
        )
        cursor_x = rear_x

    exit_x = cursor_x + float(kwargs["exit_goal_distance"])
    if exit_x > track_length - float(kwargs["end_margin"]):
        raise ValueError("The sampled layout does not leave enough exit ground.")
    goals[box_count:] = [exit_x, center_y]

    terrain.goals = goals
    terrain.env_origin = np.asarray([spawn_x, center_y, 0.0], dtype=np.float32)
    terrain.box_specs = box_specs
    terrain.box_count = box_count
    terrain.layout_index = layout_index
    return box_specs
