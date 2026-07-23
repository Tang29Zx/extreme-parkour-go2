"""Deterministic five-box terrain generation for the Go2 capability task."""

import numpy as np


def _aligned_sample(rng, value_range, scale):
    lower, upper = map(float, value_range)
    lower_cell = int(np.ceil(lower / scale - 1e-8))
    upper_cell = int(np.floor(upper / scale + 1e-8))
    if lower_cell > upper_cell:
        raise ValueError("The configured range contains no grid-aligned value.")
    return float(rng.integers(lower_cell, upper_cell + 1) * scale)


def _sample_gap(rng, box_index, previous_height, kwargs, scale):
    if box_index == 0:
        return _aligned_sample(rng, kwargs["first_gap_range"], scale)

    minimum_gap = 0.0
    if previous_height >= float(kwargs["high_box_threshold"]):
        minimum_gap = float(kwargs["post_high_min_gap"])

    candidates = []
    weights = []
    for distribution in kwargs["gap_distributions"]:
        lower, upper = map(float, distribution["range"])
        lower = max(lower, minimum_gap)
        if lower <= upper:
            candidates.append((lower, upper))
            weights.append(float(distribution["weight"]))

    if not candidates:
        raise ValueError("No gap distribution satisfies the minimum gap.")
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    selected = int(rng.choice(len(candidates), p=probabilities))
    return _aligned_sample(rng, candidates[selected], scale)


def build_five_box_terrain(terrain, cfg, layout_index):
    """Fill one SubTerrain with the seeded five-box course.

    The layout sampling intentionally matches the source parkour task while the
    output stays in Extreme Parkour's native heightfield and waypoint format.
    """

    kwargs = cfg.five_box_kwargs
    boxes = kwargs["boxes"]
    if len(boxes) != 5:
        raise ValueError("The five-box capability task requires exactly five boxes.")
    if int(cfg.num_goals) != len(boxes) + 1:
        raise ValueError("num_goals must contain five box goals and one exit goal.")

    horizontal_scale = float(terrain.horizontal_scale)
    vertical_scale = float(terrain.vertical_scale)
    track_length = terrain.width * horizontal_scale
    track_width = terrain.length * horizontal_scale
    spawn_x = float(kwargs["spawn_margin"])
    center_y = track_width / 2.0

    layout_count = int(kwargs["num_unique_layouts"])
    if layout_count <= 0:
        raise ValueError("num_unique_layouts must be positive.")
    layout_index = int(layout_index) % layout_count
    rng = np.random.default_rng(
        np.random.SeedSequence([int(kwargs["seed"]), 0, layout_index])
    )

    terrain.height_field_raw.fill(0)
    goals = np.zeros((cfg.num_goals, 2), dtype=np.float32)
    box_specs = []
    cursor_x = spawn_x
    previous_height = 0.0

    for box_index, box in enumerate(boxes):
        if "gap_range" in box:
            gap = _aligned_sample(rng, box["gap_range"], horizontal_scale)
        else:
            gap = _sample_gap(
                rng,
                box_index,
                previous_height,
                kwargs,
                horizontal_scale,
            )
        front_x = cursor_x + gap
        rear_x = front_x + float(box["length"])
        width = float(box["width"])
        height = float(box["height"])
        center_offset = float(box.get("lateral_offset", 0.0))
        y_min = center_y + center_offset - width / 2.0
        y_max = y_min + width

        if rear_x > track_length or y_min < 0.0 or y_max > track_width:
            raise ValueError("A configured box extends beyond the terrain bounds.")

        x0 = int(round(front_x / horizontal_scale))
        x1 = int(round(rear_x / horizontal_scale))
        y0 = int(round(y_min / horizontal_scale))
        y1 = int(round(y_max / horizontal_scale))
        height_cells = int(round(height / vertical_scale))
        terrain.height_field_raw[x0:x1, y0:y1] = height_cells

        goals[box_index] = [(front_x + rear_x) / 2.0, (y_min + y_max) / 2.0]
        box_specs.append(
            {
                "index": box_index,
                "gap": gap,
                "front_x": front_x,
                "rear_x": rear_x,
                "y_min": y_min,
                "y_max": y_max,
                "height": height,
            }
        )
        cursor_x = rear_x
        previous_height = height

    exit_x = min(cursor_x + float(kwargs["exit_goal_distance"]), track_length - 0.5)
    if exit_x <= cursor_x:
        raise ValueError("The course does not leave enough flat ground after box five.")
    goals[-1] = [exit_x, center_y]

    terrain.goals = goals
    terrain.env_origin = np.asarray([spawn_x, center_y, 0.0], dtype=np.float32)
    terrain.box_specs = box_specs
    terrain.layout_index = layout_index
    return box_specs
