"""Output-only timeline trimming: never change prompts, camera data or KV."""


def first_turn_offset(boundaries, num_frames):
    if (
        boundaries is None
        or len(boundaries) < 3
        or boundaries[0] != 0
        or boundaries[-1] != num_frames
        or any(a >= b for a, b in zip(boundaries[:-1], boundaries[1:]))
    ):
        raise ValueError(
            "skip_first_turn requires at least two turns with valid pixel boundaries"
        )
    return int(boundaries[1])
