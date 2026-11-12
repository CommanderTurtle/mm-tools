"""MIDI output utilities."""

from pathlib import Path

from muscriptor.tokenizer.notes import Note, note2note_event, note_event2midi
from muscriptor.utils.beats import BeatGrid

# Written when no grid was detected: 120 BPM and no time signature, leaving the
# meter for notation software to guess.
PLACEHOLDER_GRID = BeatGrid(bpm=120, beats_per_bar=None, first_downbeat=0.0)


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    program_names: dict[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
):
    """Convert a list of Note objects to a mido MidiFile.

    `program_names` maps program numbers to human-readable track names
    (see note_event2midi).

    `grid` is a detected beat grid (see muscriptor.utils.beats): it supplies the
    tempo, the time signature and a delay that puts bar lines on real downbeats.
    Defaults to PLACEHOLDER_GRID.
    """
    beat_grid = beat_grid or PLACEHOLDER_GRID
    return note_event2midi(
        note2note_event(notes),
        output_file=None,
        velocity=velocity,
        tempo=round(60_000_000 / beat_grid.bpm),
        program_names=program_names,
        beats_per_bar=beat_grid.beats_per_bar,
        offset_s=beat_grid.bar_offset(),
    )


def save_midi(
    notes: list[Note],
    path: str | Path,
    velocity: int = 100,
    program_names: dict[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
) -> None:
    """Save a list of Note objects as a MIDI file."""
    midi = notes_to_midi(
        notes,
        velocity=velocity,
        program_names=program_names,
        beat_grid=beat_grid,
    )
    midi.save(str(path))
