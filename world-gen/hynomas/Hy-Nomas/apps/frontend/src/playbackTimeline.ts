import type { PromptTurn } from './components/PromptTurnTimeline'

/**
 * Pick exactly one prompt for a point on the cumulative video timeline.
 *
 * A generated segment can contain a short stable-prefix carry copied from the
 * previous segment. That creates a deliberate gap between one turn's rendered
 * end and the next turn's semantic start. During that gap the previous prompt
 * must remain visible; falling back to the newest "playing" turn makes every
 * prompt flash while replaying the beginning of a long video.
 */
export function playbackTurnAtTime(
  turns: PromptTurn[],
  currentTime: number,
  activeTurnId?: number,
): PromptTurn | undefined {
  const timedTurns = turns
    .filter(
      (turn) =>
        Number.isFinite(turn.startSeconds)
        && Number.isFinite(turn.endSeconds),
    )
    .sort(
      (left, right) =>
        (left.startSeconds ?? 0) - (right.startSeconds ?? 0)
        || left.number - right.number,
    )

  if (timedTurns.length > 0) {
    return (
      [...timedTurns].reverse().find(
        (turn) => currentTime >= (turn.startSeconds ?? 0),
      )
      ?? timedTurns[0]
    )
  }

  return (
    turns.find((turn) => turn.id === activeTurnId)
    ?? turns.at(-1)
  )
}
