import { describe, expect, it } from 'vitest'
import type { PromptTurn } from './components/PromptTurnTimeline'
import { playbackTurnAtTime } from './playbackTimeline'

const turn = (
  id: number,
  number: number,
  startSeconds: number,
  endSeconds: number,
): PromptTurn => ({
  id,
  number,
  prompt: `Prompt ${number}`,
  submittedAt: '00:00:00',
  stage: 'complete',
  startSeconds,
  endSeconds,
})

describe('playbackTurnAtTime', () => {
  const turns = [
    turn(10, 1, 0, 6.67),
    turn(13, 2, 8.33, 15),
    turn(16, 3, 16.67, 23.33),
  ]

  it('keeps the previous prompt through stable-prefix carry gaps', () => {
    expect(playbackTurnAtTime(turns, 1, 16)?.number).toBe(1)
    expect(playbackTurnAtTime(turns, 7.5, 16)?.number).toBe(1)
    expect(playbackTurnAtTime(turns, 8.33, 16)?.number).toBe(2)
    expect(playbackTurnAtTime(turns, 16, 16)?.number).toBe(2)
    expect(playbackTurnAtTime(turns, 16.67, 16)?.number).toBe(3)
  })

  it('does not fall back to the latest turn while replaying the first segment', () => {
    expect(playbackTurnAtTime(turns, 0, 16)?.number).toBe(1)
    expect(playbackTurnAtTime(turns, 3, 16)?.number).toBe(1)
  })

  it('uses the active turn only before timing metadata exists', () => {
    const pending: PromptTurn[] = [{
      id: 22,
      number: 1,
      prompt: 'Pending prompt',
      submittedAt: '00:00:00',
      stage: 'playing',
    }]
    expect(playbackTurnAtTime(pending, 0, 22)?.id).toBe(22)
  })
})
