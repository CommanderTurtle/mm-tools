import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import {
  PromptTurnTimeline,
  type PromptTurn,
} from './PromptTurnTimeline'

afterEach(cleanup)

const baseTurn: PromptTurn = {
  id: 10,
  number: 1,
  prompt: 'Enter a restaurant',
  submittedAt: '12:34:56',
  stage: 'simulating',
}

describe('PromptTurnTimeline', () => {
  it('streams Kimi thinking and JSON through one persistent output box', () => {
    const { rerender } = render(
      <PromptTurnTimeline
        turns={[
          {
            ...baseTurn,
            k3Reasoning: 'Planning the embodied first-person scene',
          },
        ]}
      />,
    )

    expect(
      screen.getAllByLabelText('Enhanced output for prompt 1'),
    ).toHaveLength(1)
    expect(screen.getByText('Enhanced Thinking')).toBeInTheDocument()
    expect(
      screen.getByText('Planning the embodied first-person scene'),
    ).toBeInTheDocument()

    rerender(
      <PromptTurnTimeline
        turns={[
          {
            ...baseTurn,
            k3Output: '{"segments":[',
            k3OutputComplete: false,
          },
        ]}
      />,
    )

    expect(
      screen.getAllByLabelText('Enhanced output for prompt 1'),
    ).toHaveLength(1)
    expect(
      screen.getByText('Enhanced Scene Plan · Streaming'),
    ).toBeInTheDocument()
    expect(
      screen
        .getByLabelText('Enhanced output for prompt 1')
        .querySelector('.prompt-json-raw'),
    ).toHaveTextContent('{"segments":[')
    expect(
      screen.queryByText('Planning the embodied first-person scene'),
    ).not.toBeInTheDocument()

    const finalJson = '{"segments":[{"description_change":"Inside."}]}'
    rerender(
      <PromptTurnTimeline
        turns={[
          {
            ...baseTurn,
            stage: 'rendering',
            k3Output: finalJson,
            k3OutputComplete: true,
            structuredPrompt: finalJson,
            promptEnhancementMode: 'kimi-k3',
            promptEnhancerSeed: 424242,
          },
        ]}
      />,
    )

    expect(
      screen.getAllByLabelText('Enhanced output for prompt 1'),
    ).toHaveLength(1)
    expect(screen.getByText('Enhanced Scene Plan')).toBeInTheDocument()
    expect(screen.getByText('Segment 01')).toBeInTheDocument()
    expect(screen.getByText('Inside.')).toBeInTheDocument()
    expect(screen.getByText('View exact JSON')).toBeInTheDocument()
    expect(
      screen.getByText('kimi-k3'),
    ).toBeInTheDocument()
    expect(screen.getByText('LLM seed 424242')).toBeInTheDocument()
  })

  it('keeps the rendering, playing, and complete stages visible', () => {
    const { rerender } = render(
      <PromptTurnTimeline
        turns={[{ ...baseTurn, stage: 'rendering' }]}
      />,
    )

    const status = screen.getByLabelText('Prompt 1 status')
    expect(within(status).getByText('Rendering')).toHaveClass('current')
    expect(within(status).getByText('Playing')).toBeInTheDocument()
    expect(within(status).getByText('Complete')).toBeInTheDocument()

    rerender(
      <PromptTurnTimeline turns={[{ ...baseTurn, stage: 'playing' }]} />,
    )
    expect(
      within(screen.getByLabelText('Prompt 1 status')).getByText('Playing'),
    ).toHaveClass('current')

    rerender(
      <PromptTurnTimeline turns={[{ ...baseTurn, stage: 'complete' }]} />,
    )
    expect(
      within(screen.getByLabelText('Prompt 1 status')).getByText('Complete'),
    ).toHaveClass('current')
  })
})
