import '@testing-library/jest-dom/vitest'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StructuredPromptView } from './StructuredPromptView'

describe('StructuredPromptView', () => {
  it('turns complete K3 JSON into a scannable scene plan', () => {
    render(
      <StructuredPromptView
        complete
        source={JSON.stringify({
          perspective_view: 'third-person follow',
          Style: 'real-world footage',
          subject_list: [
            {
              subject_id: '<|Subject1|>',
              display_name: 'clockwork fox',
              subject_attribute: 'Brass body, green glass eyes, red scarf.',
            },
          ],
          segments: [
            {
              start_time: '00:00.000',
              end_time: '00:08.000',
              visual_description_long:
                'A mossy observatory glows at dawn. <Camera1>The camera tracks forward.</Camera1>',
              description_change: null,
              contained_actions: [
                {
                  start_time: '00:00.000',
                  end_time: '00:08.000',
                  action_description:
                    '<Action1><|Subject1|> walks toward the orrery.</Action1>',
                },
              ],
            },
          ],
        })}
      />,
    )

    expect(screen.getByText('Third person · Follow')).toBeInTheDocument()
    expect(screen.getByText('Real-world footage')).toBeInTheDocument()
    expect(screen.getByText('clockwork fox')).toBeInTheDocument()
    expect(screen.getByText('Segment 01')).toBeInTheDocument()
    expect(screen.getAllByText('Subject 1')).not.toHaveLength(0)
    expect(screen.getByText('Camera 1')).toBeInTheDocument()
    expect(screen.getByText('View exact JSON')).toBeInTheDocument()
  })

  it('keeps partial streaming JSON readable without parsing it', () => {
    const { container } = render(
      <StructuredPromptView source={'{"segments":['} complete={false} />,
    )

    expect(container.querySelector('.prompt-json-raw')).toHaveTextContent(
      '{"segments":[',
    )
    expect(container.querySelector('.json-key')).toHaveTextContent('"segments"')
  })
})
