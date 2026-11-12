import '@testing-library/jest-dom/vitest'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  GALLERY_ITEM_REMOVED_EVENT,
  type GalleryItemRemovedDetail,
} from '../types'
import { WorldGallery } from './WorldGallery'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('WorldGallery wheel browsing', () => {
  it('captures the wheel and immediately advances the gallery', () => {
    const { container } = render(
      <WorldGallery onSelect={vi.fn()} />,
    )
    const viewport = container.querySelector<HTMLElement>(
      '.world-gallery-viewport',
    )
    const track = container.querySelector<HTMLElement>(
      '.world-gallery-track',
    )
    expect(viewport).not.toBeNull()
    expect(track).not.toBeNull()
    Object.defineProperty(track, 'scrollWidth', {
      configurable: true,
      value: 4_000,
    })

    const wheel = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      deltaY: 120,
    })
    viewport?.dispatchEvent(wheel)

    expect(wheel.defaultPrevented).toBe(true)
    expect(viewport?.scrollLeft).toBeGreaterThanOrEqual(320)
    expect(track).toHaveClass('wheel-active')
  })

  it('uses gentler wheel movement inside the Studio gallery', () => {
    const { container } = render(<WorldGallery viewOnly />)
    const viewport = container.querySelector<HTMLElement>(
      '.world-gallery-viewport',
    )
    const track = container.querySelector<HTMLElement>(
      '.world-gallery-track',
    )
    Object.defineProperty(track, 'scrollWidth', {
      configurable: true,
      value: 4_000,
    })

    fireEvent.wheel(viewport as HTMLElement, {
      deltaX: 0,
      deltaY: 100,
    })

    expect(viewport?.scrollLeft).toBeGreaterThanOrEqual(140)
    expect(viewport?.scrollLeft).toBeLessThan(320)
  })

  it('keeps use-as-starting-point separate from full video preview', async () => {
    const onSelect = vi.fn()
    const { container } = render(<WorldGallery onSelect={onSelect} />)
    const gallery = within(container)
    const user = userEvent.setup()

    fireEvent.click(
      gallery.getAllByRole('button', {
        name: /Use as starting point/i,
      })[0],
    )
    expect(onSelect).toHaveBeenCalledOnce()
    onSelect.mockClear()

    await user.click(
      gallery.getAllByRole('button', {
        name: /Watch full video Blue Station Wagon/i,
      })[0],
    )
    const dialog = gallery.getByRole('dialog', {
      name: 'Blue Station Wagon',
    })
    expect(dialog).toBeInTheDocument()
    expect(dialog.querySelector('video')).toHaveAttribute('controls')
    expect(onSelect).not.toHaveBeenCalled()
    expect(
      gallery.getAllByRole('button', {
        name: /Watch full video Blue Station Wagon/i,
      })[0],
    ).toHaveClass('world-gallery-preview')
  })

  it('shows the prompt matching the current gallery video time', () => {
    const sample = {
      id: 'dynamic-prompts',
      title: 'Clockwork Fox',
      video: 'gallery/clockwork-fox.mp4',
      duration: '24s',
      perspective: 'third-person follow' as const,
      subject: 'A clockwork fox',
      attributes: 'A mossy observatory',
      direction: 'Fallback direction',
      prompt_timeline: [
        {
          number: 1,
          prompt: 'The fox enters the observatory',
          start_seconds: 0,
          end_seconds: 12,
        },
        {
          number: 2,
          prompt: 'The fox activates the brass orrery',
          start_seconds: 12,
          end_seconds: 24,
        },
      ],
    }
    render(<WorldGallery savedSamples={[sample]} viewOnly />)
    const card = screen.getAllByLabelText('Gallery video Clockwork Fox')[0]
    const video = card.querySelector('video') as HTMLVideoElement

    expect(within(card).getByText('The fox enters the observatory'))
      .toBeInTheDocument()
    Object.defineProperty(video, 'currentTime', {
      configurable: true,
      value: 13,
    })
    fireEvent.timeUpdate(video)
    expect(within(card).getByText('The fox activates the brass orrery'))
      .toBeInTheDocument()

    fireEvent.click(
      within(card).getByRole('button', {
        name: 'Watch full video Clockwork Fox',
      }),
    )
    const dialog = screen.getByRole('dialog', { name: 'Clockwork Fox' })
    const dialogVideo = dialog.querySelector('video') as HTMLVideoElement
    Object.defineProperty(dialogVideo, 'currentTime', {
      configurable: true,
      value: 13,
    })
    fireEvent.timeUpdate(dialogVideo)
    expect(within(dialog).getByText('The fox activates the brass orrery'))
      .toBeInTheDocument()
  })
})

describe('WorldGallery editing', () => {
  const savedSample = {
    id: 'saved/a sample',
    title: 'Saved Moon Walk',
    video: 'gallery/favorites/saved-moon-walk.mp4',
    duration: '12s',
    perspective: 'first-person' as const,
    subject: 'An astronaut',
    attributes: 'A moon base',
    direction: 'Walk toward the crater',
  }

  it('stages hidden cases and restores them when editing is cancelled', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<WorldGallery savedSamples={[savedSample]} />)
    const user = userEvent.setup()

    await user.click(screen.getByRole('button', { name: 'Edit gallery' }))
    expect(
      screen.getByRole('button', {
        name: 'Remove Saved Moon Walk from gallery',
      }),
    ).toBeInTheDocument()

    await user.click(
      screen.getByRole('button', {
        name: 'Remove Saved Moon Walk from gallery',
      }),
    )
    expect(
      screen.queryByLabelText('Gallery video Saved Moon Walk'),
    ).not.toBeInTheDocument()

    await user.click(
      screen.getByRole('button', { name: 'Cancel gallery edits' }),
    )
    expect(
      screen.getAllByLabelText('Gallery video Saved Moon Walk'),
    ).toHaveLength(2)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('deletes selected cases, updates the list, and announces the removal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ removed: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const removed = vi.fn()
    const onRemoved = (event: Event) => {
      removed(
        (event as CustomEvent<GalleryItemRemovedDetail>).detail.itemId,
      )
    }
    window.addEventListener(GALLERY_ITEM_REMOVED_EVENT, onRemoved)
    render(<WorldGallery savedSamples={[savedSample]} />)
    const user = userEvent.setup()

    await user.click(screen.getByRole('button', { name: 'Edit gallery' }))
    await user.click(
      screen.getByRole('button', {
        name: 'Remove Saved Moon Walk from gallery',
      }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Done editing gallery' }),
    )

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        `${import.meta.env.BASE_URL}v1/gallery/saved%2Fa%20sample`,
        {
          method: 'DELETE',
          headers: { Accept: 'application/json' },
        },
      )
    })
    expect(
      screen.queryByLabelText('Gallery video Saved Moon Walk'),
    ).not.toBeInTheDocument()
    expect(removed).toHaveBeenCalledWith(savedSample.id)
    expect(
      screen.getByRole('button', { name: 'Edit gallery' }),
    ).toBeInTheDocument()
    window.removeEventListener(GALLERY_ITEM_REMOVED_EVENT, onRemoved)
  })

  it('keeps failed cases selected so the user can retry or cancel', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Gallery is busy' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )
    render(<WorldGallery savedSamples={[savedSample]} />)
    const user = userEvent.setup()

    await user.click(screen.getByRole('button', { name: 'Edit gallery' }))
    await user.click(
      screen.getByRole('button', {
        name: 'Remove Saved Moon Walk from gallery',
      }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Done editing gallery' }),
    )

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not hide that gallery case',
    )
    await user.click(
      screen.getByRole('button', { name: 'Cancel gallery edits' }),
    )
    expect(
      screen.getAllByLabelText('Gallery video Saved Moon Walk'),
    ).toHaveLength(2)
  })

  it('does not expose editing controls in the Studio gallery', () => {
    render(<WorldGallery savedSamples={[savedSample]} viewOnly />)
    expect(
      screen.queryByRole('button', { name: 'Edit gallery' }),
    ).not.toBeInTheDocument()
  })

  it('honors persisted hidden ids for built-in gallery cases', () => {
    render(<WorldGallery hiddenSampleIds={['station-wagon']} />)

    expect(
      screen.queryByLabelText('Gallery video Blue Station Wagon'),
    ).not.toBeInTheDocument()
    expect(
      screen.getAllByLabelText('Gallery video Alpine Armored Convoy'),
    ).toHaveLength(2)
  })
})
