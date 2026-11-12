import '@testing-library/jest-dom/vitest'
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  WORLD_SESSION_IDLE_TIMEOUT_MS,
  WorldStudio,
} from './WorldStudio'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'succeeded' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  )
})

describe('WorldStudio session inactivity', () => {
  it('opens the gallery inside Studio without releasing the world', async () => {
    const onReconfigure = vi.fn()
    render(
      <WorldStudio
        config={{
          perspective: 'first-person',
          subjectDescription: 'A clockwork fox',
          environmentDescription: '',
        }}
        backendConfig={{
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 80,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
          model_name: 'Stage 4 SFT',
          dit_weight: '/models/stage-4-sft',
        }}
        onReconfigure={onReconfigure}
        onSessionExpired={vi.fn()}
        theme="dark"
        onThemeToggle={vi.fn()}
      />,
    )

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'More options' }))
    await user.click(screen.getByRole('menuitem', { name: /View gallery/i }))

    expect(
      screen.getByRole('dialog', { name: 'Saved and featured worlds' }),
    ).toBeInTheDocument()
    expect(onReconfigure).not.toHaveBeenCalled()
    expect(
      vi.mocked(fetch).mock.calls.some(([url]) =>
        String(url).includes('/release'),
      ),
    ).toBe(false)
    expect(screen.getByText('Stage 4 SFT')).toBeInTheDocument()
  })

  it('expires after 15 minutes and resets on real user interaction', () => {
    vi.useFakeTimers()
    const onSessionExpired = vi.fn()

    render(
      <WorldStudio
        config={{
          perspective: 'first-person',
          subjectDescription: 'A clockwork fox',
          environmentDescription: '',
        }}
        backendConfig={{
          rollout_type: 'seg_ar',
          prompt_source: 'prompt',
          num_frames: 80,
          num_frames_source: 'request_or_config',
          fps: 24,
          worker_online: true,
        }}
        onReconfigure={vi.fn()}
        onSessionExpired={onSessionExpired}
        theme="dark"
        onThemeToggle={vi.fn()}
      />,
    )

    act(() => {
      vi.advanceTimersByTime(WORLD_SESSION_IDLE_TIMEOUT_MS - 1)
    })
    expect(onSessionExpired).not.toHaveBeenCalled()

    fireEvent.pointerDown(window)
    act(() => {
      vi.advanceTimersByTime(WORLD_SESSION_IDLE_TIMEOUT_MS - 1)
    })
    expect(onSessionExpired).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(onSessionExpired).toHaveBeenCalledOnce()
  })
})
