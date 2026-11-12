import { ArrowUpRight, Check, Expand, EyeOff, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { gallerySamples, type GallerySample } from '../gallerySamples'
import { perspectiveLabel } from '../perspective'
import {
  GALLERY_ITEM_REMOVED_EVENT,
  type GalleryItemRemovedDetail,
} from '../types'

interface WorldGalleryProps {
  onSelect?: (sample: GallerySample) => void
  savedSamples?: GallerySample[]
  hiddenSampleIds?: string[]
  viewOnly?: boolean
}

export function galleryPromptAtTime(
  sample: GallerySample,
  currentTime: number,
): string {
  const turns = (sample.prompt_timeline ?? [])
    .filter((turn) =>
      Number.isFinite(turn.start_seconds)
      && Number.isFinite(turn.end_seconds)
      && turn.end_seconds > turn.start_seconds
      && turn.prompt.trim(),
    )
    .sort((left, right) => left.start_seconds - right.start_seconds)
  if (!turns.length) return sample.direction
  return (
    turns.find(
      (turn) =>
        currentTime >= turn.start_seconds
        && currentTime < turn.end_seconds,
    )
    ?? [...turns].reverse().find((turn) => currentTime >= turn.start_seconds)
    ?? turns[0]
  ).prompt
}

function GalleryCard({
  sample,
  onSelect,
  onPreview,
  editMode = false,
  onHide,
  viewOnly = false,
}: {
  sample: GallerySample
  onSelect?: (sample: GallerySample) => void
  onPreview: (sample: GallerySample) => void
  editMode?: boolean
  onHide?: (sample: GallerySample) => void
  viewOnly?: boolean
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const activePrompt = galleryPromptAtTime(sample, currentTime)

  useEffect(() => {
    const video = videoRef.current
    if (!video || typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          void video.play().catch(() => {
            // Autoplay can still be deferred by a browser policy.
          })
        } else {
          video.pause()
        }
      },
      { rootMargin: '160px', threshold: 0.08 },
    )
    observer.observe(video)
    return () => observer.disconnect()
  }, [])

  const content = (
    <>
      <video
        ref={videoRef}
        src={`${import.meta.env.BASE_URL}${sample.video}`}
        muted
        loop
        playsInline
        preload="metadata"
        onTimeUpdate={(event) => {
          setCurrentTime(event.currentTarget.currentTime)
        }}
        onSeeked={(event) => {
          setCurrentTime(event.currentTarget.currentTime)
        }}
      />
      <span className="world-gallery-shade" />
      <span className="world-gallery-duration">{sample.duration}</span>
      {editMode && onHide ? (
        <button
          className="world-gallery-remove"
          type="button"
          aria-label={`Remove ${sample.title} from gallery`}
          onClick={(event) => {
            event.stopPropagation()
            onHide(sample)
          }}
        >
          <EyeOff size={14} />
          Hide
        </button>
      ) : (
        <button
          className="world-gallery-preview"
          type="button"
          aria-label={`Watch full video ${sample.title}`}
          title="Click the upper part of the video to watch full screen"
          onClick={(event) => {
            event.stopPropagation()
            onPreview(sample)
          }}
        >
          <span className="world-gallery-preview-icon">
            <Expand size={15} />
          </span>
        </button>
      )}
      <span className="world-gallery-copy">
        <small>{perspectiveLabel(sample.perspective)}</small>
        <strong>{sample.title}</strong>
        <span>{activePrompt}</span>
        {!editMode && !viewOnly && onSelect && (
          <button
            className="world-gallery-use"
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onSelect(sample)
            }}
          >
            Use as starting point
            <ArrowUpRight size={14} />
          </button>
        )}
      </span>
    </>
  )

  return (
    <article
      className={[
        'world-gallery-card',
        viewOnly ? 'view-only' : '',
        editMode ? 'editing' : '',
      ].filter(Boolean).join(' ')}
      aria-label={`Gallery video ${sample.title}`}
    >
      {content}
    </article>
  )
}

export function WorldGallery({
  onSelect,
  savedSamples = [],
  hiddenSampleIds: configuredHiddenSampleIds = [],
  viewOnly = false,
}: WorldGalleryProps) {
  const uniqueSamples = [...savedSamples, ...gallerySamples].filter(
    (sample, index, values) =>
      values.findIndex((candidate) => candidate.id === sample.id) === index,
  )
  const viewportRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const wheelTimerRef = useRef<number | undefined>(undefined)
  const [previewSample, setPreviewSample] = useState<GallerySample>()
  const [previewTime, setPreviewTime] = useState(0)
  const [isEditing, setIsEditing] = useState(false)
  const [isSavingEdits, setIsSavingEdits] = useState(false)
  const [hiddenSampleIds, setHiddenSampleIds] = useState<Set<string>>(
    () => new Set(),
  )
  const [pendingHiddenSampleIds, setPendingHiddenSampleIds] =
    useState<Set<string>>(() => new Set())
  const [editError, setEditError] = useState<string>()
  const visibleSamples = uniqueSamples.filter(
    (sample) =>
      !configuredHiddenSampleIds.includes(sample.id)
      &&
      !hiddenSampleIds.has(sample.id)
      && !pendingHiddenSampleIds.has(sample.id),
  )
  const loop = isEditing
    ? visibleSamples
    : [...visibleSamples, ...visibleSamples]

  const beginEditing = () => {
    setPendingHiddenSampleIds(new Set())
    setEditError(undefined)
    setIsEditing(true)
  }

  const cancelEditing = () => {
    setPendingHiddenSampleIds(new Set())
    setEditError(undefined)
    setIsEditing(false)
  }

  const stageHiddenSample = (sample: GallerySample) => {
    setPendingHiddenSampleIds((current) => {
      const next = new Set(current)
      next.add(sample.id)
      return next
    })
    setEditError(undefined)
  }

  const finishEditing = async () => {
    const itemIds = [...pendingHiddenSampleIds]
    if (!itemIds.length) {
      setIsEditing(false)
      return
    }

    setIsSavingEdits(true)
    setEditError(undefined)
    const results = await Promise.all(itemIds.map(async (itemId) => {
      try {
        const response = await fetch(
          `${import.meta.env.BASE_URL}v1/gallery/${encodeURIComponent(itemId)}`,
          {
            method: 'DELETE',
            headers: { Accept: 'application/json' },
          },
        )
        if (!response.ok) {
          const body = await response.text()
          let detail = body
          try {
            detail = String(
              (JSON.parse(body) as { detail?: unknown }).detail ?? body,
            )
          } catch {
            // Keep the raw response when the gateway did not return JSON.
          }
          throw new Error(detail || `HTTP ${response.status}`)
        }
        return { itemId }
      } catch (error) {
        return {
          itemId,
          error: error instanceof Error ? error.message : String(error),
        }
      }
    }))

    const hiddenIds = results
      .filter((result) => !result.error)
      .map((result) => result.itemId)
    const failedIds = results
      .filter((result) => result.error)
      .map((result) => result.itemId)

    if (hiddenIds.length) {
      setHiddenSampleIds((current) => new Set([...current, ...hiddenIds]))
      hiddenIds.forEach((itemId) => {
        window.dispatchEvent(
          new CustomEvent<GalleryItemRemovedDetail>(
            GALLERY_ITEM_REMOVED_EVENT,
            { detail: { itemId } },
          ),
        )
      })
    }
    setPendingHiddenSampleIds(new Set(failedIds))
    setIsSavingEdits(false)

    if (failedIds.length) {
      setEditError(
        failedIds.length === 1
          ? 'Could not hide that gallery case. Please try again.'
          : `Could not hide ${failedIds.length} gallery cases. Please try again.`,
      )
      return
    }
    setIsEditing(false)
  }

  useEffect(() => {
    const viewport = viewportRef.current
    const track = trackRef.current
    if (!viewport || !track) return

    const accelerateWithWheel = (event: WheelEvent) => {
      const rawDelta =
        Math.abs(event.deltaY) >= Math.abs(event.deltaX)
          ? event.deltaY
          : event.deltaX
      if (!rawDelta) return
      event.preventDefault()

      const deltaUnit =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? Math.max(1, viewport.clientWidth)
            : 1
      const normalizedDelta = rawDelta * deltaUnit
      const direction = normalizedDelta > 0 ? 1 : -1
      const minimumDistance = viewOnly ? 140 : 320
      const maximumDistance = viewOnly ? 560 : 1_400
      const distanceMultiplier = viewOnly ? 2 : 4.5
      const acceleratedRate = viewOnly ? 2 : 5
      const distance =
        direction
        * Math.min(
          maximumDistance,
          Math.max(
            minimumDistance,
            Math.abs(normalizedDelta) * distanceMultiplier,
          ),
        )
      const loopWidth = track.scrollWidth / 2
      let nextScrollLeft = viewport.scrollLeft + distance
      if (loopWidth > 0) {
        while (nextScrollLeft >= loopWidth) nextScrollLeft -= loopWidth
        while (nextScrollLeft < 0) nextScrollLeft += loopWidth
      }
      viewport.scrollLeft = nextScrollLeft

      track.classList.add('wheel-active')
      const animation = track.getAnimations?.()[0]
      if (animation) {
        animation.playbackRate = direction * acceleratedRate
        void animation.play()
      }

      if (wheelTimerRef.current !== undefined) {
        window.clearTimeout(wheelTimerRef.current)
      }
      wheelTimerRef.current = window.setTimeout(() => {
        track.classList.remove('wheel-active')
        if (animation) {
          animation.playbackRate = 1
          void animation.play()
        }
      }, 420)
    }

    viewport.addEventListener('wheel', accelerateWithWheel, {
      passive: false,
    })
    return () => {
      viewport.removeEventListener('wheel', accelerateWithWheel)
      if (wheelTimerRef.current !== undefined) {
        window.clearTimeout(wheelTimerRef.current)
      }
    }
  }, [isEditing, viewOnly, visibleSamples.length])

  useEffect(() => {
    if (!previewSample) return
    setPreviewTime(0)
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPreviewSample(undefined)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [previewSample])

  return (
    <>
      <section
        className={`world-gallery${isEditing ? ' editing' : ''}`}
        id="world-gallery"
        aria-label="Selected generated worlds"
      >
        <header className="world-gallery-head">
          <small>World gallery</small>
          {!viewOnly && (
            <div className="world-gallery-editor">
              {editError && <span role="alert">{editError}</span>}
              {isEditing ? (
                <div className="world-gallery-editor-actions">
                  <span role="status">
                    {pendingHiddenSampleIds.size
                      ? `${pendingHiddenSampleIds.size} selected`
                      : 'Select cases to hide'}
                  </span>
                  <button
                    type="button"
                    className="world-gallery-edit-cancel"
                    aria-label="Cancel gallery edits"
                    disabled={isSavingEdits}
                    onClick={cancelEditing}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="world-gallery-edit-done"
                    aria-label="Done editing gallery"
                    disabled={isSavingEdits}
                    onClick={() => void finishEditing()}
                  >
                    <Check size={14} />
                    {isSavingEdits ? 'Saving…' : 'Done'}
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="world-gallery-edit-trigger"
                  onClick={beginEditing}
                >
                  <EyeOff size={14} />
                  Edit gallery
                </button>
              )}
            </div>
          )}
        </header>
        {loop.length ? (
          <div
            className="world-gallery-viewport"
            ref={viewportRef}
            title="Scroll to move through the gallery faster"
          >
            <div className="world-gallery-track" ref={trackRef}>
              {loop.map((sample, index) => (
                <GalleryCard
                  key={`${sample.id}-${index}`}
                  sample={sample}
                  onSelect={onSelect}
                  onPreview={setPreviewSample}
                  editMode={isEditing}
                  onHide={stageHiddenSample}
                  viewOnly={viewOnly}
                />
              ))}
            </div>
          </div>
        ) : (
          <p className="world-gallery-empty">
            {isEditing
              ? 'All gallery cases are selected to hide.'
              : 'No gallery cases to show.'}
          </p>
        )}
      </section>
      {previewSample && (
        <div
          className="gallery-player-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setPreviewSample(undefined)
            }
          }}
        >
          <section
            className="gallery-player-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="gallery-player-title"
          >
            <header>
              <div>
                <small>FULL VIDEO</small>
                <h2 id="gallery-player-title">{previewSample.title}</h2>
              </div>
              <button
                type="button"
                aria-label="Close full gallery video"
                onClick={() => setPreviewSample(undefined)}
              >
                <X size={18} />
              </button>
            </header>
            <video
              src={`${import.meta.env.BASE_URL}${previewSample.video}`}
              controls
              autoPlay
              playsInline
              preload="auto"
              onTimeUpdate={(event) => {
                setPreviewTime(event.currentTarget.currentTime)
              }}
              onSeeked={(event) => {
                setPreviewTime(event.currentTarget.currentTime)
              }}
            />
            <footer>
              <span>{perspectiveLabel(previewSample.perspective)}</span>
              <p>{galleryPromptAtTime(previewSample, previewTime)}</p>
            </footer>
          </section>
        </div>
      )}
    </>
  )
}
