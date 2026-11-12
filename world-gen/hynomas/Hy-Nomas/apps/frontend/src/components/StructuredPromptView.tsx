import type { ReactNode } from 'react'

interface StructuredPromptViewProps {
  source: string
  complete: boolean
}

interface PromptSubject {
  subject_id?: unknown
  display_name?: unknown
  subject_attribute?: unknown
}

interface PromptAction {
  start_time?: unknown
  end_time?: unknown
  action_description?: unknown
}

interface PromptSegment {
  start_time?: unknown
  end_time?: unknown
  visual_description_long?: unknown
  description_change?: unknown
  contained_actions?: unknown
}

interface ParsedPrompt {
  perspective_view?: unknown
  Style?: unknown
  subject_list?: unknown
  segments?: unknown
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function displayEnum(value: string) {
  const labels: Record<string, string> = {
    'first-person': 'First person',
    'second-person': 'Second person',
    'third-person follow': 'Third person · Follow',
    'third-person observer': 'Third person · Observer',
    'real-world footage': 'Real-world footage',
    'animation/anime': 'Animation / Anime',
    CG: 'CG',
    'video game': 'Video game',
    'software/web/app/computer screen recording': 'Screen recording',
    slideshow: 'Slideshow',
  }
  return labels[value] ?? value
}

function highlightSemanticText(value: string): ReactNode[] {
  return value
    .replace(/<\/?Action\d+>/g, '')
    .split(/(<\|Subject\d+\|>|<\/?Camera\d+>)/g)
    .filter(Boolean)
    .map((part, index) => {
      if (/^<\|Subject\d+\|>$/.test(part)) {
        return (
          <mark className="semantic-token subject" key={`${part}-${index}`}>
            {part.replace(/<\|Subject(\d+)\|>/, 'Subject $1')}
          </mark>
        )
      }
      if (/^<Camera\d+>$/.test(part)) {
        return (
          <mark className="semantic-token camera" key={`${part}-${index}`}>
            {part.slice(1, -1).replace(/(\D)(\d+)$/, '$1 $2')}
          </mark>
        )
      }
      if (/^<\/Camera\d+>$/.test(part)) return null
      return part
    })
}

function JsonSyntax({ source }: { source: string }) {
  const matcher =
    /"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g
  const nodes: ReactNode[] = []
  let cursor = 0
  let match: RegExpExecArray | null

  while ((match = matcher.exec(source))) {
    if (match.index > cursor) {
      nodes.push(source.slice(cursor, match.index))
    }
    const token = match[0]
    const after = source.slice(match.index + token.length)
    const kind = token.startsWith('"')
      ? after.trimStart().startsWith(':')
        ? 'key'
        : 'string'
      : token === 'true' || token === 'false'
        ? 'boolean'
        : token === 'null'
          ? 'null'
          : 'number'
    nodes.push(
      <span className={`json-${kind}`} key={`${match.index}-${token}`}>
        {token}
      </span>,
    )
    cursor = match.index + token.length
  }
  if (cursor < source.length) nodes.push(source.slice(cursor))

  return <pre className="prompt-json-raw">{nodes}</pre>
}

function parsePrompt(source: string): ParsedPrompt | undefined {
  try {
    const parsed = JSON.parse(source)
    return parsed && typeof parsed === 'object'
      ? parsed as ParsedPrompt
      : undefined
  } catch {
    return undefined
  }
}

export function StructuredPromptView({
  source,
  complete,
}: StructuredPromptViewProps) {
  const parsed = complete ? parsePrompt(source) : undefined
  if (!parsed) return <JsonSyntax source={source} />

  const perspective = text(parsed.perspective_view)
  const style = text(parsed.Style)
  const subjects = Array.isArray(parsed.subject_list)
    ? parsed.subject_list.filter(
        (item): item is PromptSubject => Boolean(item && typeof item === 'object'),
      )
    : []
  const segments = Array.isArray(parsed.segments)
    ? parsed.segments.filter(
        (item): item is PromptSegment => Boolean(item && typeof item === 'object'),
      )
    : []

  return (
    <div className="structured-prompt">
      {(perspective || style) && (
        <div className="structured-prompt-overview">
          {perspective && (
            <span>
              <small>Perspective</small>
              <strong>{displayEnum(perspective)}</strong>
            </span>
          )}
          {style && (
            <span>
              <small>Style</small>
              <strong>{displayEnum(style)}</strong>
            </span>
          )}
        </div>
      )}

      {subjects.length > 0 && (
        <section className="structured-prompt-section">
          <h4>Subjects</h4>
          <div className="structured-subjects">
            {subjects.map((subject, index) => (
              <article key={`${text(subject.subject_id) ?? 'subject'}-${index}`}>
                <span>
                  {text(subject.subject_id)?.replace(/[<|>]/g, '')
                    ?? `Subject ${index + 1}`}
                </span>
                <strong>
                  {text(subject.display_name) ?? `Subject ${index + 1}`}
                </strong>
                {text(subject.subject_attribute) && (
                  <p>{text(subject.subject_attribute)}</p>
                )}
              </article>
            ))}
          </div>
        </section>
      )}

      {segments.map((segment, segmentIndex) => {
        const scene =
          text(segment.visual_description_long)
          ?? text(segment.description_change)
        const actions = Array.isArray(segment.contained_actions)
          ? segment.contained_actions.filter(
              (item): item is PromptAction =>
                Boolean(item && typeof item === 'object'),
            )
          : []
        return (
          <section
            className="structured-prompt-section segment"
            key={segmentIndex}
          >
            <header>
              <h4>Segment {String(segmentIndex + 1).padStart(2, '0')}</h4>
              {(text(segment.start_time) || text(segment.end_time)) && (
                <time>
                  {text(segment.start_time) ?? '—'} →{' '}
                  {text(segment.end_time) ?? '—'}
                </time>
              )}
            </header>
            {scene && (
              <div className="structured-scene">
                <small>Scene & camera</small>
                <p>{highlightSemanticText(scene)}</p>
              </div>
            )}
            {actions.length > 0 && (
              <div className="structured-actions">
                <small>Actions</small>
                {actions.map((action, actionIndex) => (
                  <article key={actionIndex}>
                    <span>{String(actionIndex + 1).padStart(2, '0')}</span>
                    <div>
                      <time>
                        {text(action.start_time) ?? '—'} →{' '}
                        {text(action.end_time) ?? '—'}
                      </time>
                      <p>
                        {highlightSemanticText(
                          text(action.action_description) ?? 'Action',
                        )}
                      </p>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )
      })}

      <details
        className="structured-prompt-raw"
        onClick={(event) => event.stopPropagation()}
      >
        <summary>View exact JSON</summary>
        <JsonSyntax source={JSON.stringify(parsed, null, 2)} />
      </details>
    </div>
  )
}
