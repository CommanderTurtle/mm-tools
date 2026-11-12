import type { LegacyPerspective, Perspective } from './types'

export function normalizePerspective(
  value: LegacyPerspective | string | null | undefined,
): Perspective {
  if (value === 'third-person') return 'third-person follow'
  if (
    value === 'first-person'
    || value === 'third-person follow'
    || value === 'second-person'
    || value === 'third-person observer'
  ) {
    return value
  }
  return 'first-person'
}

export function perspectiveLabel(
  value: LegacyPerspective | string | null | undefined,
): string {
  const perspective = normalizePerspective(value)
  return {
    'first-person': 'First person',
    'third-person follow': 'Third person',
    'second-person': 'Second person (talking head)',
    'third-person observer': 'Third person observer',
  }[perspective]
}
