/** Deliberate local-only compatibility shim. No telemetry is ever loaded. */

export type EventParams = Record<string, string | number | boolean | undefined>;

export function analyticsAvailable(): boolean {
  return false;
}

export function storedConsent(): boolean | null {
  return false;
}

export function setConsent(_consent: boolean): void {}

export function initAnalytics(): void {}

export function track(_event: string, _params: EventParams = {}): void {}
