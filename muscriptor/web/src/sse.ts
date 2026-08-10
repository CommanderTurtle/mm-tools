/** A failed `/transcribe` request. `userMessage`, when set, is a server-sent
 *  explanation safe to show the user (e.g. "could not decode audio file …"). */
export class TranscribeError extends Error {
  readonly userMessage?: string;
  readonly status?: number;
  constructor(message: string, opts: { userMessage?: string; status?: number } = {}) {
    super(message);
    this.name = "TranscribeError";
    this.userMessage = opts.userMessage;
    this.status = opts.status;
  }
}

export interface TranscribeOptions {
  /** Extra multipart form fields to send alongside the file (e.g. instruments). */
  extra?: Record<string, string | string[]>;
  signal?: AbortSignal;
  /** Sent as the `X-Client-Id` header. The server uses it to scope preemption: a
   *  new request only cancels an in-flight one that carries the same id (a
   *  resubmit from this same tab). Two tabs send different ids, so opening a
   *  second one no longer stops the first tab's transcription. */
  clientId?: string;
  /** Called once the server has accepted the upload (2xx — it holds the
   *  transcription lock and the stream is opening). The UI waits for this before
   *  leaving the welcome screen, so a refused attempt never navigates away. */
  onAccepted?: () => void;
  /** Called on each 503 refusal (another transcription is in progress), with the
   *  wait until the next attempt. Only `streamTranscribeWithRetry` reports it. */
  onBusy?: (info: { attempt: number; retryInMs: number }) => void;
}

/** Stream SSE `data:` JSON payloads from a POST upload of `file`. */
export async function* streamTranscribe(
  url: string,
  file: File,
  opts: TranscribeOptions = {},
): AsyncGenerator<unknown> {
  const { extra, signal, clientId, onAccepted } = opts;
  const form = new FormData();
  form.append("file", file, file.name);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (Array.isArray(v)) {
        for (const item of v) form.append(k, item);
      } else {
        form.append(k, v);
      }
    }
  }
  const headers: Record<string, string> = {};
  if (clientId) headers["X-Client-Id"] = clientId;
  const resp = await fetch(url, { method: "POST", body: form, signal, headers });
  if (!resp.ok || !resp.body) {
    // FastAPI's HTTPException bodies are `{"detail": "..."}`. Pull the detail
    // out (when present) so the UI can show why the upload was rejected
    // instead of a bare status code.
    let detail: string | undefined;
    try {
      const body = await resp.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON body (or network error reading it) — leave detail unset.
    }
    throw new TranscribeError(`server returned ${resp.status}`, {
      userMessage: detail,
      status: resp.status,
    });
  }
  onAccepted?.();
  const reader = resp.body
    .pipeThrough(new TextDecoderStream())
    .getReader();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          yield JSON.parse(line.slice(6));
        }
      }
    }
  }
}

/** Resolves after `ms`, or rejects immediately (or as soon as it fires) with
 *  an `AbortError` if `signal` is aborted — so a retry wait can't outlive the
 *  request it's retrying. */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

const RETRY_INTERVAL_MS = 5000;

/** Like `streamTranscribe`, but transparently retries every
 *  `RETRY_INTERVAL_MS` when the server reports 503 (busy with another
 *  transcription). The backend refuses instantly rather than holding the
 *  connection open, so a reverse proxy fronting multiple backends (e.g.
 *  Traefik) gets a chance to route each retry to a different, idle machine.
 *
 *  Each refusal is reported through `opts.onBusy` so the UI can say the servers
 *  are busy and count down to the next attempt; `opts.onAccepted` fires on the
 *  attempt that gets through. */
export async function* streamTranscribeWithRetry(
  url: string,
  file: File,
  opts: TranscribeOptions = {},
): AsyncGenerator<unknown> {
  for (let attempt = 1; ; attempt++) {
    try {
      yield* streamTranscribe(url, file, opts);
      return;
    } catch (e) {
      if (!(e instanceof TranscribeError) || e.status !== 503) throw e;
      opts.onBusy?.({ attempt, retryInMs: RETRY_INTERVAL_MS });
      await delay(RETRY_INTERVAL_MS, opts.signal);
    }
  }
}
