import { useRef } from "react";
import { AudioEngine } from "../audio";

/**
 * Lazily construct a single {@link AudioEngine} for the app's lifetime.
 *
 * The engine wraps Tone's *global* transport + AudioContext and registers
 * transport listeners in its constructor, so it must exist exactly once. The
 * ref guard guarantees that even if React re-renders the owning component. (We
 * deliberately don't wrap the app in <StrictMode>, whose simulated double-mount
 * would otherwise create two engines competing over the one global transport.)
 */
export function useAudioEngine(): AudioEngine {
  const ref = useRef<AudioEngine | null>(null);
  if (ref.current === null) ref.current = new AudioEngine();
  return ref.current;
}
