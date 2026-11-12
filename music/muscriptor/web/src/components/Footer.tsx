import clsx from "clsx";

export function RuntimeBadge({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        "flex items-center gap-3 rounded-full border border-line-strong bg-surface px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-muted",
        className,
      )}
    >
      <span className="h-2 w-2 rounded-full bg-ok shadow-[0_0_12px_var(--color-ok)]" />
      Local runtime · no cloud
    </div>
  );
}

export function Footer() {
  return (
    <footer className="mx-auto mt-4 flex max-w-7xl flex-wrap items-center justify-between gap-6 border-t border-line px-7 py-10 max-[760px]:flex-col max-[760px]:items-start">
      <div className="max-w-xl">
        <p className="mb-2 font-mono text-xs uppercase tracking-[0.18em] text-accent">
          Audio → events → editable MIDI
        </p>
        <p className="m-0 text-muted">
          Files are processed by the model running on this machine. The local
          studio keeps the piano roll, audition controls, and exports together
          in one private workspace.
        </p>
      </div>
      <RuntimeBadge />
    </footer>
  );
}
