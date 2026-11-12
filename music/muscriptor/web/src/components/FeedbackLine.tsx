import clsx from "clsx";

export function FeedbackLine({ className }: { className?: string }) {
  return (
    <p className={clsx("font-mono text-[11px] uppercase tracking-[0.12em] text-muted", className)}>
      Space toggles playback · exports stay local
    </p>
  );
}
