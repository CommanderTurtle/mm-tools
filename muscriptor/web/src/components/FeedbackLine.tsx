import { useEffect, useState } from "react";
import clsx from "clsx";

/**
 * "Feedback, question, bug?" line under the output bar. Held back for a few
 * seconds and then typed out left to right, so it catches the eye of someone
 * who has already settled into watching the progress bar.
 */
const SEGMENTS: { text: string; href?: string }[] = [
  { text: "Feedback, question, bug? " },
  { text: "Email us", href: "mailto:muscriptor@kyutai.org" },
  { text: " or " },
  {
    text: "open an issue",
    href: "https://github.com/muscriptor/muscriptor/issues/new",
  },
];

const TOTAL = SEGMENTS.reduce((n, s) => n + s.text.length, 0);
const DELAY_MS = 3000;
const CHAR_MS = 35;

export function FeedbackLine({ className }: { className?: string }) {
  const [shown, setShown] = useState(0);

  useEffect(() => {
    let id: ReturnType<typeof setInterval>;
    const start = setTimeout(() => {
      id = setInterval(() => {
        setShown((n) => {
          if (n >= TOTAL) {
            clearInterval(id);
            return n;
          }
          return n + 1;
        });
      }, CHAR_MS);
    }, DELAY_MS);
    return () => {
      clearTimeout(start);
      clearInterval(id);
    };
  }, []);

  let cut = shown;
  return (
    <p className={clsx("text-xs text-muted", className)} aria-live="polite">
      {SEGMENTS.map((seg, i) => {
        const text = seg.text.slice(0, Math.max(0, cut));
        cut -= seg.text.length;
        if (!text) return null;
        return seg.href ? (
          <a
            key={i}
            href={seg.href}
            target={seg.href.startsWith("http") ? "_blank" : undefined}
            rel="noreferrer"
            className="text-accent underline underline-offset-4 opacity-90 hover:opacity-100"
          >
            {text}
          </a>
        ) : (
          <span key={i}>{text}</span>
        );
      })}
    </p>
  );
}
