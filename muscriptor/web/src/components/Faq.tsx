/**
 * FAQ below the welcome screen. Mostly for SEO: the questions are the phrasings
 * people actually search for, and native <details> keeps collapsed answers in
 * the DOM where crawlers still see them — that's where the value is.
 *
 * The FAQPage JSON-LD is a bonus, not the point: Google stopped showing FAQ
 * rich results for general sites in 2023 (government/health domains only), so
 * it buys nothing there. Other engines and AI crawlers still read it, and it's
 * generated from the same strings, so it costs ~8 lines and can't go stale.
 */

/** Answers are single strings with `[label](href)` links and `code` spans
 *  inline, so the visible copy and the JSON-LD can't drift: both are rendered
 *  from the same text (see `renderAnswer` / `plainText`). */
const QA: { q: string; a: string }[] = [
  {
    q: "What is MuScriptor?",
    a: "MuScriptor is a local audio-to-MIDI workbench. It separates a recording into note events and instruments, then presents the result as downloadable MIDI and an interactive piano roll.",
  },
  {
    q: "Does audio leave this machine?",
    a: "No. This build talks only to the MuScriptor service running beside the page. Uploaded working files are temporary and the finished MIDI is returned directly to the browser.",
  },
  {
    q: "What audio formats can I upload?",
    a: "MP3, WAV, FLAC, OGG, M4A, AIFF, or Opus. Drop the file on the page or use the file picker.",
  },
  {
    q: "Which instruments can it transcribe?",
    a: "Voice, piano, guitars, bass, strings, brass, woodwinds, drums and many more. To improve accuracy, you can also tell it up front which instruments to expect. See the dropdown in the instrument selection for the full list. ",
  },
  {
    q: "Can I convert a song to sheet music with it?",
    a: "Indirectly: MuScriptor outputs MIDI, which you can open in MuseScore, Logic, Ableton, Guitar Pro or any notation editor and export as a score. MuScriptor transcribes pitch, onset and offset timing per instrument.",
  },
  {
    q: "Can I run it locally or use it from Python?",
    a: "Yes. `./startwithuv` opens this local studio, while `./startwithuv song.mp3` runs the single-file CLI. The setup prompt prepares either a GPU or CPU environment.",
  },
  {
    q: "How accurate is it?",
    a: "Treat the output as an editable first pass. Clear acoustic parts generally produce cleaner note boundaries than dense or heavily processed mixes, and instrument conditioning can improve difficult material.",
  },
  {
    q: "How does it work, technically?",
    a: "MuScriptor is a decoder-only transformer that reads the audio in 5-second chunks and generates a token stream describing note onsets, offsets and instruments, which is then assembled into MIDI. A major reason why it works so well is the dataset: MuScriptor is trained on 170k songs spanning classical music to heavy metal. [Read more in the paper](https://arxiv.org/abs/2607.08168).",
  },
  {
    q: "What is saved after a run?",
    a: "Only the files you explicitly download. A browser refresh clears the current workspace; the server does not create an account, cloud library, or analytics profile.",
  },
];

/** `[label](href)` or `code`. */
const TOKEN = /\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`/g;

/** Answer text → React nodes, with the two inline forms above marked up. */
function renderAnswer(a: string) {
  const out = [];
  let last = 0;
  for (const m of a.matchAll(TOKEN)) {
    if (m.index > last) out.push(a.slice(last, m.index));
    out.push(
      m[1] ? (
        <a
          key={m.index}
          href={m[2]}
          target="_blank"
          rel="noreferrer"
          className="text-accent underline underline-offset-4"
        >
          {m[1]}
        </a>
      ) : (
        <code key={m.index} className="font-mono text-[0.9em] text-content">
          {m[3]}
        </code>
      ),
    );
    last = m.index + m[0].length;
  }
  out.push(a.slice(last));
  return out;
}

/** Same text with the markup stripped, for the JSON-LD. */
const plainText = (a: string) =>
  a.replace(TOKEN, (_m, label, _href, code) => label ?? code);

export function Faq() {
  return (
    <section className="mx-auto max-w-3xl px-7 pb-16">
      <h2 className="mb-4 text-3xl font-bold leading-none text-white">
        Frequently asked questions
      </h2>
      <div className="border-t border-line">
        {QA.map(({ q, a }) => (
          <details key={q} className="group border-b border-line">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 py-4 text-base font-semibold text-content marker:content-none hover:text-accent">
              <h3 className="m-0 text-base font-semibold">{q}</h3>
              <span
                className="shrink-0 text-muted transition-transform group-open:rotate-90"
                aria-hidden="true"
              >
                ›
              </span>
            </summary>
            <p className="m-0 pb-4 pr-8 text-base leading-relaxed text-muted">
              {renderAnswer(a)}
            </p>
          </details>
        ))}
      </div>
      <script
        type="application/ld+json"
        // Static, locally-authored strings — no user input reaches this.
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "FAQPage",
            mainEntity: QA.map(({ q, a }) => ({
              "@type": "Question",
              name: q,
              acceptedAnswer: { "@type": "Answer", text: plainText(a) },
            })),
          }),
        }}
      />
    </section>
  );
}
