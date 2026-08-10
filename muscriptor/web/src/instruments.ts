/**
 * Fuzzy matching for the conditioning instrument picker.
 *
 * The backend instrument ids (e.g. `french_horn`) are terse and don't cover the
 * many ways a musician might search — by family ("brass"), by colloquial name
 * ("rhodes", "fiddle"), or with a typo. `ALIASES` maps each id to extra search
 * terms, and `scoreInstrument` ranks a candidate against a query using those
 * terms plus a subsequence fallback for typos.
 */

/**
 * Family / category keywords, each expanded to the instrument ids it covers.
 * Kept separate from per-instrument aliases so a family can be edited in one
 * place; both are folded into the per-instrument alias lists below.
 */
const FAMILIES: Record<string, string[]> = {
  keys: ["acoustic_piano", "electric_piano", "organ"],
  keyboard: ["acoustic_piano", "electric_piano", "organ"],
  piano: ["acoustic_piano", "electric_piano"],
  guitar: [
    "acoustic_guitar",
    "clean_electric_guitar",
    "distorted_electric_guitar",
  ],
  bass: ["acoustic_bass", "electric_bass", "contrabass"],
  strings: [
    "violin",
    "viola",
    "cello",
    "contrabass",
    "orchestral_harp",
    "string_ensemble",
    "synth_strings",
  ],
  orchestra: [
    "violin",
    "viola",
    "cello",
    "contrabass",
    "orchestral_harp",
    "timpani",
    "string_ensemble",
    "french_horn",
    "trumpet",
    "trombone",
    "tuba",
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "flutes",
  ],
  brass: ["trumpet", "trombone", "tuba", "french_horn", "brass_section"],
  horns: ["trumpet", "trombone", "tuba", "french_horn", "brass_section"],
  woodwind: [
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "flutes",
    "soprano_and_alto_sax",
    "tenor_sax",
    "baritone_sax",
  ],
  woodwinds: [
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "flutes",
    "soprano_and_alto_sax",
    "tenor_sax",
    "baritone_sax",
  ],
  reed: ["oboe", "english_horn", "bassoon", "clarinet"],
  sax: ["soprano_and_alto_sax", "tenor_sax", "baritone_sax"],
  saxophone: ["soprano_and_alto_sax", "tenor_sax", "baritone_sax"],
  wind: [
    "trumpet",
    "trombone",
    "tuba",
    "french_horn",
    "brass_section",
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "flutes",
    "soprano_and_alto_sax",
    "tenor_sax",
    "baritone_sax",
  ],
  percussion: ["drums", "timpani", "chromatic_percussion"],
  synth: ["synth_strings", "synth_lead", "synth_pad"],
  vocals: ["voice"],
  vocal: ["voice"],
};

/** Per-instrument colloquial names and common spellings. */
const SPECIFIC: Record<string, string[]> = {
  acoustic_piano: ["grand piano", "grand", "upright piano"],
  electric_piano: ["rhodes", "wurlitzer", "wurli", "ep", "e-piano"],
  chromatic_percussion: [
    "vibraphone",
    "vibes",
    "marimba",
    "xylophone",
    "glockenspiel",
    "bells",
    "celesta",
  ],
  organ: ["hammond", "pipe organ", "b3"],
  acoustic_guitar: ["nylon", "steel string", "acoustic"],
  clean_electric_guitar: ["electric guitar", "clean guitar"],
  distorted_electric_guitar: [
    "electric guitar",
    "distortion",
    "overdrive",
    "distorted guitar",
  ],
  acoustic_bass: ["upright bass", "double bass", "stand-up bass"],
  electric_bass: ["bass guitar", "e-bass", "electric bass"],
  violin: ["fiddle"],
  cello: ["violoncello"],
  contrabass: ["double bass", "upright bass"],
  orchestral_harp: ["harp"],
  timpani: ["kettle drum", "kettledrums"],
  string_ensemble: ["string section", "strings"],
  synth_strings: ["synth strings", "string pad"],
  voice: ["vocals", "vocal", "singer", "choir", "vox"],
  orchestra_hit: ["orch hit", "stab", "hit"],
  french_horn: ["horn"],
  brass_section: ["brass", "horn section", "horns"],
  soprano_and_alto_sax: ["alto sax", "soprano sax", "alto", "soprano"],
  tenor_sax: ["tenor"],
  baritone_sax: ["bari sax", "baritone", "bari"],
  english_horn: ["cor anglais"],
  flutes: ["flute", "piccolo", "recorder"],
  synth_lead: ["lead"],
  synth_pad: ["pad"],
  drums: ["drum kit", "drumkit", "kit", "beat", "percussion"],
};

/** Final id -> alias-terms map, merging families and per-instrument names. */
const ALIASES: Record<string, string[]> = (() => {
  const out: Record<string, Set<string>> = {};
  for (const [term, ids] of Object.entries(FAMILIES)) {
    for (const id of ids) (out[id] ??= new Set()).add(term);
  }
  for (const [id, terms] of Object.entries(SPECIFIC)) {
    for (const t of terms) (out[id] ??= new Set()).add(t);
  }
  return Object.fromEntries(
    Object.entries(out).map(([id, set]) => [id, [...set]]),
  );
})();

/** Display form of an instrument id ("electric_bass" -> "electric bass"). */
export function label(name: string): string {
  return name.replace(/_/g, " ");
}

/** True if every char of `q` appears in `text` in order (typo-tolerant). */
function isSubsequence(q: string, text: string): boolean {
  let i = 0;
  for (let j = 0; j < text.length && i < q.length; j++) {
    if (text[j] === q[i]) i++;
  }
  return i === q.length;
}

/**
 * Score `name` against `query` for ranking; 0 means no match. Higher is a
 * tighter match: exact label > label prefix > label substring > alias hits >
 * subsequence (typo) fallback.
 */
export function scoreInstrument(name: string, query: string): number {
  const q = query.trim().toLowerCase();
  if (q === "") return 1;

  const text = label(name).toLowerCase();
  if (text === q) return 100;
  if (text.startsWith(q)) return 80;
  if (text.includes(q)) return 60;

  let best = 0;
  for (const alias of ALIASES[name] ?? []) {
    if (alias === q) best = Math.max(best, 55);
    else if (alias.startsWith(q)) best = Math.max(best, 45);
    else if (alias.includes(q)) best = Math.max(best, 40);
  }
  if (best > 0) return best;

  // Typo tolerance — only for queries long enough to be meaningful.
  if (q.length >= 3) {
    if (isSubsequence(q, text)) return 20;
    for (const alias of ALIASES[name] ?? []) {
      if (isSubsequence(q, alias)) return 15;
    }
  }
  return 0;
}
