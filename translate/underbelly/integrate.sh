#!/usr/bin/env bash
set -Eeuo pipefail

readonly UNDERBELLY_VERSION="1"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CONFIG_FILE="${UNDERBELLY_CONFIG:-$SCRIPT_DIR/config.env}"
readonly STATE_ROOT="${XDG_STATE_HOME:-$HOME/.local/state}/mm-tools-underbelly"

usage() {
  printf '%s\n' \
    "Usage: ./integrate.sh [--dry-run|--verify]" \
    "" \
    "With no arguments, validates every target, installs --ml support into" \
    "Firecrawl CLI, Firecrawl /v2/search and /v2/scrape, and AnyDoc, then" \
    "rebuilds/restarts the configured Firecrawl Docker API." \
    "" \
    "  --dry-run  Validate paths, source anchors, and the translator; write nothing" \
    "  --verify   Verify an existing integration and its live service connections"
}

MODE=install
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry-run ;;
  --verify) MODE=verify ;;
  -h|--help) usage; exit 0 ;;
  *) printf 'underbelly: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'underbelly: expected at most one argument\n' >&2
  exit 2
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  printf 'underbelly: config not found: %s\n' "$CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${ANYDOC_INSTALL_LOCATION:?missing ANYDOC_INSTALL_LOCATION in config}"
: "${FIRECRAWL_CLI_INSTALL_LOCATION:?missing FIRECRAWL_CLI_INSTALL_LOCATION in config}"
: "${FIRECRAWL_SERVICE_IS_DOCKER:?missing FIRECRAWL_SERVICE_IS_DOCKER in config}"
: "${FIRECRAWL_INSTALL_LOCATION:?missing FIRECRAWL_INSTALL_LOCATION in config}"
: "${NATIVE_LANGUAGE:?missing NATIVE_LANGUAGE in config}"
: "${TRANSLATE_HTTP_SERVICE_PORT:?missing TRANSLATE_HTTP_SERVICE_PORT in config}"

case "$FIRECRAWL_SERVICE_IS_DOCKER" in
  true|false) ;;
  *) printf 'underbelly: FIRECRAWL_SERVICE_IS_DOCKER must be true or false\n' >&2; exit 1 ;;
esac
if [[ ! "$TRANSLATE_HTTP_SERVICE_PORT" =~ ^[0-9]+$ ]] ||
   (( TRANSLATE_HTTP_SERVICE_PORT < 1 || TRANSLATE_HTTP_SERVICE_PORT > 65535 )); then
  printf 'underbelly: TRANSLATE_HTTP_SERVICE_PORT must be between 1 and 65535\n' >&2
  exit 1
fi

readonly HOST_TRANSLATE_URL="http://127.0.0.1:${TRANSLATE_HTTP_SERVICE_PORT}"
if [[ "$FIRECRAWL_SERVICE_IS_DOCKER" == true ]]; then
  readonly SERVICE_TRANSLATE_URL="http://host.docker.internal:${TRANSLATE_HTTP_SERVICE_PORT}"
else
  readonly SERVICE_TRANSLATE_URL="$HOST_TRANSLATE_URL"
fi

for command_name in python3 bun curl sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'underbelly: required command is missing: %s\n' "$command_name" >&2
    exit 1
  }
done

readonly CLI_INDEX="$FIRECRAWL_CLI_INSTALL_LOCATION/dist/index.js"
readonly CLI_OPTIONS="$FIRECRAWL_CLI_INSTALL_LOCATION/dist/utils/options.js"
readonly CLI_SEARCH="$FIRECRAWL_CLI_INSTALL_LOCATION/dist/commands/search.js"
readonly CLI_SCRAPE="$FIRECRAWL_CLI_INSTALL_LOCATION/dist/commands/scrape.js"
readonly CLI_SEARCH_TYPES="$FIRECRAWL_CLI_INSTALL_LOCATION/dist/types/search.d.ts"
readonly CLI_SCRAPE_TYPES="$FIRECRAWL_CLI_INSTALL_LOCATION/dist/types/scrape.d.ts"
readonly ANYDOC_CLI="$ANYDOC_INSTALL_LOCATION/cli.js"
readonly ANYDOC_CORE="$ANYDOC_INSTALL_LOCATION/underbelly.cjs"
readonly FIRECRAWL_ROUTES="$FIRECRAWL_INSTALL_LOCATION/apps/api/src/routes/v2.ts"
readonly FIRECRAWL_CORE="$FIRECRAWL_INSTALL_LOCATION/apps/api/src/lib/underbelly.ts"

for required_path in \
  "$CLI_INDEX" "$CLI_OPTIONS" "$CLI_SEARCH" "$CLI_SCRAPE" \
  "$CLI_SEARCH_TYPES" "$CLI_SCRAPE_TYPES" "$ANYDOC_CLI" \
  "$FIRECRAWL_ROUTES" "$FIRECRAWL_INSTALL_LOCATION/docker-compose.yaml"; do
  if [[ ! -f "$required_path" ]]; then
    printf 'underbelly: required target not found: %s\n' "$required_path" >&2
    exit 1
  fi
done

health_json="$(curl --fail --silent --show-error --max-time 10 "$HOST_TRANSLATE_URL/health")"
python3 - "$health_json" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if payload.get("status") != "ok" or not payload.get("loaded"):
    raise SystemExit("underbelly: translation service is not healthy and loaded")
if payload.get("cloud") is not False:
    raise SystemExit("underbelly: translation service did not report local-only mode")
PY

export UB_CLI_INDEX="$CLI_INDEX"
export UB_CLI_OPTIONS="$CLI_OPTIONS"
export UB_CLI_SEARCH="$CLI_SEARCH"
export UB_CLI_SCRAPE="$CLI_SCRAPE"
export UB_CLI_SEARCH_TYPES="$CLI_SEARCH_TYPES"
export UB_CLI_SCRAPE_TYPES="$CLI_SCRAPE_TYPES"
export UB_ANYDOC_CLI="$ANYDOC_CLI"
export UB_ANYDOC_CORE="$ANYDOC_CORE"
export UB_FIRECRAWL_ROUTES="$FIRECRAWL_ROUTES"
export UB_FIRECRAWL_CORE="$FIRECRAWL_CORE"
export UB_HOST_TRANSLATE_URL="$HOST_TRANSLATE_URL"
export UB_SERVICE_TRANSLATE_URL="$SERVICE_TRANSLATE_URL"
export UB_NATIVE_LANGUAGE="$NATIVE_LANGUAGE"
export UB_WRITE=$([[ "$MODE" == install ]] && printf 1 || printf 0)

run_patcher() {
python3 <<'PY'
from __future__ import annotations

import json
import os
import re
from pathlib import Path

VERSION = "1"
WRITE = os.environ["UB_WRITE"] == "1"
NATIVE = os.environ["UB_NATIVE_LANGUAGE"].strip().lower()
HOST_URL = os.environ["UB_HOST_TRANSLATE_URL"].rstrip("/")
SERVICE_URL = os.environ["UB_SERVICE_TRANSLATE_URL"].rstrip("/")

LANGUAGE_ALIASES = {
    "en": "en", "english": "en",
    "cn": "zh", "zh": "zh", "zh-cn": "zh", "chinese": "zh",
    "in": "hi", "hi": "hi", "indian": "hi", "hindi": "hi",
    "es": "es", "spanish": "es",
    "fr": "fr", "french": "fr",
    "de": "de", "german": "de",
    "rs": "ru", "ru": "ru", "russian": "ru",
    "kr": "ko", "ko": "ko", "korean": "ko",
    "jp": "ja", "ja": "ja", "japanese": "ja",
    "ar": "ar", "arabic": "ar",
    "pt": "pt", "portuguese": "pt",
    "it": "it", "italian": "it",
    "nl": "nl", "dutch": "nl",
    "pl": "pl", "polish": "pl",
    "tr": "tr", "turkish": "tr",
    "uk": "uk", "ukrainian": "uk",
    "vi": "vi", "vietnamese": "vi",
    "th": "th", "thai": "th",
    "id": "id", "indonesian": "id",
    "bn": "bn", "bengali": "bn",
}
if NATIVE not in set(LANGUAGE_ALIASES.values()) and not re.fullmatch(
    r"[a-z]{2,3}(?:-[a-z0-9]+)?", NATIVE
):
    raise SystemExit(f"underbelly: unsupported NATIVE_LANGUAGE: {NATIVE}")


def marked(identifier: str, body: str, comment: str = "//") -> str:
    return (
        f"{comment} UNDERBELLY-BEGIN:{identifier}:v{VERSION}\n"
        f"{body.rstrip()}\n"
        f"{comment} UNDERBELLY-END:{identifier}\n"
    )


def upsert_marked(text: str, identifier: str, body: str, *, anchor: str,
                  where: str = "after", comment: str = "//") -> str:
    block = marked(identifier, body, comment)
    begin = re.escape(f"{comment} UNDERBELLY-BEGIN:{identifier}:")
    end = re.escape(f"{comment} UNDERBELLY-END:{identifier}")
    pattern = re.compile(begin + r"v\d+\n.*?" + end + r"\n?", re.DOTALL)
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise RuntimeError(f"duplicate installed blocks for {identifier}")
    if matches:
        return pattern.sub(block, text, count=1)
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"source anchor for {identifier} matched {count} times; target version is unsupported"
        )
    if where == "after":
        return text.replace(anchor, anchor + "\n" + block, 1)
    if where == "before":
        return text.replace(anchor, block + anchor, 1)
    raise ValueError(where)


CORE = r'''
const UNDERBELLY_VERSION = "1";
const TRANSLATE_BASE_URL = __TRANSLATE_BASE_URL__;
const NATIVE_LANGUAGE = __NATIVE_LANGUAGE__;
const SEARCH_EXTRA_RESULTS = 9;
const TRANSLATION_CHUNK_CHARACTERS = 400;
const REQUEST_TIMEOUT_MS = 120000;
const DEFAULT_SEARCH_LANGUAGES = Object.freeze(["zh", "hi", "es", "fr", "de", "ru", "ko", "ja", "ar"]);
const LANGUAGE_ALIASES = Object.freeze({
  en: "en", english: "en",
  cn: "zh", zh: "zh", "zh-cn": "zh", chinese: "zh",
  in: "hi", hi: "hi", indian: "hi", hindi: "hi",
  es: "es", spanish: "es", fr: "fr", french: "fr",
  de: "de", german: "de", rs: "ru", ru: "ru", russian: "ru",
  kr: "ko", ko: "ko", korean: "ko", jp: "ja", ja: "ja", japanese: "ja",
  ar: "ar", arabic: "ar", pt: "pt", portuguese: "pt",
  it: "it", italian: "it", nl: "nl", dutch: "nl",
  pl: "pl", polish: "pl", tr: "tr", turkish: "tr",
  uk: "uk", ukrainian: "uk", vi: "vi", vietnamese: "vi",
  th: "th", thai: "th", id: "id", indonesian: "id",
  bn: "bn", bengali: "bn"
});

class UnderbellyRemoteError extends Error {
  constructor(message) {
    super(message);
    this.name = "UnderbellyRemoteError";
    this.fatal = true;
  }
}

const detectionCache = new Map();
const translationCache = new Map();

function normalizeLanguage(value) {
  const key = String(value ?? "").trim().toLowerCase();
  const normalized = LANGUAGE_ALIASES[key];
  if (normalized) return normalized;
  if (/^[a-z]{2,3}(?:-[a-z0-9]+)?$/u.test(key)) return key;
  throw new Error(`unsupported multilingual language '${value}'`);
}

function parseMlSpec(value, purpose = "content") {
  if (value === undefined || value === null || value === false) return null;
  let raw = [];
  if (value !== true) {
    raw = Array.isArray(value) ? value : String(value).split(",");
    raw = raw.map(item => String(item).trim()).filter(Boolean);
  }
  let languages = [];
  for (const item of raw) {
    const language = normalizeLanguage(item);
    if (!languages.includes(language)) languages.push(language);
  }
  if (languages.length > 9) throw new Error("--ml accepts at most 9 languages");
  if (purpose === "search" && languages.length === 0) {
    languages = [...DEFAULT_SEARCH_LANGUAGES];
  }
  return {
    requested: true,
    allSourceLanguages: purpose === "content" && languages.length === 0,
    languages
  };
}

function createStats() {
  return {
    translatedSegments: 0,
    skippedSegments: 0,
    preservedSegments: 0,
    warnings: new Array()
  };
}

async function postTranslator(path, payload) {
  let response;
  try {
    response = await fetch(`${TRANSLATE_BASE_URL}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
    });
  } catch (error) {
    throw new UnderbellyRemoteError(`translator request failed: ${error.message}`);
  }
  const body = await response.text();
  let parsed;
  try {
    parsed = body ? JSON.parse(body) : {};
  } catch {
    throw new UnderbellyRemoteError(`translator returned non-JSON HTTP ${response.status}`);
  }
  if (!response.ok) {
    throw new UnderbellyRemoteError(
      `translator HTTP ${response.status}: ${parsed.error || body.slice(0, 240)}`
    );
  }
  return parsed;
}

function scriptHint(text) {
  if (/\p{Script=Hangul}/u.test(text)) return "ko";
  if (/[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(text)) return "ja";
  if (/\p{Script=Arabic}/u.test(text)) return "ar";
  if (/\p{Script=Cyrillic}/u.test(text)) return "ru";
  if (/\p{Script=Devanagari}/u.test(text)) return "hi";
  if (/\p{Script=Han}/u.test(text)) return "zh";
  return null;
}

function translatableText(text) {
  const compact = String(text ?? "").trim();
  if (!compact || !/\p{L}/u.test(compact)) return false;
  if (/^(?:https?:\/\/|data:|mailto:)/i.test(compact)) return false;
  return true;
}

async function detectLanguage(text) {
  const sample = String(text).trim().slice(0, 1200);
  if (!sample) return NATIVE_LANGUAGE;
  const cacheKey = sample;
  if (detectionCache.has(cacheKey)) return detectionCache.get(cacheKey);
  const hint = scriptHint(sample);
  let language = hint && hint !== "zh" ? hint : null;
  if (language === null) {
    try {
      const detected = await postTranslator("/detect", { text: sample });
      language = normalizeLanguage(detected.language);
      if (hint === "zh" && language !== "zh" && language !== "ja") language = "zh";
    } catch (error) {
      if (error && error.fatal) {
        const fallback = hint;
        if (!fallback) throw error;
        language = fallback;
      } else {
        throw error;
      }
    }
  }
  detectionCache.set(cacheKey, language);
  return language;
}

function splitForTranslation(text, maximum = TRANSLATION_CHUNK_CHARACTERS) {
  const points = Array.from(text);
  if (points.length <= maximum) return [text];
  const chunks = [];
  let offset = 0;
  while (offset < points.length) {
    let end = Math.min(offset + maximum, points.length);
    if (end < points.length) {
      const floor = offset + Math.floor(maximum * 0.55);
      for (let index = end; index > floor; index -= 1) {
        if (/\s|[.!?。！？;；]/u.test(points[index - 1])) {
          end = index;
          break;
        }
      }
    }
    chunks.push(points.slice(offset, end).join(""));
    offset = end;
  }
  return chunks;
}

async function translateChunk(text, spec, stats, forcedTarget = null, forcedSource = null) {
  const match = String(text).match(/^(\s*)([\s\S]*?\S)(\s*)$/u);
  if (!match || !translatableText(match[2])) {
    stats.skippedSegments += 1;
    return text;
  }
  const [, leading, body, trailing] = match;
  const source = forcedSource || await detectLanguage(body);
  const target = forcedTarget || NATIVE_LANGUAGE;
  if (source === target) {
    stats.skippedSegments += 1;
    return text;
  }
  if (!forcedTarget && spec && !spec.allSourceLanguages &&
      spec.languages.length > 0 && !spec.languages.includes(source)) {
    stats.skippedSegments += 1;
    return text;
  }
  const cacheKey = `${source}\u0000${target}\u0000${body}`;
  let translated = translationCache.get(cacheKey);
  if (translated === undefined) {
    const payload = await postTranslator("/translate", {
      text: body,
      source_language: source,
      target_language: target,
      max_tokens: 512
    });
    translated = String(payload.translation ?? "").trim();
    if (!translated) throw new UnderbellyRemoteError("translator returned an empty translation");
    if (!body.includes("\n")) translated = translated.replace(/[\r\n]+/g, " ").trim();
    translationCache.set(cacheKey, translated);
  }
  if (translated !== body) stats.translatedSegments += 1;
  else stats.skippedSegments += 1;
  return `${leading}${translated}${trailing}`;
}

async function translatePlain(text, mlValue = true, options = {}) {
  const spec = options.spec || parseMlSpec(mlValue, "content") || parseMlSpec(true, "content");
  const stats = options.stats || createStats();
  const chunks = splitForTranslation(String(text));
  const output = [];
  for (const chunk of chunks) {
    output.push(await translateChunk(
      chunk,
      spec,
      stats,
      options.targetLanguage || null,
      options.sourceLanguage || null
    ));
  }
  return options.returnObject ? { text: output.join(""), stats } : output.join("");
}

const IMMUTABLE_RE_SOURCE =
  /(`+[^`\n]*`+|\${1,2}[^$\n]*\${1,2}|!\[[^\]\n]*\]\([^\)\n]*\)|\]\([^\)\n]*\)|https?:\/\/[^\s<>()\[\]"']+|<[^>\n]+>|&[#A-Za-z0-9]+;|\\.|[\[\]|#*_~>])/u.source;

function immutableTokens(text) {
  return String(text).match(new RegExp(IMMUTABLE_RE_SOURCE, "gu")) || [];
}

function urls(text) {
  return String(text).match(/https?:\/\/[^\s<>()\[\]"']+/gu) || [];
}

function newlineSequence(text) {
  return String(text).match(/\r\n|\n|\r/g) || [];
}

function linePrefix(line) {
  return line.match(/^(\s*(?:(?:#{1,6}|>|[-+*]|\d+[.)])\s+))/u)?.[1] || "";
}

function lineSignature(line) {
  return JSON.stringify({
    prefix: linePrefix(line),
    immutable: immutableTokens(line),
    urls: urls(line),
    pipes: (line.match(/\|/g) || []).length
  });
}

function splitImmutable(text) {
  const regex = new RegExp(IMMUTABLE_RE_SOURCE, "gu");
  const parts = [];
  let start = 0;
  for (const match of text.matchAll(regex)) {
    if (match.index > start) parts.push({ immutable: false, text: text.slice(start, match.index) });
    parts.push({ immutable: true, text: match[0] });
    start = match.index + match[0].length;
  }
  if (start < text.length) parts.push({ immutable: false, text: text.slice(start) });
  return parts;
}

async function translateMutablePart(text, spec, stats, conservative) {
  if (!conservative) return translatePlain(text, true, { spec, stats });
  const pieces = text.split(/([,;:，；：.!?。！？]+)/u);
  const output = [];
  for (let index = 0; index < pieces.length; index += 1) {
    if (index % 2 === 1) output.push(pieces[index]);
    else output.push(await translatePlain(pieces[index], true, { spec, stats }));
  }
  return output.join("");
}

async function translateMarkdownLine(line, spec, stats, conservative = false) {
  const prefix = linePrefix(line);
  const body = line.slice(prefix.length);
  const output = [prefix];
  for (const part of splitImmutable(body)) {
    output.push(part.immutable
      ? part.text
      : await translateMutablePart(part.text, spec, stats, conservative));
  }
  const candidate = output.join("");
  if (lineSignature(line) !== lineSignature(candidate)) {
    const error = new Error("Markdown line structure changed during translation");
    error.validation = true;
    throw error;
  }
  return candidate;
}

function fencedBlocks(text) {
  const lines = String(text).split(/(?<=\n)/u);
  const blocks = [];
  let marker = null;
  let buffer = "";
  for (const line of lines) {
    const bare = line.replace(/[\r\n]+$/u, "");
    const match = bare.match(/^\s{0,3}(`{3,}|~{3,})/u);
    if (!marker && match) {
      marker = match[1][0];
      buffer = line;
    } else if (marker) {
      buffer += line;
      if (match && match[1][0] === marker) {
        blocks.push(buffer);
        marker = null;
        buffer = "";
      }
    }
  }
  if (buffer) blocks.push(buffer);
  return blocks;
}

function markdownSignature(text) {
  return JSON.stringify({
    newlines: newlineSequence(text),
    urls: urls(text),
    immutable: immutableTokens(text),
    fences: fencedBlocks(text)
  });
}

async function translateMarkdown(markdown, mlValue = true) {
  const spec = parseMlSpec(mlValue, "content") || parseMlSpec(true, "content");
  const stats = createStats();
  const source = String(markdown);
  const pieces = source.split(/(\r\n|\n|\r)/u);
  const output = [];
  let fence = null;
  let frontMatter = false;
  let htmlComment = false;
  let blockedHtml = null;
  let mathFence = false;
  let lineNumber = 0;

  for (let index = 0; index < pieces.length; index += 2) {
    const line = pieces[index];
    const newline = pieces[index + 1] || "";
    lineNumber += 1;
    const fenceMatch = line.match(/^\s{0,3}(`{3,}|~{3,})/u);
    if (fence) {
      output.push(line, newline);
      stats.preservedSegments += 1;
      if (fenceMatch && fenceMatch[1][0] === fence) fence = null;
      continue;
    }
    if (fenceMatch) {
      fence = fenceMatch[1][0];
      output.push(line, newline);
      stats.preservedSegments += 1;
      continue;
    }
    if (lineNumber === 1 && /^---\s*$/u.test(line)) frontMatter = true;
    if (frontMatter) {
      output.push(line, newline);
      stats.preservedSegments += 1;
      if (lineNumber > 1 && /^(?:---|\.\.\.)\s*$/u.test(line)) frontMatter = false;
      continue;
    }
    if (mathFence || /^\s*\$\$\s*$/u.test(line)) {
      output.push(line, newline);
      stats.preservedSegments += 1;
      if (/^\s*\$\$\s*$/u.test(line)) mathFence = !mathFence;
      continue;
    }
    if (htmlComment || line.includes("<!--")) {
      output.push(line, newline);
      stats.preservedSegments += 1;
      if (line.includes("<!--") && !line.includes("-->")) htmlComment = true;
      if (line.includes("-->")) htmlComment = false;
      continue;
    }
    const openingBlocked = line.match(/^\s*<(script|style|pre|code)(?:\s|>)/iu);
    if (blockedHtml || openingBlocked) {
      if (openingBlocked) blockedHtml = openingBlocked[1].toLowerCase();
      output.push(line, newline);
      stats.preservedSegments += 1;
      if (blockedHtml && new RegExp(`</${blockedHtml}\\s*>`, "iu").test(line)) blockedHtml = null;
      continue;
    }
    if (/^(?:\t| {4})/u.test(line) ||
        /^\s*(?:\|?\s*:?-{3,}:?\s*)+\|?\s*$/u.test(line) ||
        /^\s*(?:-{3,}|_{3,}|\*{3,})\s*$/u.test(line) ||
        /^\s*\[[^\]]+\]:\s*\S+/u.test(line)) {
      output.push(line, newline);
      stats.preservedSegments += 1;
      continue;
    }
    try {
      output.push(await translateMarkdownLine(line, spec, stats, false), newline);
    } catch (error) {
      if (error && error.fatal) throw error;
      try {
        output.push(await translateMarkdownLine(line, spec, stats, true), newline);
      } catch (retryError) {
        if (retryError && retryError.fatal) throw retryError;
        output.push(line, newline);
        stats.preservedSegments += 1;
        stats.warnings.push(`line ${lineNumber} was preserved after Markdown validation`);
      }
    }
  }
  const text = output.join("");
  if (markdownSignature(source) !== markdownSignature(text)) {
    throw new Error("Markdown structural validation failed after reconstruction");
  }
  return { text, stats };
}

function htmlTags(text) {
  return String(text).match(/<!--[\s\S]*?-->|<![^>]*>|<\/?[A-Za-z][^>]*>/g) || [];
}

async function translateHtml(html, mlValue = true) {
  const spec = parseMlSpec(mlValue, "content") || parseMlSpec(true, "content");
  const stats = createStats();
  const source = String(html);
  const tagRegex = /<!--[\s\S]*?-->|<![^>]*>|<\/?[A-Za-z][^>]*>/g;
  const output = [];
  const blocked = [];
  let start = 0;
  for (const match of source.matchAll(tagRegex)) {
    const textNode = source.slice(start, match.index);
    if (blocked.length || !translatableText(textNode)) {
      output.push(textNode);
      if (textNode) stats.preservedSegments += 1;
    } else {
      const linePieces = textNode.split(/(\r\n|\n|\r)/u);
      for (let index = 0; index < linePieces.length; index += 2) {
        output.push(await translatePlain(linePieces[index], true, { spec, stats }));
        if (linePieces[index + 1]) output.push(linePieces[index + 1]);
      }
    }
    const tag = match[0];
    output.push(tag);
    const open = tag.match(/^<(script|style|pre|code)(?:\s|>)/iu);
    const close = tag.match(/^<\/(script|style|pre|code)\s*>/iu);
    if (open) blocked.push(open[1].toLowerCase());
    if (close && blocked[blocked.length - 1] === close[1].toLowerCase()) blocked.pop();
    start = match.index + tag.length;
  }
  const tail = source.slice(start);
  output.push(blocked.length ? tail : await translatePlain(tail, true, { spec, stats }));
  const text = output.join("");
  if (JSON.stringify(htmlTags(source)) !== JSON.stringify(htmlTags(text)) ||
      JSON.stringify(urls(source)) !== JSON.stringify(urls(text))) {
    throw new Error("HTML structural validation failed after reconstruction");
  }
  return { text, stats };
}

function mergeStats(target, source) {
  target.translatedSegments += source.translatedSegments;
  target.skippedSegments += source.skippedSegments;
  target.preservedSegments += source.preservedSegments;
  target.warnings.push(...source.warnings);
}

function looksLikeOpaqueData(value) {
  const text = String(value).trim();
  if (/^(?:https?:\/\/|data:|[A-Za-z][A-Za-z0-9+.-]*:\/\/)/i.test(text)) return true;
  if (text.length > 160 && /^[A-Za-z0-9+/=_-]+$/u.test(text)) return true;
  return false;
}

async function translateJsonValue(value, spec, stats) {
  if (typeof value === "string") {
    if (looksLikeOpaqueData(value)) return value;
    return translatePlain(value, true, { spec, stats });
  }
  if (Array.isArray(value)) {
    const result = [];
    for (const item of value) result.push(await translateJsonValue(item, spec, stats));
    return result;
  }
  if (value && typeof value === "object") {
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = await translateJsonValue(item, spec, stats);
    }
    return result;
  }
  return value;
}

async function translateDocument(document, mlValue = true) {
  const spec = parseMlSpec(mlValue, "content") || parseMlSpec(true, "content");
  const stats = createStats();
  const output = { ...document };
  if (typeof output.markdown === "string") {
    const translated = await translateMarkdown(output.markdown, mlValue);
    output.markdown = translated.text;
    mergeStats(stats, translated.stats);
  }
  for (const field of ["html", "rawHtml"]) {
    if (typeof output[field] === "string") {
      const translated = await translateHtml(output[field], mlValue);
      output[field] = translated.text;
      mergeStats(stats, translated.stats);
    }
  }
  for (const field of ["summary", "content"]) {
    if (typeof output[field] === "string") {
      output[field] = await translatePlain(output[field], true, { spec, stats });
    }
  }
  if (output.json !== undefined) output.json = await translateJsonValue(output.json, spec, stats);
  const metadata = output.metadata && typeof output.metadata === "object"
    ? { ...output.metadata }
    : {};
  for (const field of ["title", "description"]) {
    if (typeof metadata[field] === "string") {
      metadata[field] = await translatePlain(metadata[field], true, { spec, stats });
    }
  }
  metadata.underbelly = {
    version: UNDERBELLY_VERSION,
    targetLanguage: NATIVE_LANGUAGE,
    translatedSegments: stats.translatedSegments,
    skippedSegments: stats.skippedSegments,
    preservedSegments: stats.preservedSegments,
    warnings: stats.warnings
  };
  output.metadata = metadata;
  return { document: output, stats };
}
'''

MIDDLEWARE = r'''
function canonicalUrl(value) {
  try {
    const parsed = new URL(String(value));
    parsed.hash = "";
    parsed.hostname = parsed.hostname.toLowerCase();
    if (parsed.pathname !== "/") parsed.pathname = parsed.pathname.replace(/\/+$/u, "");
    return parsed.toString();
  } catch {
    return String(value ?? "");
  }
}

function resultUrl(result) {
  return result && (result.url || result.imageUrl || result.sourceURL) || "";
}

function resultGroups(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return [];
  return ["web", "news", "images", "developer"]
    .filter(group => Array.isArray(data[group]))
    .map(group => [group, data[group]]);
}

async function translateSearchItem(item, sourceLanguage) {
  const output = { ...item, mlLanguage: sourceLanguage };
  const spec = parseMlSpec(true, "content");
  const stats = createStats();
  for (const field of ["title", "description", "snippet"]) {
    if (typeof output[field] === "string") {
      output[field] = await translatePlain(output[field], true, { spec, stats });
    }
  }
  if (typeof output.markdown === "string") {
    output.markdown = (await translateMarkdown(output.markdown, true)).text;
  }
  for (const field of ["html", "rawHtml"]) {
    if (typeof output[field] === "string") {
      output[field] = (await translateHtml(output[field], true)).text;
    }
  }
  if (output.metadata && typeof output.metadata === "object") {
    output.metadata = { ...output.metadata };
    for (const field of ["title", "description"]) {
      if (typeof output.metadata[field] === "string") {
        output.metadata[field] = await translatePlain(
          output.metadata[field], true, { spec, stats }
        );
      }
    }
  }
  return output;
}

async function translateQuery(query, targetLanguage) {
  if (targetLanguage === NATIVE_LANGUAGE) return query;
  const stats = createStats();
  return translatePlain(query, true, {
    spec: parseMlSpec(true, "content"),
    stats,
    sourceLanguage: NATIVE_LANGUAGE,
    targetLanguage
  });
}

async function localSearch(body, authorization) {
  const port = process.env.PORT || "3002";
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (authorization) headers.authorization = authorization;
  const response = await fetch(`http://127.0.0.1:${port}/v2/search`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(Math.max(REQUEST_TIMEOUT_MS, Number(body.timeout || 0) + 10000))
  });
  const raw = await response.text();
  let payload;
  try { payload = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`multilingual search returned non-JSON HTTP ${response.status}`); }
  if (!response.ok || !payload.success) {
    throw new Error(`multilingual search HTTP ${response.status}: ${payload.error || raw.slice(0, 240)}`);
  }
  return payload;
}

async function augmentSearch(payload, requestBody, mlValue, authorization) {
  const spec = parseMlSpec(mlValue, "search");
  if (!spec || !payload?.success || !payload.data) return payload;
  const languages = spec.languages;
  const quotient = Math.floor(SEARCH_EXTRA_RESULTS / languages.length);
  const remainder = SEARCH_EXTRA_RESULTS % languages.length;
  const allocations = languages.map((_, index) => quotient + (index < remainder ? 1 : 0));
  const seen = new Set();
  for (const [, items] of resultGroups(payload.data)) {
    for (const item of items) seen.add(canonicalUrl(resultUrl(item)));
  }
  const warnings: string[] = [];
  let added = 0;
  let extraCredits = 0;

  for (let languageIndex = 0; languageIndex < languages.length; languageIndex += 1) {
    const language = languages[languageIndex];
    const allocation = allocations[languageIndex];
    const translatedQuery = await translateQuery(String(requestBody.query || ""), language);
    const branchBody = { ...requestBody };
    delete branchBody.ml;
    branchBody.query = translatedQuery;
    branchBody.lang = language;
    branchBody.limit = Math.min(100, allocation + 6);
    let branch = await localSearch(branchBody, authorization);
    extraCredits += Number(branch.creditsUsed || 0);

    let candidates = resultGroups(branch.data).flatMap(([group, items]) =>
      items.map(item => ({ group, item }))
    );
    if (candidates.length === 0 && translatedQuery !== requestBody.query) {
      branchBody.query = requestBody.query;
      branch = await localSearch(branchBody, authorization);
      extraCredits += Number(branch.creditsUsed || 0);
      candidates = resultGroups(branch.data).flatMap(([group, items]) =>
        items.map(item => ({ group, item }))
      );
    }

    let branchAdded = 0;
    for (const candidate of candidates) {
      if (branchAdded >= allocation) break;
      const key = canonicalUrl(resultUrl(candidate.item));
      if (!key || seen.has(key)) continue;
      seen.add(key);
      const translated = await translateSearchItem(candidate.item, language);
      if (!Array.isArray(payload.data[candidate.group])) payload.data[candidate.group] = [];
      payload.data[candidate.group].push(translated);
      branchAdded += 1;
      added += 1;
    }
    if (branchAdded < allocation) {
      warnings.push(
        `${language} produced ${branchAdded}/${allocation} unique multilingual results`
      );
    }
  }

  payload.creditsUsed = Number(payload.creditsUsed || 0) + extraCredits;
  payload.underbelly = {
    version: UNDERBELLY_VERSION,
    nativeLanguage: NATIVE_LANGUAGE,
    languages,
    requestedExtraResults: SEARCH_EXTRA_RESULTS,
    addedExtraResults: added,
    warnings
  };
  if (warnings.length) {
    payload.warning = [payload.warning, ...warnings].filter(Boolean).join("; ");
  }
  return payload;
}

async function transformScrape(payload, mlValue) {
  if (!payload?.success || !payload.data) return payload;
  const translated = await translateDocument(payload.data, mlValue);
  payload.data = translated.document;
  if (translated.stats.warnings.length) {
    payload.warning = [payload.warning, ...translated.stats.warnings]
      .filter(Boolean).join("; ");
  }
  return payload;
}

export const underbellyMiddleware = (req, res, next) => {
  if (req.method !== "POST" || (req.path !== "/search" && req.path !== "/scrape") ||
      !req.body || typeof req.body !== "object" || req.body.ml === undefined) {
    next();
    return;
  }
  const mlValue = req.body.ml;
  const requestBody = { ...req.body };
  delete req.body.ml;
  const authorization = typeof req.headers.authorization === "string"
    ? req.headers.authorization
    : undefined;
  const originalJson = res.json.bind(res);
  let intercepted = false;
  res.json = body => {
    if (intercepted) return res;
    intercepted = true;
    void (async () => {
      try {
        const transformed = req.path === "/search"
          ? await augmentSearch(body, requestBody, mlValue, authorization)
          : await transformScrape(body, mlValue);
        originalJson(transformed);
      } catch (error) {
        res.statusCode = 502;
        originalJson({
          success: false,
          error: `Underbelly multilingual processing failed: ${error.message}`
        });
      }
    })();
    return res;
  };
  next();
};

export {
  parseMlSpec,
  translateDocument,
  translateHtml,
  translateMarkdown,
  translatePlain
};
'''


def render_core(base_url: str, module_kind: str) -> str:
    rendered = CORE.replace("__TRANSLATE_BASE_URL__", json.dumps(base_url))
    rendered = rendered.replace("__NATIVE_LANGUAGE__", json.dumps(NATIVE))
    header = (
        "// Generated by ~/multimedia/translate/underbelly/integrate.sh.\n"
        "// Re-run that installer after updating the owning package.\n"
    )
    if module_kind == "cjs":
        return header + '"use strict";\n' + rendered + (
            "\nmodule.exports = { parseMlSpec, translateDocument, translateHtml, "
            "translateMarkdown, translatePlain };\n"
        )
    if module_kind == "ts":
        # The adapter intentionally shares its runtime implementation with the
        # injected CommonJS client. Docker's TypeScript build still parses and
        # emits it, while runtime validation owns the dynamic Firecrawl envelopes.
        return "// @ts-nocheck\n" + header + rendered + "\n" + MIDDLEWARE
    raise ValueError(module_kind)


paths = {key: Path(os.environ[key]) for key in [
    "UB_CLI_INDEX", "UB_CLI_OPTIONS", "UB_CLI_SEARCH", "UB_CLI_SCRAPE",
    "UB_CLI_SEARCH_TYPES", "UB_CLI_SCRAPE_TYPES", "UB_ANYDOC_CLI",
    "UB_ANYDOC_CORE", "UB_FIRECRAWL_ROUTES", "UB_FIRECRAWL_CORE",
]}

updates: dict[Path, str] = {}

text = paths["UB_CLI_INDEX"].read_text()
text = upsert_marked(
    text,
    "cli-scrape-option",
    "        .option('--ml [languages]', 'Translate scraped prose to the configured native language; optionally restrict source languages (for example: cn,rs)')",
    anchor="        .option('-Q, --query <prompt>', 'Ask a question about the page content (query format)')",
    where="before",
)
text = upsert_marked(
    text,
    "cli-search-option",
    "        .option('--ml [languages]', 'Append 9 multilingual results; optionally select up to 9 languages (for example: cn,rs)')",
    anchor="        .option('--json', 'Output as compact JSON', false)\n        .action(async (query, options) => {",
    where="before",
)
text = upsert_marked(
    text,
    "cli-search-value",
    "            ml: options.ml,",
    anchor="            pretty: options.pretty,\n        };\n        await (0, search_1.handleSearchCommand)(searchOptions);",
    where="before",
)
updates[paths["UB_CLI_INDEX"]] = text

text = paths["UB_CLI_OPTIONS"].read_text()
text = upsert_marked(
    text,
    "cli-scrape-value",
    "        ml: options.ml,",
    anchor="        redactPII: options.redactPii ?? options.redactPII,\n    };",
    where="before",
)
updates[paths["UB_CLI_OPTIONS"]] = text

text = paths["UB_CLI_SCRAPE"].read_text()
text = upsert_marked(
    text,
    "cli-scrape-request",
    "    if (options.ml !== undefined) {\n        scrapeParams.ml = options.ml;\n    }",
    anchor="    const scrapeParams = {\n        formats: resolvedFormats,\n        integration: 'cli',\n    };",
)
text = upsert_marked(
    text,
    "cli-scrape-warning",
    "    const underbellyWarnings = result.data?.metadata?.underbelly?.warnings;\n    if (Array.isArray(underbellyWarnings) && underbellyWarnings.length > 0) {\n        process.stderr.write(`Warning: ${underbellyWarnings.join('; ')}\\n`);\n    }",
    anchor="    const result = await executeScrape(options);",
)
updates[paths["UB_CLI_SCRAPE"]] = text

text = paths["UB_CLI_SEARCH"].read_text()
text = upsert_marked(
    text,
    "cli-search-request",
    "        if (options.ml !== undefined) {\n            searchParams.ml = options.ml;\n        }",
    anchor="        const searchParams = {\n            limit: options.limit,\n            integration: 'cli',\n        };",
)
text = upsert_marked(
    text,
    "cli-search-warning",
    "    if (result.warning) {\n        process.stderr.write(`Warning: ${result.warning}\\n`);\n    }",
    anchor="    if (!result.data) {\n        return;\n    }",
)
updates[paths["UB_CLI_SEARCH"]] = text

text = paths["UB_CLI_SCRAPE_TYPES"].read_text()
text = upsert_marked(
    text,
    "cli-scrape-type",
    "    /** Translate emitted prose to the configured native language. */\n    ml?: boolean | string;",
    anchor="    /** Redact personally identifiable information from returned content */",
    where="before",
)
updates[paths["UB_CLI_SCRAPE_TYPES"]] = text

text = paths["UB_CLI_SEARCH_TYPES"].read_text()
text = upsert_marked(
    text,
    "cli-search-type",
    "    /** Append nine results found through translated multilingual queries. */\n    ml?: boolean | string;",
    anchor="    /** Enable scraping of search results */",
    where="before",
)
updates[paths["UB_CLI_SEARCH_TYPES"]] = text

text = paths["UB_ANYDOC_CLI"].read_text()
text = upsert_marked(
    text,
    "anydoc-help",
    "  --ml [languages]       Translate Markdown prose to the configured native language;\\n                         optionally restrict source languages (for example: cn,rs)",
    anchor="  -f, --format <format>  Name the input format instead of detecting it:",
    where="before",
)
text = upsert_marked(
    text,
    "anydoc-args",
    "  args.ml = null",
    anchor="  let positionalOnly = false",
    where="before",
)
text = upsert_marked(
    text,
    "anydoc-option",
    "      case '--ml': {\n        if (inline !== null) {\n          args.ml = inline || true\n        } else {\n          const candidate = argv[i + 1]\n          if (candidate && !candidate.startsWith('-') && /^[A-Za-z-]+(?:,[A-Za-z-]+)*$/.test(candidate)) {\n            args.ml = candidate\n            i += 1\n          } else {\n            args.ml = true\n          }\n        }\n        break\n      }",
    anchor="      default:\n        fail(USAGE_ERROR, `unknown option '${arg}' (see anydoc --help)`)\n",
    where="before",
)
text = upsert_marked(
    text,
    "anydoc-translate",
    "  if (args.ml !== null) {\n    try {\n      const { translateMarkdown } = require('./underbelly.cjs')\n      const translated = await translateMarkdown(markdown, args.ml)\n      markdown = translated.text\n      if (translated.stats.warnings.length > 0) {\n        process.stderr.write(`Warning: ${translated.stats.warnings.join('; ')}\\n`)\n      }\n    } catch (error) {\n      fail(CONVERSION_ERROR, `multilingual translation failed: ${error.message}`)\n    }\n  }",
    anchor="  if (args.output !== null) {",
    where="before",
)
updates[paths["UB_ANYDOC_CLI"]] = text
updates[paths["UB_ANYDOC_CORE"]] = render_core(HOST_URL, "cjs")

text = paths["UB_FIRECRAWL_ROUTES"].read_text()
text = upsert_marked(
    text,
    "firecrawl-import",
    'import { underbellyMiddleware } from "../lib/underbelly";',
    anchor='import { scrapeController } from "../controllers/v2/scrape";',
)
text = upsert_marked(
    text,
    "firecrawl-middleware",
    "v2Router.use(underbellyMiddleware);",
    anchor='v2Router.use(requestTimingMiddleware("v2"));',
)
updates[paths["UB_FIRECRAWL_ROUTES"]] = text
updates[paths["UB_FIRECRAWL_CORE"]] = render_core(SERVICE_URL, "ts")

for path, content in updates.items():
    if not content.endswith("\n"):
        content += "\n"
    if WRITE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"{'updated' if WRITE else 'validated'}: {path}")
PY
}

syntax_check_js() {
  bun build --no-bundle --target=node --outfile=/dev/null "$1" >/dev/null
}

if [[ "$MODE" == verify ]]; then
  export UB_WRITE=0
  run_patcher >/dev/null
  for marker_file in \
    "$CLI_INDEX" "$CLI_OPTIONS" "$CLI_SEARCH" "$CLI_SCRAPE" \
    "$ANYDOC_CLI" "$FIRECRAWL_ROUTES"; do
    grep -q 'UNDERBELLY-BEGIN:' "$marker_file" || {
      printf 'underbelly: integration marker missing from %s\n' "$marker_file" >&2
      exit 1
    }
  done
  [[ -s "$ANYDOC_CORE" && -s "$FIRECRAWL_CORE" ]] || {
    printf 'underbelly: generated runtime module is missing\n' >&2
    exit 1
  }
  syntax_check_js "$ANYDOC_CORE"
  syntax_check_js "$CLI_INDEX"
  syntax_check_js "$CLI_OPTIONS"
  syntax_check_js "$CLI_SEARCH"
  syntax_check_js "$CLI_SCRAPE"
  if [[ "$FIRECRAWL_SERVICE_IS_DOCKER" == true ]]; then
    (
      cd "$FIRECRAWL_INSTALL_LOCATION"
      docker compose exec -T api curl --fail --silent --show-error \
        --max-time 10 "$SERVICE_TRANSLATE_URL/health" >/dev/null
    )
  fi
  printf 'underbelly: integration verified (native language: %s)\n' "$NATIVE_LANGUAGE"
  exit 0
fi

if [[ "$MODE" == dry-run ]]; then
  run_patcher
  if [[ "$FIRECRAWL_SERVICE_IS_DOCKER" == true ]]; then
    (cd "$FIRECRAWL_INSTALL_LOCATION" && docker compose config --quiet)
  fi
  printf 'underbelly: dry-run passed; no files changed\n'
  exit 0
fi

mkdir -p "$STATE_ROOT"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly BACKUP_DIR="$STATE_ROOT/backups/$RUN_ID"
readonly MANIFEST="$BACKUP_DIR/manifest.tsv"
mkdir -p "$BACKUP_DIR/files"
: > "$MANIFEST"

backup_path() {
  local label=$1
  local path=$2
  printf '%s\t%s\n' "$label" "$path" >> "$MANIFEST"
  if [[ -e "$path" ]]; then
    cp -a -- "$path" "$BACKUP_DIR/files/$label"
  else
    : > "$BACKUP_DIR/files/$label.missing"
  fi
}

backup_path cli-index "$CLI_INDEX"
backup_path cli-options "$CLI_OPTIONS"
backup_path cli-search "$CLI_SEARCH"
backup_path cli-scrape "$CLI_SCRAPE"
backup_path cli-search-types "$CLI_SEARCH_TYPES"
backup_path cli-scrape-types "$CLI_SCRAPE_TYPES"
backup_path anydoc-cli "$ANYDOC_CLI"
backup_path anydoc-core "$ANYDOC_CORE"
backup_path firecrawl-routes "$FIRECRAWL_ROUTES"
backup_path firecrawl-core "$FIRECRAWL_CORE"

MUTATED=0
DOCKER_REPLACED=0
rollback() {
  local status=${1:-$?}
  trap - ERR INT TERM
  set +e
  if (( MUTATED )); then
    printf 'underbelly: integration failed; restoring touched files\n' >&2
    while IFS=$'\t' read -r label path; do
      if [[ -f "$BACKUP_DIR/files/$label.missing" ]]; then
        rm -f -- "$path"
      else
        cp -a -- "$BACKUP_DIR/files/$label" "$path"
      fi
    done < "$MANIFEST"
    if (( DOCKER_REPLACED )) && [[ "$FIRECRAWL_SERVICE_IS_DOCKER" == true ]]; then
      (
        cd "$FIRECRAWL_INSTALL_LOCATION" || exit
        docker compose build api && docker compose up -d api
      ) >&2
    fi
  fi
  exit "$status"
}
trap 'rollback $?' ERR
trap 'rollback 130' INT
trap 'rollback 143' TERM

MUTATED=1
run_patcher

syntax_check_js "$ANYDOC_CORE"
syntax_check_js "$CLI_INDEX"
syntax_check_js "$CLI_OPTIONS"
syntax_check_js "$CLI_SEARCH"
syntax_check_js "$CLI_SCRAPE"


if [[ "$FIRECRAWL_SERVICE_IS_DOCKER" == true ]]; then
  DOCKER_REPLACED=1
  (
    trap - ERR
    cd "$FIRECRAWL_INSTALL_LOCATION"
    docker compose build api
    docker compose up -d api
  )
  for attempt in $(seq 1 60); do
    if curl --fail --silent --max-time 3 http://127.0.0.1:3002/ >/dev/null 2>&1; then
      break
    fi
    if (( attempt == 60 )); then
      printf 'underbelly: rebuilt Firecrawl API did not become healthy\n' >&2
      false
    fi
    sleep 1
  done
  (
    cd "$FIRECRAWL_INSTALL_LOCATION"
    docker compose exec -T api curl --fail --silent --show-error \
      --max-time 10 "$SERVICE_TRANSLATE_URL/health" >/dev/null
  )
fi

printf '中文标题,说明\n测试,本地翻译\n' |
  anydoc - --format csv --ml >/dev/null

ln -sfn "$BACKUP_DIR" "$STATE_ROOT/last-successful"
trap - ERR INT TERM
printf '%s\n' \
  "underbelly: integration complete" \
  "  native language: $NATIVE_LANGUAGE" \
  "  translator:      $HOST_TRANSLATE_URL" \
  "  backup:          $BACKUP_DIR"
