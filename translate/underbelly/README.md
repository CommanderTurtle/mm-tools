# Underbelly

Underbelly is a local overlay for Firecrawl CLI, Firecrawl `/v2`, and AnyDoc. It adds multilingual search and structure-preserving translation without maintaining a Firecrawl fork or adding another HTTP endpoint. The authoritative implementation lives in `integrate.sh`; installed files are generated or marked so the overlay can be verified, replaced, and removed mechanically.

## What it changes

| Owner | Patched surface | Result |
| --- | --- | --- |
| `firecrawl-cli` | compiled scrape/search command files and their types | accepts `--ml` and forwards it to `/v2` |
| `@firecrawl/anydoc` | `cli.js` plus generated `underbelly.cjs` | translates the converted Markdown buffer before output |
| Firecrawl checkout | `apps/api/src/routes/v2.ts` plus generated `apps/api/src/lib/underbelly.ts` | intercepts only `/v2/search` and `/v2/scrape` requests containing `ml` |

Requests without `ml` follow the ordinary Firecrawl path. Underbelly does not add `/v2` routes, replace the scraper, or alter normal result limits. Search waits for Firecrawl's native result, runs additional translated queries, deduplicates their URLs, translates the selected results back to the configured native language, and appends at most nine results. Scrape waits for the native document response and translates its supported in-memory fields while preserving Markdown, HTML, links, code, math, and opaque data.

The translator is the existing local service in `../`:

```text
host CLI and AnyDoc  -> http://127.0.0.1:8176
Firecrawl container  -> http://host.docker.internal:8176
                         /health, /detect, /translate
```

Installation refuses a translator that is unloaded or reports cloud mode.

## Language contract

Content translation and multilingual search intentionally use different argument shapes:

```bash
# Search: add nine results from the default language set.
firecrawl search "query" --ml

# Search: divide the nine additions between selected search languages.
firecrawl search "query" --ml cn,rs

# Content: detect each source segment and target config.env's native language.
firecrawl scrape "https://example.com" --ml
anydoc document.pdf --ml -o document.en.md

# Content: detect the source and target Spanish.
firecrawl scrape "https://example.com" --ml es

# Content: explicitly translate English to Spanish.
firecrawl scrape "https://example.com" --ml en,es
```

The same content contract is available by sending `ml: true`, `ml: "es"`, or `ml: "en,es"` to the existing `POST /v2/scrape` endpoint.

## Configuration

Edit `config.env` when an installation path, native language, or port differs:

| Variable | Meaning |
| --- | --- |
| `ANYDOC_INSTALL_LOCATION` | installed `@firecrawl/anydoc` package directory |
| `FIRECRAWL_CLI_INSTALL_LOCATION` | installed `firecrawl-cli` package directory |
| `FIRECRAWL_INSTALL_LOCATION` | root of the real Firecrawl Git checkout containing `docker-compose.yaml` |
| `FIRECRAWL_SERVICE_IS_DOCKER` | rebuild/restart and verify the Compose `api` service when `true` |
| `NATIVE_LANGUAGE` | default translation destination and the language used to query other search languages |
| `TRANSLATE_HTTP_SERVICE_PORT` | host port of the loaded local translation service |

Paths are sourced by Bash and may use `$HOME`. Run the installer as the same unprivileged user that owns the checkout and Bun global installation.

### Independent component handling

Every command evaluates the three integrations independently. There is no reason to rebuild Firecrawl merely because a Bun global package was replaced:

| What changed upstream | What `./integrate.sh` writes | Docker API rebuild |
| --- | --- | --- |
| Firecrawl CLI only | six compiled CLI files | no |
| AnyDoc only | `cli.js` and `underbelly.cjs` | no |
| Firecrawl CLI + AnyDoc | both Bun global packages | no |
| Firecrawl checkout only | route block and generated server module | yes, when Docker mode is `true` |
| all three | all owned targets | yes, when Docker mode is `true` |
| nothing | nothing | no |

The script still reads all configured paths so it can report a complete contract. “Independent” means unchanged components are left byte-for-byte alone and only a changed Firecrawl server component triggers Compose. `FIRECRAWL_SERVICE_IS_DOCKER` describes how that server component is deployed; it is not a selector that disables the Firecrawl integration.

## Install and verify

Start the translation service, validate every current source layout without writing, install, and verify:

```bash
cd ~/multimedia/translate
./starthttp.sh

cd ~/multimedia/translate/underbelly
./integrate.sh --dry-run
./integrate.sh
./integrate.sh --verify
```

`--dry-run` is deliberately fail-closed. Every required path must exist and every absent patch block must have exactly one known upstream anchor. It also validates Compose configuration when Docker mode is enabled.

An installation then:

1. Computes every target file in memory and records which component differs.
2. Backs up all ten owned target paths under `~/.local/state/mm-tools-underbelly/backups/`.
3. Replaces existing versioned blocks or inserts new blocks at exact anchors.
4. Syntax-checks the generated and modified JavaScript and runs content/search contract assertions.
5. Rebuilds and restarts only the Firecrawl `api` service when its two files changed.
6. Verifies host-to-translator and container-to-translator connectivity and runs an AnyDoc translation smoke test.

Any write, syntax, build, startup, connectivity, or smoke-test failure restores the pre-run files. If the API image was replaced, rollback rebuilds and starts it from the restored source. A successful install updates the `last-successful` backup symlink.

Running `./integrate.sh` again is safe: matching files are reported as unchanged, and the API is not rebuilt unless the Firecrawl component actually differs.

## Updating Firecrawl, Firecrawl CLI, or AnyDoc

The cleanest Firecrawl Git update removes the overlay before pulling, then reapplies it:

```bash
cd ~/multimedia/translate/underbelly
./integrate.sh --uninstall

git -C ~/Hermes/firecrawl/firecrawl status --short
git -C ~/Hermes/firecrawl/firecrawl pull --ff-only

./integrate.sh --dry-run
./integrate.sh
./integrate.sh --verify
```

While integrated, the Firecrawl checkout is intentionally dirty only at Underbelly's route injection and generated module. `--uninstall` cleans those owned changes; it does not discard unrelated Compose, package, environment, or application edits. Resolve any other dirty paths through their owning workflow before pulling.

A Bun global upgrade normally replaces the patched compiled files. After upgrading either package, simply rerun the three `--dry-run`, install, and `--verify` commands. The installer evaluates Firecrawl CLI, AnyDoc, and Firecrawl independently, so one updated component does not force rewrites or rebuilds of the other two.

Do not uninstall all three integrations for a Bun-only update. Upgrade the desired global package or packages, then rerun the installer; the still-current Firecrawl component will be reported as `already patched` and Compose will be left alone. Full unintegration is primarily for cleaning the Firecrawl Git checkout before a pull or for removing Underbelly altogether.

If upstream moved a file or changed an insertion point, `--dry-run` reports the exact missing or multiply matched anchor and writes nothing. Update `integrate.sh` for the new upstream shape; do not weaken the anchor check or paste blocks manually.

## Unintegrate

```bash
cd ~/multimedia/translate/underbelly
./integrate.sh --uninstall
# --unintegrate is an equivalent alias
```

Unintegration is marker- and ownership-aware:

- It removes only balanced `UNDERBELLY-BEGIN`/`UNDERBELLY-END` blocks, regardless of their installed version.
- It deletes a generated module only when the file contains Underbelly's generator signature.
- It refuses malformed/duplicate markers or a same-named generated file owned by something else.
- It preserves every other byte of the surrounding upstream files.
- It backs up all targets, syntax-checks the remaining CLI files, and rebuilds the API only when Firecrawl changed.
- It does not require the translation service to be running.
- It is idempotent; an already clean installation exits successfully without rebuilding.

A successful removal records `last-uninstall` beside the normal backups. Automatic rollback is the supported recovery path during an operation; backup manifests and original files remain available for manual disaster recovery.

## Maintenance guarantees and limits

The patcher is hardened as a versioned overlay, not as a promise that arbitrary future Firecrawl internals are compatible:

- Exact source anchors prevent a plausible-looking patch from landing in the wrong function.
- Existing marked blocks update in place across Underbelly versions.
- Deterministic generated modules and `--verify` detect stale or locally edited integration output.
- Component-level change detection avoids unnecessary Firecrawl rebuilds.
- Backups plus transactional rollback protect all touched files.
- Unintegration makes a normal fast-forward pull possible without committing Underbelly into the Firecrawl clone.

Major upstream source-layout or response-envelope changes should fail explicitly and require a reviewed patcher update. That failure is intentional. Underbelly does not run `git pull`, reset a checkout, reinstall Bun packages, edit `.env`, or clean changes it does not own.
