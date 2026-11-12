#!/usr/bin/env python3

"""Deploy versioned Sphinx documentation to GitHub Pages.

The workflow builds HTML into docs/_build/html, then this script commits that
HTML to the local gh-pages branch under /latest/ or /vMAJOR.MINOR/. It also
maintains /stable/, /versions.json, /index.html, /404.html, and .nojekyll.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

BASE_URL = "https://nvlabs.github.io/SOMA-X"
HTML_DIR = Path("docs/_build/html")
DEPLOY_EXCLUDE = shutil.ignore_patterns(".doctrees", "__pycache__", "*.pyc")
VERSION_RE = re.compile(r"^\d+\.\d+$")


def git_run(*args: str, cwd: Path | None = None) -> str:
    cmd = ("git", *args)
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        result.check_returncode()
    return result.stdout.strip()


def resolve_version_folder(version: str) -> str:
    if version == "latest":
        return "latest"
    if not VERSION_RE.match(version):
        raise ValueError(
            f"Invalid --version '{version}': expected 'latest' or MAJOR.MINOR"
        )
    return f"v{version}"


def version_key(version: str) -> tuple[int, int]:
    major, minor = version.split(".")
    return int(major), int(minor)


def discover_versions(gh_pages_dir: Path) -> list[str]:
    versions = []
    for entry in gh_pages_dir.iterdir():
        if not entry.is_dir():
            continue
        match = re.fullmatch(r"v(\d+\.\d+)", entry.name)
        if match:
            versions.append(match.group(1))
    return sorted(versions, key=version_key, reverse=True)


def is_released(version: str) -> bool:
    result = subprocess.run(
        ("git", "tag", "-l", f"v{version}.*"),
        check=False,
        capture_output=True,
        text=True,
    )
    pattern = re.compile(rf"^v{re.escape(version)}\.\d+$")
    return any(pattern.fullmatch(tag) for tag in result.stdout.splitlines())


def write_versions_json(
    gh_pages_dir: Path,
    versions: list[str],
    released: list[str],
    has_latest: bool,
) -> None:
    entries: list[dict[str, object]] = []
    if has_latest:
        entries.append(
            {
                "name": "latest (main)",
                "version": "latest",
                "url": f"{BASE_URL}/latest/",
            }
        )

    released_set = set(released)
    preferred = released[0] if released else None
    for version in versions:
        entry: dict[str, object] = {
            "version": version,
            "url": f"{BASE_URL}/v{version}/",
        }
        if version == preferred:
            entry["name"] = f"{version} (stable)"
            entry["preferred"] = True
        elif version not in released_set:
            entry["name"] = f"{version} (prerelease)"
        else:
            entry["name"] = version
        entries.append(entry)

    path = gh_pages_dir / "versions.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {path} with {len(entries)} entries")


def write_root_redirect(gh_pages_dir: Path, target: str) -> None:
    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url={target}">
  <script>window.location.href = "{target}";</script>
</head>
<body>
  <p>Redirecting to <a href="{target}">{target}</a>...</p>
</body>
</html>
"""
    (gh_pages_dir / "index.html").write_text(html)
    print(f"Root redirect -> {target}")


def write_404_redirect(gh_pages_dir: Path, target: str) -> None:
    prefix = urllib.parse.urlparse(BASE_URL).path.rstrip("/") + "/"
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>Page not found - SOMA-X</title>
  <script>
(function () {{
  var prefix = {json.dumps(prefix)};
  var path = location.pathname;
  if (path.indexOf(prefix) === 0) {{
    path = path.slice(prefix.length);
  }}
  if (/^(stable|latest|v\\d+\\.\\d+)(\\/|$)/.test(path)) {{
    return;
  }}
  location.replace(prefix + {json.dumps(target)} + path + location.search + location.hash);
}})();
  </script>
  <style>
    body {{
      background: #f1f1f1;
      color: #222;
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 0;
    }}
    .container {{
      margin: 0 auto;
      max-width: 600px;
      padding: 32px 16px 40px;
      text-align: center;
    }}
    h1 {{
      color: #222;
      font-size: 120px;
      font-weight: 800;
      line-height: 1;
      margin: 0;
    }}
    h2 {{
      color: #5a5a5a;
      font-size: 24px;
      font-weight: 400;
      margin: 12px 0 24px;
    }}
    p {{
      color: #5a5a5a;
      font-size: 14px;
    }}
    a {{ color: #76b900; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>404</h1>
    <h2>File not found.</h2>
    <p>The requested page does not exist in this version of the SOMA-X documentation.</p>
    <p><a href="{prefix}stable/">Latest stable docs</a> &middot; <a href="{prefix}latest/">Development docs</a></p>
  </div>
</body>
</html>
"""
    (gh_pages_dir / "404.html").write_text(html)
    print(f"404 fallback -> {target}")


def update_stable(gh_pages_dir: Path, stable_version: str) -> None:
    stable_dir = gh_pages_dir / "stable"
    source_dir = gh_pages_dir / f"v{stable_version}"
    if stable_dir.exists():
        shutil.rmtree(stable_dir)
    shutil.copytree(source_dir, stable_dir)
    print(f"/stable/ -> v{stable_version}")


def ensure_gh_pages_worktree(worktree_dir: Path) -> None:
    branch_exists = bool(git_run("branch", "--list", "gh-pages"))
    if branch_exists:
        git_run("worktree", "add", str(worktree_dir), "gh-pages")
        return

    remote = subprocess.run(
        ("git", "ls-remote", "--heads", "origin", "gh-pages"),
        check=False,
        capture_output=True,
        text=True,
    )
    if remote.returncode == 0 and remote.stdout.strip():
        raise RuntimeError(
            "origin has a gh-pages branch but it is not present locally; fetch it first"
        )

    git_run("worktree", "add", "--detach", str(worktree_dir), "HEAD")
    git_run("checkout", "--orphan", "gh-pages", cwd=worktree_dir)
    git_run("rm", "-rf", ".", cwd=worktree_dir)


def deploy(version: str, metadata_only: bool = False) -> None:
    folder = resolve_version_folder(version)
    print(f"Deploying docs to /{folder}/ (metadata_only={metadata_only})")

    if not metadata_only and not HTML_DIR.exists():
        raise FileNotFoundError(f"Built docs not found at {HTML_DIR}")

    with tempfile.TemporaryDirectory() as tmp:
        gh_pages_dir = Path(tmp) / "gh-pages"
        ensure_gh_pages_worktree(gh_pages_dir)
        try:
            target = gh_pages_dir / folder
            if metadata_only:
                if not target.exists():
                    raise RuntimeError(
                        f"--metadata-only requested, but /{folder}/ is not deployed"
                    )
            else:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(HTML_DIR, target, ignore=DEPLOY_EXCLUDE)
                print(f"Copied {HTML_DIR} -> {target}")

            versions = discover_versions(gh_pages_dir)
            released = [v for v in versions if is_released(v)]
            has_latest = (gh_pages_dir / "latest").is_dir()
            print(
                f"Deployed versions: {versions}; released: {released}; "
                f"has latest: {has_latest}"
            )

            if released:
                update_stable(gh_pages_dir, released[0])
                write_root_redirect(gh_pages_dir, "stable/")
                write_404_redirect(gh_pages_dir, "stable/")
            elif has_latest:
                stable_dir = gh_pages_dir / "stable"
                if stable_dir.exists():
                    shutil.rmtree(stable_dir)
                write_root_redirect(gh_pages_dir, "latest/")
                write_404_redirect(gh_pages_dir, "latest/")

            write_versions_json(gh_pages_dir, versions, released, has_latest)
            (gh_pages_dir / ".nojekyll").touch()

            git_run("add", "-A", cwd=gh_pages_dir)
            if not git_run("status", "--porcelain", cwd=gh_pages_dir):
                print("No changes to commit")
                return
            git_run(
                "-c",
                "user.email=actions@github.com",
                "-c",
                "user.name=GitHub Actions",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-m",
                f"Deploy docs: {folder}",
                cwd=gh_pages_dir,
            )
        finally:
            git_run("worktree", "remove", "--force", str(gh_pages_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy versioned Sphinx documentation to GitHub Pages."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="'latest' or MAJOR.MINOR, for example '0.2'",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only refresh stable/version metadata for an already deployed version.",
    )
    args = parser.parse_args()

    try:
        deploy(args.version, metadata_only=args.metadata_only)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
