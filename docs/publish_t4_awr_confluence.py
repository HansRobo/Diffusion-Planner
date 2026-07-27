#!/usr/bin/env python3
"""Publish the T4 Original-DP AWR report to a Confluence draft.

Credentials are read exclusively from ATLASSIAN_SITE, ATLASSIAN_EMAIL, and
ATLASSIAN_API_TOKEN.  The script never stores or prints them.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "docs" / "t4_dp_awr_confluence_ja.html"
IMAGE_RE = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.IGNORECASE)
ATTR_RE = re.compile(r"""([:\w-]+)\s*=\s*(["'])(.*?)\2""", re.DOTALL)
LOCAL_LINK_RE = re.compile(r"""<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>""", re.IGNORECASE | re.DOTALL)
SUPPORT_SUFFIXES = {".json", ".md", ".csv", ".sha256", ".zip", ".txt"}


def fail(message: str) -> None:
    raise SystemExit(message)


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        fail(f"Missing required environment variable: {name}")
    return value


def parse_attrs(text: str) -> dict[str, str]:
    return {key.lower(): html.unescape(value) for key, _, value in ATTR_RE.findall(text)}


def resolve_image(source_html: Path, src: str) -> Path:
    path = (source_html.parent / src).resolve()
    if not path.is_file():
        fail(f"Referenced image does not exist: {src} -> {path}")
    return path


def attachment_name(path: Path) -> str:
    """Create stable, readable, collision-resistant names for page attachments."""
    try:
        rel = path.relative_to(REPO_ROOT)
        parts = list(rel.parts)
    except ValueError:
        parts = [path.parent.name, path.name]
    stem = "-".join(parts).lower()
    stem = re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-.")
    name = f"t4-awr-{stem}"
    if len(name) > 220:
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        name = f"{name[:200]}-{digest}{path.suffix.lower()}"
    return name


def discover_assets(source_html: Path) -> dict[str, tuple[Path, str]]:
    source = source_html.read_text(encoding="utf-8")
    result: dict[str, tuple[Path, str]] = {}
    names: dict[str, Path] = {}
    for match in IMAGE_RE.finditer(source):
        attrs = parse_attrs(match.group("attrs"))
        src = attrs.get("src")
        if not src or src.startswith(("http://", "https://", "data:")):
            continue
        path = resolve_image(source_html, src)
        name = attachment_name(path)
        previous = names.get(name)
        if previous is not None and previous != path:
            fail(f"Attachment name collision: {previous} and {path} -> {name}")
        names[name] = path
        result[src] = (path, name)
    return result


def discover_supporting_files(source_html: Path) -> dict[str, tuple[Path, str]]:
    source = source_html.read_text(encoding="utf-8")
    result: dict[str, tuple[Path, str]] = {}
    names: dict[str, Path] = {}
    for match in LOCAL_LINK_RE.finditer(source):
        href = parse_attrs(match.group("attrs")).get("href", "")
        if not href or href.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (source_html.parent / href).resolve()
        if not path.is_file() or path.suffix.lower() not in SUPPORT_SUFFIXES:
            continue
        # Attach report evidence and its self-contained bundle, not arbitrary
        # source files referenced elsewhere in the repository.
        try:
            rel = path.relative_to(REPO_ROOT / "docs")
        except ValueError:
            continue
        if not (
            str(rel).startswith("t4_conference_assets/")
            or str(rel).startswith("t4_rl_assets/")
            or str(rel).startswith("assets/fonts/")
        ):
            continue
        name = attachment_name(path)
        previous = names.get(name)
        if previous is not None and previous != path:
            fail(f"Attachment name collision: {previous} and {path} -> {name}")
        names[name] = path
        result[href] = (path, name)
    return result


def replace_local_links(value: str, supporting_files: dict[str, tuple[Path, str]]) -> str:
    def replacement(match: re.Match[str]) -> str:
        attrs = parse_attrs(match.group("attrs"))
        href = attrs.get("href", "")
        body = match.group("body")
        if href.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        if href in supporting_files:
            _, name = supporting_files[href]
            return (
                "<ac:link>"
                f'<ri:attachment ri:filename="{html.escape(name, quote=True)}" />'
                f"<ac:link-body>{body}</ac:link-body>"
                "</ac:link>"
            )
        return f"<span>{body}</span>"

    return LOCAL_LINK_RE.sub(replacement, value)


def confluence_storage(
    source_html: Path,
    assets: dict[str, tuple[Path, str]],
    supporting_files: dict[str, tuple[Path, str]],
) -> str:
    source = source_html.read_text(encoding="utf-8")
    main_match = re.search(r"<main\b[^>]*>(?P<body>.*)</main>", source, re.DOTALL | re.IGNORECASE)
    if not main_match:
        fail(f"No <main> element found in {source_html}")
    body = main_match.group("body")

    def image_macro(match: re.Match[str]) -> str:
        attrs = parse_attrs(match.group("attrs"))
        src = attrs.get("src", "")
        if src not in assets:
            return ""
        _, name = assets[src]
        alt = html.escape(attrs.get("alt", "T4 AWR visualization"), quote=True)
        return (
            f'<ac:image ac:align="center" ac:layout="center" ac:width="1200" ac:alt="{alt}">'
            f'<ri:attachment ri:filename="{html.escape(name, quote=True)}" />'
            "</ac:image>"
        )

    body = IMAGE_RE.sub(image_macro, body)
    body = replace_local_links(body, supporting_files)
    body = re.sub(r"<figure\b[^>]*>", "<div>", body, flags=re.IGNORECASE)
    body = re.sub(r"</figure>", "</div>", body, flags=re.IGNORECASE)
    body = re.sub(r"<figcaption\b[^>]*>", "<p><em>", body, flags=re.IGNORECASE)
    body = re.sub(r"</figcaption>", "</em></p>", body, flags=re.IGNORECASE)
    body = re.sub(r"<details\b[^>]*>", "<div>", body, flags=re.IGNORECASE)
    body = re.sub(r"</details>", "</div>", body, flags=re.IGNORECASE)
    body = re.sub(r"<summary\b[^>]*>", "<h3>", body, flags=re.IGNORECASE)
    body = re.sub(r"</summary>", "</h3>", body, flags=re.IGNORECASE)
    body = re.sub(r"<br\s*>", "<br />", body, flags=re.IGNORECASE)

    # Convert report callouts to native Confluence panels before dropping CSS classes.
    callout_re = re.compile(
        r'<div\s+class="callout(?:\s+(good|warn|stop))?"[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )

    def callout_macro(match: re.Match[str]) -> str:
        kind = (match.group(1) or "").lower()
        macro = {"good": "tip", "warn": "warning", "stop": "warning"}.get(kind, "info")
        return (
            f'<ac:structured-macro ac:name="{macro}" ac:schema-version="1">'
            f"<ac:rich-text-body><p>{match.group(2)}</p></ac:rich-text-body>"
            "</ac:structured-macro>"
        )

    body = callout_re.sub(callout_macro, body)
    body = re.sub(r'\s+(?:class|id|loading|style)="[^"]*"', "", body)
    body = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", body)

    preface = """
<ac:structured-macro ac:name="info" ac:schema-version="1">
  <ac:rich-text-body>
    <p><strong>Audience:</strong> Post-Training / AWRを初めて読む人を含むtrajectory-planningチーム。</p>
    <p><strong>Purpose:</strong> 業界背景から始め、HDPのRL、Original DPへの適用、rewardの意味、実scene evidence、現在の実験結果を一つの流れで説明する。</p>
  </ac:rich-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="toc" ac:schema-version="1">
  <ac:parameter ac:name="maxLevel">3</ac:parameter>
  <ac:parameter ac:name="printable">true</ac:parameter>
</ac:structured-macro>
""".strip()
    storage = preface + "\n" + body.strip()

    # Confluence storage is XML-like. Validate structure before any write.
    wrapped = (
        '<root xmlns:ac="http://atlassian.com/content" '
        'xmlns:ri="http://atlassian.com/resource/identifier">' + storage + "</root>"
    )
    try:
        ET.fromstring(wrapped)
    except ET.ParseError as exc:
        Path("/tmp/t4_awr_confluence_storage_invalid.xml").write_text(wrapped, encoding="utf-8")
        fail(f"Generated Confluence storage is not well-formed XML: {exc}")
    return storage


class Confluence:
    def __init__(self) -> None:
        self.site = required_env("ATLASSIAN_SITE").rstrip("/")
        self.session = requests.Session()
        self.session.auth = (required_env("ATLASSIAN_EMAIL"), required_env("ATLASSIAN_API_TOKEN"))
        self.session.headers.update({"Accept": "application/json"})

    def _url(self, suffix: str) -> str:
        return f"{self.site}/wiki/rest/api{suffix}"

    def _v2_url(self, suffix: str) -> str:
        return f"{self.site}/wiki/api/v2{suffix}"

    @staticmethod
    def _check(response: requests.Response, operation: str) -> dict[str, Any]:
        if response.ok:
            if not response.content:
                return {}
            return response.json()
        try:
            detail = response.json().get("message", "")
        except Exception:
            detail = response.text[:300]
        fail(f"{operation} failed: HTTP {response.status_code}: {detail}")
        return None

    def get_page(self, page_id: str) -> dict[str, Any]:
        response = self.session.get(
            self._url(f"/content/{page_id}"),
            params={"status": "any", "expand": "body.storage,version,space"},
            timeout=60,
        )
        return self._check(response, "Read draft")

    def list_attachments(self, page_id: str) -> dict[str, dict[str, Any]]:
        start = 0
        result: dict[str, dict[str, Any]] = {}
        while True:
            response = self.session.get(
                self._url(f"/content/{page_id}/child/attachment"),
                params={"status": "draft", "limit": 200, "start": start, "expand": "version"},
                timeout=60,
            )
            data = self._check(response, "List attachments")
            rows = data.get("results") or []
            for row in rows:
                result[row["title"]] = row
            if len(rows) < 200:
                break
            start += len(rows)
        return result

    def upload_attachment(
        self,
        page_id: str,
        path: Path,
        name: str,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if existing:
            url = self._url(f"/content/{page_id}/child/attachment/{existing['id']}/data")
            operation = f"Update attachment {name}"
        else:
            url = self._url(f"/content/{page_id}/child/attachment")
            operation = f"Upload attachment {name}"
        with path.open("rb") as handle:
            response = self.session.post(
                url,
                params={"status": "draft"},
                headers={"X-Atlassian-Token": "no-check"},
                files={"file": (name, handle, mime)},
                data={"comment": "T4 Original-DP AWR evidence asset", "minorEdit": "true"},
                timeout=300,
            )
        return self._check(response, operation)

    def update_draft(self, page: dict[str, Any], storage: str) -> dict[str, Any]:
        # Unpublished shared drafts use a fixed draft version.  REST v1 asks for
        # an increment and then rejects draft versioning; v2 accepts the current
        # draft version with status=draft.
        version = int((page.get("version") or {}).get("number") or 1)
        payload = {
            "id": page["id"],
            "status": "draft",
            "title": page["title"],
            "spaceId": str((page.get("space") or {}).get("id") or ""),
            "body": {"value": storage, "representation": "storage"},
            "version": {
                "number": version,
                "message": "Full T4 Original-DP AWR know-how, evidence, and proposal",
            },
        }
        response = self.session.put(
            self._v2_url(f"/pages/{page['id']}"),
            json=payload,
            timeout=180,
        )
        return self._check(response, "Update draft")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-id", default="5392827425")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-attachments", action="store_true")
    parser.add_argument(
        "--replace-entire-draft",
        action="store_true",
        help="Explicitly authorize replacing the complete remote draft body.",
    )
    parser.add_argument(
        "--expected-remote-sha256",
        help="Required with --replace-entire-draft; fail if the live draft changed.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    assets = discover_assets(source)
    supporting_files = discover_supporting_files(source)
    storage = confluence_storage(source, assets, supporting_files)
    unique_attachments = {
        (path, name) for path, name in (*assets.values(), *supporting_files.values())
    }
    print(
        json.dumps(
            {
                "source": str(source),
                "storage_chars": len(storage),
                "image_attachments": len({name for _, name in assets.values()}),
                "support_attachments": len({name for _, name in supporting_files.values()}),
                "attachments": len(unique_attachments),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return 0

    confluence = Confluence()
    page = confluence.get_page(args.page_id)
    if page.get("status") != "draft":
        fail(f"Refusing to update non-draft content: status={page.get('status')!r}")
    remote_storage = ((page.get("body") or {}).get("storage") or {}).get("value") or ""
    remote_sha256 = hashlib.sha256(remote_storage.encode("utf-8")).hexdigest()
    if not args.replace_entire_draft:
        fail(
            "Refusing full-page replacement: this Confluence draft is manually edited. "
            "Use a targeted merge workflow instead."
        )
    expected_sha256 = (args.expected_remote_sha256 or "").strip().lower()
    if not expected_sha256:
        fail("--expected-remote-sha256 is required with --replace-entire-draft")
    if remote_sha256 != expected_sha256:
        fail(
            "Remote draft changed since it was reviewed; refusing to overwrite it. "
            f"actual_sha256={remote_sha256}"
        )

    if not args.skip_attachments:
        existing = confluence.list_attachments(args.page_id)
        unique_assets = sorted(unique_attachments, key=lambda x: x[1])
        for index, (path, name) in enumerate(unique_assets, 1):
            remote = existing.get(name)
            remote_size = int(((remote or {}).get("extensions") or {}).get("fileSize") or -1)
            if remote is not None and remote_size == path.stat().st_size:
                print(f"attachment {index}/{len(unique_assets)} unchanged: {name}")
                continue
            confluence.upload_attachment(args.page_id, path, name, remote)
            print(f"attachment {index}/{len(unique_assets)} uploaded: {name}")

    updated = confluence.update_draft(page, storage)
    print(
        json.dumps(
            {
                "updated": True,
                "id": updated.get("id"),
                "status": updated.get("status"),
                "title": updated.get("title"),
                "version": (updated.get("version") or {}).get("number"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
