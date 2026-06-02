#!/usr/bin/env python3
"""Render maintained markdown docs into static HTML under docs/maintained."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
OUTPUT_ROOT = DOCS_ROOT / "maintained"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

PORTAL_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Core workflow",
        (
            "readme.md",
            "INSTALL.md",
            "tools/README.md",
            "tests/README.md",
            "notes/DOCS_OWNERSHIP_MAP.md",
            "notes/DASHBOARD_SHELL_HOWTO.md",
        ),
    ),
    (
        "Reference data and validation",
        (
            "notes/REFERENCE_DATASET_HOWTO.md",
            "notes/REFERENCE_VALIDATION_HOWTO.md",
            "notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md",
            "research_context/README.md",
        ),
    ),
    (
        "Remote and build operations",
        (
            "notes/porting/SOL_REMOTE_WORKFLOW.md",
            "notes/porting/NEURON_UPGRADE_WORKFLOW.md",
            "notes/porting/MODERN_NEURON_PORT_NOTES.md",
        ),
    ),
)

MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)")
HTML_HREF_PATTERN = re.compile(r'(?P<prefix>href=")(?P<href>[^"]+)(?P<suffix>")')
HTML_SRC_PATTERN = re.compile(r'(?P<prefix>src=")(?P<href>[^"]+)(?P<suffix>")')


@dataclass(frozen=True)
class ManagedDoc:
    source_rel: Path
    output_rel: Path
    title: str


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _seed_docs() -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for _section, relpaths in PORTAL_SECTIONS:
        for rel in relpaths:
            path = Path(rel)
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _normalize_repo_markdown_target(source_rel: Path, href: str) -> Path | None:
    target = _resolve_repo_target(source_rel, href)
    if target is None:
        return None
    rel = target.relative_to(REPO_ROOT.resolve())
    if rel.suffix.lower() != ".md":
        return None
    return rel


def _resolve_repo_target(source_rel: Path, href: str) -> Path | None:
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        return None
    raw_path = parts.path
    if not raw_path:
        return None
    if raw_path.startswith("/home/michael/OlfactoryBulb/") or raw_path.startswith("/home/alek/OlfactoryBulb/"):
        target = Path(raw_path).expanduser().resolve()
    elif raw_path.startswith("/"):
        return None
    else:
        target = ((REPO_ROOT / source_rel).parent / raw_path).resolve()
    try:
        target.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    if not target.exists():
        return None
    return target


def _discover_managed_docs(seed_docs: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    queue: list[Path] = []
    seen: set[Path] = set()
    for rel in seed_docs:
        if rel not in seen:
            seen.add(rel)
            queue.append(rel)
            discovered.append(rel)
    while queue:
        current = queue.pop(0)
        text = (REPO_ROOT / current).read_text()
        for href in MARKDOWN_LINK_PATTERN.findall(text):
            target = _normalize_repo_markdown_target(current, href)
            if target is None or target in seen:
                continue
            seen.add(target)
            queue.append(target)
            discovered.append(target)
    return discovered


def _output_rel_for_doc(source_rel: Path) -> Path:
    return Path("maintained") / source_rel.with_suffix(".html")


def _relative_href(from_output: Path, to_output: Path, fragment: str = "") -> str:
    rel = os.path.relpath((DOCS_ROOT / to_output), (DOCS_ROOT / from_output).parent).replace(os.sep, "/")
    return f"{rel}{fragment}"


def _rewrite_repo_href(source_doc: ManagedDoc, href: str, managed_by_source: dict[Path, ManagedDoc]) -> tuple[str, str | None]:
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        return href, None
    fragment = f"#{parts.fragment}" if parts.fragment else ""
    target = _normalize_repo_markdown_target(source_doc.source_rel, href)
    if target is not None:
        managed = managed_by_source.get(target)
        if managed is not None:
            return _relative_href(source_doc.output_rel, managed.output_rel, fragment), None
    repo_target = _resolve_repo_target(source_doc.source_rel, href)
    if repo_target is not None:
        rel = repo_target.relative_to(REPO_ROOT.resolve())
        local_href = os.path.relpath(repo_target, (DOCS_ROOT / source_doc.output_rel).parent).replace(os.sep, "/")
        served_href = urlunsplit(("", "", f"/repo/{rel.as_posix()}", parts.query, parts.fragment))
        return local_href, served_href
    return href, None


def _rewrite_html_links(html: str, source_doc: ManagedDoc, managed_by_source: dict[Path, ManagedDoc]) -> str:
    def _rewrite(match: re.Match[str]) -> str:
        href = match.group("href")
        rewritten, served_href = _rewrite_repo_href(source_doc, href, managed_by_source)
        if served_href:
            return f'{match.group("prefix")}{rewritten}{match.group("suffix")} data-served-href="{served_href}"'
        return f"{match.group('prefix')}{rewritten}{match.group('suffix')}"

    html = HTML_HREF_PATTERN.sub(_rewrite, html)
    html = HTML_SRC_PATTERN.sub(_rewrite, html)
    return html


def _render_doc(source_doc: ManagedDoc, managed_by_source: dict[Path, ManagedDoc]) -> str:
    source_path = REPO_ROOT / source_doc.source_rel
    text = source_path.read_text()
    md = markdown.Markdown(extensions=["extra", "toc", "sane_lists"])
    body = md.convert(text)
    toc = getattr(md, "toc", "")
    body = _rewrite_html_links(body, source_doc, managed_by_source)
    toc = _rewrite_html_links(toc, source_doc, managed_by_source)
    portal_href = _relative_href(source_doc.output_rel, Path("index.html"))
    source_href = _relative_href(source_doc.output_rel, source_doc.source_rel)
    served_source_href = f"/repo/{source_doc.source_rel.as_posix()}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{source_doc.title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --panel-alt: #f0f4ff;
      --text: #17202a;
      --muted: #667085;
      --accent: #2563eb;
      --line: #d9dee8;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
      --code-bg: #0f172a;
      --code-ink: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 72px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 24px;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ order: -1; }}
    }}
    article, aside {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    article {{ padding: 28px; min-width: 0; }}
    aside {{ padding: 20px; height: fit-content; position: sticky; top: 20px; }}
    .eyebrow {{
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      margin-bottom: 10px;
    }}
    .meta {{
      margin: 0 0 18px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-alt);
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .meta a {{ color: var(--accent); text-decoration: none; }}
    .meta a:hover {{ text-decoration: underline; }}
    h1, h2, h3, h4 {{ line-height: 1.25; }}
    h1 {{ margin-top: 0; font-size: 2rem; }}
    h2 {{ margin-top: 2rem; font-size: 1.45rem; }}
    h3 {{ margin-top: 1.5rem; font-size: 1.15rem; }}
    p, li {{ min-width: 0; }}
    a {{ color: var(--accent); }}
    pre {{
      background: var(--code-bg);
      color: var(--code-ink);
      padding: 14px 16px;
      border-radius: 8px;
      overflow-x: auto;
      border: 1px solid rgba(226, 232, 240, 0.08);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }}
    :not(pre) > code {{
      background: rgba(37, 99, 235, 0.08);
      color: #1d4ed8;
      padding: 0.12rem 0.32rem;
      border-radius: 4px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0;
      font-size: 0.95rem;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 0.55rem 0.7rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f8fafc; }}
    blockquote {{
      margin: 1rem 0;
      padding: 0.8rem 1rem;
      border-left: 3px solid var(--accent);
      background: rgba(37, 99, 235, 0.06);
      color: var(--muted);
    }}
    ul, ol {{ padding-left: 1.3rem; }}
    img {{ max-width: 100%; height: auto; }}
    .toc ul {{ padding-left: 1rem; }}
    .toc a {{ color: var(--accent); text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
  </style>
  <script>
    if (window.location.protocol !== "file:") {{
      window.addEventListener("DOMContentLoaded", function () {{
        document.querySelectorAll("a[data-served-href]").forEach(function (anchor) {{
          anchor.setAttribute("href", anchor.getAttribute("data-served-href"));
        }});
      }});
    }}
  </script>
</head>
<body>
  <main>
    <article>
      <div class="eyebrow">Maintained Docs</div>
      <p class="meta">
        <a href="{portal_href}">Docs portal</a>
        ·
        <a href="{source_href}" data-served-href="{served_source_href}">View source markdown</a>
        ·
        <code>{source_doc.source_rel.as_posix()}</code>
      </p>
      {body}
    </article>
    <aside>
      <div class="eyebrow">On This Page</div>
      <div class="toc">{toc or "<p>No table of contents for this page.</p>"}</div>
    </aside>
  </main>
</body>
</html>
"""


def build() -> dict[str, object]:
    seed_docs = _seed_docs()
    discovered = _discover_managed_docs(seed_docs)
    managed_docs: list[ManagedDoc] = []
    for rel in discovered:
        text = (REPO_ROOT / rel).read_text()
        title = _extract_title(text, rel.as_posix())
        managed_docs.append(
            ManagedDoc(
                source_rel=rel,
                output_rel=_output_rel_for_doc(rel),
                title=title,
            )
        )
    managed_by_source = {doc.source_rel: doc for doc in managed_docs}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for doc in managed_docs:
        output_path = DOCS_ROOT / doc.output_rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_render_doc(doc, managed_by_source))
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sections": [
            {
                "title": section_title,
                "pages": [
                    {
                        "source_rel": rel,
                        "output_rel": managed_by_source[Path(rel)].output_rel.as_posix(),
                        "title": managed_by_source[Path(rel)].title,
                    }
                    for rel in relpaths
                ],
            }
            for section_title, relpaths in PORTAL_SECTIONS
        ],
        "documents": [
            {
                "source_rel": doc.source_rel.as_posix(),
                "output_rel": doc.output_rel.as_posix(),
                "title": doc.title,
            }
            for doc in managed_docs
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    manifest = build()
    print(f"Rendered {len(manifest['documents'])} maintained docs to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
