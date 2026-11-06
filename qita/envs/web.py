"""Web server over fixed pages: search and fetch, English tool names as in common MCP servers."""

from __future__ import annotations

import re

from .base import FIXTURES, Env, ToolError


def load_pages() -> dict[str, dict]:
    pages = {}
    for f in sorted((FIXTURES / "web").glob("*.md")):
        head, body = f.read_text(encoding="utf-8").split("\n\n", 1)
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        pages[meta["url"]] = {"title": meta["title"], "body": body.strip()}
    return pages


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.casefold()) if len(w) > 2}


class WebEnv(Env):
    name = "web"
    title = "web"

    def build(self) -> None:
        pages = load_pages() if self.ws.world.pages is None else self.ws.world.pages
        s = self.server

        @s.tool()
        def web_search(query: str, max_results: int = 3) -> str:
            """Search the web. Returns titles, URLs and a short snippet for the best results."""
            q = words(query)
            scored = []
            for url, p in pages.items():
                score = len(q & words(p["title"])) * 3 + len(q & words(p["body"]))
                if score:
                    scored.append((-score, url))
            if not scored:
                return "No results found"
            out = []
            for _, url in sorted(scored)[: max(1, max_results)]:
                p = pages[url]
                snippet = " ".join(p["body"].split())[:160]
                out.append(f"{p['title']}\n{url}\n{snippet}...")
            return "\n\n".join(out)

        @s.tool()
        def fetch(url: str) -> str:
            """Fetch a URL and return the page content as markdown."""
            u = url.strip().rstrip("/")
            if not re.match(r"^https?://", u):
                raise ToolError(f"Invalid URL: {url}")
            u = re.sub(r"^http://", "https://", u)
            if u not in pages:
                raise ToolError(f"HTTP 404 Not Found: {url}")
            p = pages[u]
            return f"# {p['title']}\n\n{p['body']}" if not p["body"].startswith("#") else p["body"]
