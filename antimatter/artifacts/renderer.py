"""Artifact HTML Canvas Renderer for Project Antimatter.

Renders self-contained interactive web views for Markdown, Mermaid diagrams,
HTML widgets, and code diffs with offline-capable styles.
"""

import html
import re
from antimatter.artifacts.types import Artifact, ArtifactType

BASE_CSS = """
:root {
  --bg-primary: #121316;
  --bg-secondary: #1a1b20;
  --bg-surface: #24262d;
  --text-primary: #e6e8eb;
  --text-secondary: #9da3ae;
  --border-color: #2f333c;
  --accent: #3b82f6;
  --accent-hover: #60a5fa;
  --diff-add-bg: rgba(46, 160, 67, 0.15);
  --diff-add-text: #3fb950;
  --diff-del-bg: rgba(248, 81, 73, 0.15);
  --diff-del-text: #f85149;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  padding: 24px;
}
.artifact-header {
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
.artifact-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 8px; }
.artifact-summary { color: var(--text-secondary); font-size: 0.95rem; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  background-color: var(--bg-surface);
  color: var(--accent);
  margin-top: 8px;
}
.content-wrapper {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 20px;
  overflow-x: auto;
}
pre {
  font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
  font-size: 0.9rem;
}
.diff-line-add { background: var(--diff-add-bg); color: var(--diff-add-text); display: block; }
.diff-line-del { background: var(--diff-del-bg); color: var(--diff-del-text); display: block; }
.diff-line-info { color: var(--text-secondary); display: block; }
"""


class ArtifactRenderer:
    """Produces standalone HTML documents for browser and webview canvas rendering."""

    @classmethod
    def render_to_html(cls, artifact: Artifact) -> str:
        """Render artifact into a standalone responsive HTML page."""
        title = html.escape(artifact.metadata.title)
        summary = html.escape(artifact.metadata.summary)
        atype = artifact.metadata.artifact_type

        inner_body = ""

        if atype == ArtifactType.HTML:
            # Embed raw HTML inside container
            inner_body = f"<div class='html-canvas'>{artifact.content}</div>"

        elif atype == ArtifactType.MERMAID:
            clean_mermaid = html.escape(artifact.content.strip())
            inner_body = f"""
            <div class="mermaid">
{clean_mermaid}
            </div>
            <script type="module">
              import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
              mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
            </script>
            """

        elif atype == ArtifactType.DIFF:
            diff_lines = []
            for line in artifact.content.splitlines():
                esc = html.escape(line)
                if line.startswith("+"):
                    diff_lines.append(f"<span class='diff-line-add'>{esc}</span>")
                elif line.startswith("-"):
                    diff_lines.append(f"<span class='diff-line-del'>{esc}</span>")
                elif line.startswith("@"):
                    diff_lines.append(f"<span class='diff-line-info'>{esc}</span>")
                else:
                    diff_lines.append(f"<span>{esc}</span>")
            inner_body = f"<pre class='diff-view'>{''.join(diff_lines)}</pre>"

        else:
            # Markdown / Default text
            escaped_text = html.escape(artifact.content)
            inner_body = f"<pre><code>{escaped_text}</code></pre>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} - Antimatter Artifact Canvas</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <div class="artifact-header">
    <h1 class="artifact-title">{title}</h1>
    {f'<p class="artifact-summary">{summary}</p>' if summary else ''}
    <span class="badge">{atype.value}</span>
  </div>
  <div class="content-wrapper">
    {inner_body}
  </div>
</body>
</html>"""
