# Console template preview

This folder is a **visual proposal only**. It does not change the running app.

## Preview it

From the repository root, serve this folder locally:

```bash
cd design
python3 -m http.server 8123
```

Open:

- `http://127.0.0.1:8123/index.html`

The page is one static prototype with in-page sections:

- `#overview`
- `#approvals`
- `#data`
- `#jobs`
- `#components`
- `#sign-in`

Sidebar links use fragment anchors in the prototype. The implementation will use
the existing console routes.

## What is intentionally preserved

- Warm charcoal rail, muted sage accent, soft ivory workspace.
- Existing light/dark token values.
- Inter first, system fallback, no downloaded font.
- `prefers-color-scheme` only; no theme toggle.
- No inline `style`, no inline `script`, no external requests.
- Kill switch remains visible on signed-in screens with a typed reason.
- Gates remain human decisions; the requester cannot decide their own gate.
- Brand assets remain the supplied SVGs.

## Implementation mapping

When this direction is approved, port it into:

- `src/internal_ai/console/templates/base.html`
- `src/internal_ai/console/templates/_macros.html`
- `src/internal_ai/console/static/console.css`
- page templates, without changing copy, field names, routes, or checks.

Prototype forms use `action="#"` because this preview has no backend.
