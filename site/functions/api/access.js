/**
 * POST /api/access — one access request, filed as an issue.
 *
 * The rest of the site is static and ships no script. This is the only code it
 * runs, because a visitor without repository access cannot open an issue and a
 * request still has to reach a person. GitHub is the inbox: the issue arrives
 * with its own notification, and the collaborator invitation happens in the
 * same place. No third-party form processor ever sees the submission.
 */

const REPO = "taksha17/Internal-AI";
const MAX_CONTEXT = 2000;
// GitHub allows 1-39 characters: alphanumerics and single hyphens, no leading
// or trailing hyphen, no two in a row.
const HANDLE = /^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$/;

const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&", "<": "<", ">": ">", '"': """, "'": "'" })[c],
  );

/** Returns the normalised handle, or null when it is not a GitHub username. */
export function normaliseHandle(raw) {
  const handle = String(raw ?? "")
    .trim()
    .replace(/^@+/, "");
  return HANDLE.test(handle) ? handle : null;
}

/** Drops control characters a textarea should never carry and caps the length. */
export function clipContext(raw) {
  return String(raw ?? "")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "")
    .trim()
    .slice(0, MAX_CONTEXT);
}

export function issuePayload(handle, context) {
  const asked = context ? `\n\n**What they would run it on**\n\n${context}` : "";
  return {
    title: `Access request: @${handle}`,
    body:
      `**GitHub account to invite:** @${handle}${asked}\n\n` +
      "Filed by the form on the site. Inviting the account closes this.",
  };
}

function page({ status, heading, message }) {
  return new Response(
    `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(heading)} · internal-ai</title>
<meta name="robots" content="noindex">
<link rel="stylesheet" href="/styles.css">
</head>
<body class="band-night">
<main class="notfound">
<div>
<span class="kicker">Request access</span>
<h1>${escape(heading)}</h1>
<p>${message}</p>
<a class="btn btn-primary home" href="/#request">Back to the form</a>
</div>
</main>
</body>
</html>`,
    {
      status,
      headers: {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      },
    },
  );
}

export async function onRequestPost({ request, env }) {
  let form;
  try {
    form = await request.formData();
  } catch {
    return page({
      status: 400,
      heading: "That did not send",
      message: "The form could not be read. Reload the page and try again.",
    });
  }

  // A field no person can see. A bot fills it and gets a quiet success.
  if (String(form.get("website") ?? "").trim() !== "") {
    return page({
      status: 200,
      heading: "Request received.",
      message: "A person reads every request and replies.",
    });
  }

  const handle = normaliseHandle(form.get("github"));
  if (!handle) {
    return page({
      status: 400,
      heading: "That username is not one GitHub allows",
      message:
        "GitHub usernames run 1 to 39 characters: letters, numbers and single hyphens, and they cannot start or end with a hyphen.",
    });
  }

  const token = env.GITHUB_TOKEN;
  if (!token) {
    return page({
      status: 503,
      heading: "The form is not wired up yet",
      message:
        "No repository token is configured on this deployment, so nothing was filed and nobody was notified.",
    });
  }

  const { title, body } = issuePayload(handle, clipContext(form.get("context")));
  let upstream;
  try {
    upstream = await fetch(`https://api.github.com/repos/${REPO}/issues`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        accept: "application/vnd.github+json",
        "content-type": "application/json",
        "user-agent": "internal-ai-site",
      },
      body: JSON.stringify({ title, body }),
    });
  } catch {
    return page({
      status: 502,
      heading: "That did not send",
      message: "The repository could not be reached, so nothing was recorded. Try again in a moment.",
    });
  }

  if (upstream.status !== 201) {
    return page({
      status: 502,
      heading: "That did not send",
      message: `GitHub refused the request (HTTP ${upstream.status}), so nothing was recorded. Try again in a moment.`,
    });
  }

  return page({
    status: 200,
    heading: "Request received.",
    message: `Filed for @${escape(handle)}. A person reads it and invites that account — or replies to say it is not a fit.`,
  });
}

export function onRequestGet() {
  return page({
    status: 405,
    heading: "Use the form",
    message: "This address only accepts the form on the site.",
  });
}