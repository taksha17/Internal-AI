/**
 * The whole server side of the site: the access form, and one canonical host.
 *
 * Cloudflare would otherwise serve assets before the Worker runs, which leaves
 * `/api/access` unanswerable and lets `www` serve a second copy of every page.
 * So the Worker runs first and declines anything it does not own, handing the
 * request back to the assets layer so pages, headers and caching stay the
 * assets layer's business.
 */

import { accessGetOnly, handleAccess } from "./access.js";

const APEX = "internal-ai.dev";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.hostname === `www.${APEX}`) {
      url.hostname = APEX;
      return Response.redirect(url.toString(), 301);
    }

    if (url.pathname === "/api/access") {
      return request.method === "POST" ? handleAccess(request, env) : accessGetOnly();
    }

    if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
      return new Response("Not found\n", {
        status: 404,
        headers: {
          "content-type": "text/plain; charset=utf-8",
          "cache-control": "no-store",
        },
      });
    }

    return env.ASSETS.fetch(request);
  },
};
