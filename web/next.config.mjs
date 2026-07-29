/**
 * No API proxy any more.
 *
 * The cohort analysis is served from `public/data` as static files, and the one
 * dynamic endpoint (`/api/explain`) is a route handler in this app. An earlier
 * version rewrote `/api/*` to a FastAPI process; that rewrite would now shadow
 * the route handler, so it is gone along with the second port and the CORS
 * configuration it needed.
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
