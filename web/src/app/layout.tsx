import type { Metadata, Viewport } from "next";

import "./globals.css";

const DESCRIPTION =
  "개인 기저선 이탈 기반의 설명가능한 조기경보 — 연구·교육용 데모. 의료기기가 아닙니다.";

export const metadata: Metadata = {
  title: "활력징후 조기경보 대시보드",
  description: DESCRIPTION,
  // Deep links are shareable now (see lib/url.ts), so they get unfurled in chat
  // apps. Without this the card falls back to the bare URL.
  openGraph: {
    title: "활력징후 조기경보 대시보드",
    description: DESCRIPTION,
    locale: "ko_KR",
    type: "website",
  },
};

/** Matches the two `--background` values, so the browser chrome does not fight the page. */
export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f9f9f7" },
    { media: "(prefers-color-scheme: dark)", color: "#0d0d0d" },
  ],
};

/**
 * The inline script stamps `data-theme` before first paint so the page never
 * flashes the wrong palette. It is the only source of truth for the theme —
 * globals.css declares dark values under that attribute alone.
 */
const THEME_SCRIPT = `
try {
  var saved = localStorage.getItem("ews-theme");
  var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = saved || (prefersDark ? "dark" : "light");
} catch (_) {
  document.documentElement.dataset.theme = "light";
}
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
