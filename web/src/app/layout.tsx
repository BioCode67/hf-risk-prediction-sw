import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "활력징후 조기경보 대시보드",
  description: "개인 기저선 이탈 기반의 설명가능한 조기경보 — 연구·교육용 데모",
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
