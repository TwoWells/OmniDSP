// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>
import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset URLs so the built `dist/` works on static hosting served
  // from a sub-path (GitHub Pages project pages).
  base: "./",
  server: {
    // The wasm-pack output lives in ../omnidsp-wasm/pkg (outside the demo
    // root). Allow Vite to read the parent so the glue module + the `?url`
    // wasm asset resolve in dev.
    fs: { allow: [".."] },
  },
  build: {
    target: "es2022",
  },
});
