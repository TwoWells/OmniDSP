# OmniDSP Agent Context

This file serves as the single point of truth for AI agents working on the OmniDSP project.

## Project

- **Goal:** Composable DSP modules built on pluggable primitive traits.
- **Repository:** `TwoWells/OmniDSP` on GitHub.
- **License:** AGPL-3.0-or-later with commercial license option.

Rust workspace. `omnidsp-core` is the library (traits, modules, omni implementations). `omnidsp` (future) is the dispatch convenience crate.

## Coding Standards

- **Edition:** Rust 2024.
- **Safety:** `unsafe` code is strictly forbidden (`forbid(unsafe_code)`).
- **Error Handling:** Use `thiserror` for all errors. No `anyhow` — this is a library.
- **Strict Denials:** Do NOT use `unwrap()`, `panic!()`, `todo!()`, `unimplemented!()`, `dbg!()`, `println!()`, or `eprintln!()`. Use proper error handling. `expect()` is denied in production code but allowed in `#[cfg(test)]` modules — prefer `expect("reason")` over workarounds in tests.
- **Assertions:** All `assert!()`, `assert_eq!()`, and `assert_ne!()` calls must include a message explaining what failed.
- **Imports:** No wildcard imports (`use crate::*`).
- **Formatting:** Code must be formatted with `rustfmt`.
- **Linting:** Must pass `cargo clippy` with `pedantic`, `nursery`, and `cargo` groups enabled. Every `#[allow(...)]` must include a `reason` string.

## Commit Convention

- **Format:** [Conventional Commits](https://www.conventionalcommits.org/) (enforced by commit-msg hook).
- **Pattern:** `type(scope): description` or `type: description`
- **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
- **Breaking changes:** append `!` before colon, e.g. `feat(dft)!: change plan return type`
- **Examples:**
  - `feat(dft): add forward FFT plan`
  - `fix: correct buffer size validation`
  - `test(window): add Hann window reference test`
  - `chore: bump version to 0.2.0`

## Quality Standards

- **License Compliance:** All new dependencies MUST have permissive licenses (MIT, Apache-2.0, etc.) as specified in `@./deny.toml`. OmniDSP is dual-licensed under AGPL-3.0-or-later and a commercial license.
- **Copyright Headers:** Every `.rs` file must start with the SPDX header (enforced by pre-commit hook):
  ```
  // SPDX-License-Identifier: AGPL-3.0-or-later
  // Copyright (C) 2026 Two Wells <contact@twowells.dev>
  ```
- **Documentation:** All public APIs must have documentation comments.
- **No internal IDs in source:** Code and doc comments must read as self-contained prose. Never cite internal design artifacts — ADR numbers (`ADR-NNN`), surface-lock ticket IDs (`SL-12a`), workstream/issue references — in published source: a reader on docs.rs or in the AGPL source tree cannot resolve them. State the *rationale* in words; the decision record lives in the private OmniDSPInternal repo, not the code. External citations (papers, ISO standards, scipy/librosa) are fine and encouraged.
- **Testing:** All new features must include tests.

## Setup

- **First time:** `make setup` — configures git hooks and checks for required cargo tools.
- **Required tools:** cargo-deny, cargo-machete, cargo-nextest, cargo-mutants.
- **Install tools:** `cargo binstall cargo-deny cargo-machete cargo-nextest cargo-mutants`

## Development Commands

- **Check (full):** `make check` — format, lint, deny, machete, and test in one pass.
- **Test (all):** `make test`
- **Test (filtered):** `make test T=<filter>`
- **Test (repeat):** `make test T=<filter> N=<count>`
- **Bench (CQT throughput):** `make bench` — multirate vs single-FFT CQT throughput (`omnidsp-core`, `bench` feature).

### Demo / WASM (DEMO-00)

- **Floor wasm build:** `make wasm-check` — smoke-builds the `RustBackend` floor (`omnidsp`) for `wasm32-unknown-unknown` with `simd128`. Proves the floor *compiles* for the browser (not real-time headroom — that needs the running demo). Installs the target if missing.
- **WASM engine:** `make wasm-pack` — builds the `omnidsp-wasm` binding (outside the lint workspace) for the browser (`+simd128`); emits the `pkg/` ESM glue if `wasm-pack` is installed, otherwise a plain target build.
- **Run the demo:** `make demo` — builds the wasm engine and runs the Vite dev server for the CQT visualizer. Requires `wasm-pack` and `npm`.

## Release Workflow

- **Patch Release:** `make release-patch`
- **Minor Release:** `make release-minor`
- **Major Release:** `make release-major`
- **Custom Version:** `make release V=x.y.z`

Release runs: check → commit → tag.
