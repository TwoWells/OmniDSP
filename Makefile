# OmniDSP Makefile
# Usage:
#   make setup           # configure hooks + check tools (first time)
#   make check           # fmt, lint, deny, machete, test
#   make release-patch   # 0.1.0 -> 0.1.1
#   make release-minor   # 0.1.0 -> 0.2.0
#   make release-major   # 0.1.0 -> 1.0.0
#   make release V=0.2.0 # explicit version

.PHONY: bench build-release wasm-check wasm-pack demo check deny doc onemkl-check ipp-check gen-cqt-reference gen-cqt-process-reference gen-cqt-librosa-reference gen-fir-reference gen-fir-lfilter-reference gen-remez-reference gen-hilbert-reference gen-iir-reference gen-iir-sosfilt-reference gen-resample-reference gen-resample-poly-reference gen-xcorr-reference machete mutants setup setup-hooks setup-tools test release release-patch release-minor release-major publish tag-current

# Get current version from Cargo.toml
CURRENT_VERSION := $(shell grep '^version = ' omnidsp-core/Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

# Required cargo tools
CARGO_TOOLS := cargo-deny cargo-machete cargo-nextest cargo-mutants

# Hard per-process address-space cap (KiB) applied to test runs, so a runaway
# allocation (e.g. an unbounded loop in a test) aborts that single process at
# the limit instead of exhausting system RAM. Override with MEMLIMIT_KB=<kib>,
# or MEMLIMIT_KB=unlimited to disable.
MEMLIMIT_KB ?= 8388608

# --- Setup ---

# One-time setup: configure hooks and check tools
setup: setup-hooks setup-tools

# Configure git hooks (explicit opt-in, not run by check)
setup-hooks:
	@current=$$(git config core.hooksPath 2>/dev/null); \
	 if [ "$$current" = ".githooks" ]; then \
	   echo "hooks: already configured"; \
	 else \
	   git config core.hooksPath .githooks; \
	   echo "hooks: configured .githooks"; \
	 fi

# Report cargo tool status
setup-tools:
	@missing=0; \
	 for tool in $(CARGO_TOOLS); do \
	   if ! command -v $$tool >/dev/null 2>&1 && ! cargo --list 2>/dev/null | grep -qw "$${tool#cargo-}"; then \
	     echo "  missing: $$tool"; \
	     missing=1; \
	   else \
	     echo "  ok: $$tool"; \
	   fi; \
	 done; \
	 if [ $$missing -eq 1 ]; then \
	   echo ""; \
	   echo "Install missing tools with:"; \
	   echo "  cargo binstall cargo-deny cargo-machete cargo-nextest cargo-mutants"; \
	   echo ""; \
	   echo "Or with cargo install (slower, builds from source):"; \
	   echo "  cargo install cargo-deny cargo-machete cargo-nextest cargo-mutants"; \
	 else \
	   echo "All tools present."; \
	 fi

# --- Check ---

build-release:
	@cargo build --release

# Multirate vs single-FFT CQT throughput (exposes the SingleFftCqt baseline).
bench:
	@cargo bench -p omnidsp-core --features bench

check: setup-tools
	@PINNED=$$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml); \
	 LATEST=$$(rustup run stable rustc --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1); \
	 if [ -n "$$LATEST" ] && [ "$$PINNED" != "$$LATEST" ]; then \
	   printf '\033[33mNote: rust-toolchain.toml pins %s, latest stable is %s\033[0m\n' "$$PINNED" "$$LATEST"; \
	 fi
	@cargo update --quiet
	@cargo fmt -- -l | sed 's/^/fmt: formatted /'
	@cargo clippy --tests --quiet -- -D warnings
	@# The CQT bench is feature-gated (required-features = ["bench"]), so the
	@# default clippy pass above skips it. Lint it explicitly so it cannot rot
	@# unnoticed the way it did under the surface-lock landing.
	@cargo clippy -p omnidsp-core --benches --features bench --quiet -- -D warnings
	@tries=0; while true; do \
	   cargo deny --log-level error check; rc=$$?; \
	   if [ $$rc -eq 0 ]; then break; \
	   elif [ $$rc -ne 139 ]; then exit $$rc; \
	   else \
	     tries=$$((tries + 1)); \
	     if [ $$tries -ge 5 ]; then echo "cargo-deny segfaulted 5 times, giving up"; exit 139; fi; \
	     echo "cargo-deny segfaulted (EmbarkStudios/cargo-deny#855), retry $$tries/5..."; \
	   fi; \
	 done
	@cargo machete --skip-target-dir
	@cargo doc --no-deps --document-private-items --quiet
	@if [ "$(MEMLIMIT_KB)" != unlimited ]; then ulimit -v $(MEMLIMIT_KB); fi; \
	 cargo nextest run --workspace --no-fail-fast --no-tests=pass --status-level fail --final-status-level fail --cargo-quiet --show-progress only
	@cargo test --workspace --doc --quiet

deny:
	@cargo deny --log-level error check

doc:
	@cargo doc --no-deps --document-private-items

# --- Intel oneMKL vendor gate ---

# Build, lint, and test the Intel oneMKL vendor crates. Requires Intel oneMKL
# (libmkl_rt) at link time, so it runs ONLY in the `ghcr.io/twowells/omnidsp-ci`
# image (MKLROOT preset) or on a host with oneMKL installed — the floor
# `make check` deliberately excludes these crates (it cannot link MKL). The
# crates are workspace-excluded, so they build by manifest path, not `-p`
# (mirrors the wasm targets). The conformance one-liner inside
# `omnidsp-onemkl/tests/` holds the backend to the same golden vectors as the
# floor, so a vendor cannot silently drift.
onemkl-check:
	@cargo fmt --manifest-path omnidsp-onemkl-sys/Cargo.toml -- --check
	@cargo fmt --manifest-path omnidsp-onemkl/Cargo.toml -- --check
	@cargo clippy --manifest-path omnidsp-onemkl-sys/Cargo.toml --all-targets -- -D warnings
	@cargo clippy --manifest-path omnidsp-onemkl/Cargo.toml --all-targets -- -D warnings
	@cargo nextest run --manifest-path omnidsp-onemkl-sys/Cargo.toml --no-fail-fast --no-tests=pass
	@cargo nextest run --manifest-path omnidsp-onemkl/Cargo.toml --no-fail-fast --no-tests=pass
	@cargo test --manifest-path omnidsp-onemkl/Cargo.toml --doc
	@echo "onemkl-check: oneMKL vendor crates build, lint, and pass conformance"

# Throughput head-to-head vs the rustfft/realfft + scalar floor: DFT (c2c/r2c)
# and VecOps (mul/cmul/dot). Built with target-cpu=native so the Rust floor gets
# the same vector ISA oneMKL dispatches to (MKL ships precompiled + runtime
# dispatch, unaffected by RUSTFLAGS). Requires Intel oneMKL at link time, so it
# runs in the `omnidsp-ci` image. Coarse batched wall-clock (warmup + median) —
# the same-machine same-run ratio is the figure of merit.
onemkl-bench:
	@RUSTFLAGS="-C target-cpu=native" cargo bench --manifest-path omnidsp-onemkl/Cargo.toml

# --- Intel IPP vendor gate ---

# Build, lint, and test the Intel IPP vendor crates. Requires Intel IPP
# (ipps/ippvm/ippcore) at link time, so it runs ONLY in the
# `ghcr.io/twowells/omnidsp-ci` image (IPPROOT preset) or on a host with IPP
# installed — the floor `make check` deliberately excludes these crates (it
# cannot link IPP). The crates are workspace-excluded, so they build by manifest
# path, not `-p` (mirrors the oneMKL gate). The `-sys` smoke test links `ipps`
# and calls `ippsGetLibVersion`, so a green run proves the dynamic link resolves;
# the `omnidsp-ipp` wrapper is then held to the shared conformance golden vectors
# (the `run_all` one-liner in `omnidsp-ipp/tests/`), so the vendor cannot silently
# drift from the floor.
ipp-check:
	@cargo fmt --manifest-path omnidsp-ipp-sys/Cargo.toml -- --check
	@cargo fmt --manifest-path omnidsp-ipp/Cargo.toml -- --check
	@cargo clippy --manifest-path omnidsp-ipp-sys/Cargo.toml --all-targets -- -D warnings
	@cargo clippy --manifest-path omnidsp-ipp/Cargo.toml --all-targets -- -D warnings
	@cargo nextest run --manifest-path omnidsp-ipp-sys/Cargo.toml --no-fail-fast --no-tests=pass
	@cargo nextest run --manifest-path omnidsp-ipp/Cargo.toml --no-fail-fast --no-tests=pass
	@cargo test --manifest-path omnidsp-ipp/Cargo.toml --doc
	@echo "ipp-check: IPP vendor crates build, lint, and pass conformance"

# --- WASM floor (demo prerequisite) ---

# Smoke-build the RustBackend floor for the browser target the CQT visualiser
# runs on. Confirms omnidsp + rustfft/realfft
# compile to wasm32-unknown-unknown with simd128 — the auto-vectorisation knob
# the in-browser real-time budget depends on. Installs the target if missing.
# This proves the floor *builds* for wasm; it does NOT prove real-time headroom
# (that needs the running demo).
wasm-check:
	@rustup target list --installed | grep -qx wasm32-unknown-unknown \
	  || rustup target add wasm32-unknown-unknown
	@RUSTFLAGS="-C target-feature=+simd128" \
	  cargo build --target wasm32-unknown-unknown -p omnidsp
	@echo "wasm-check: omnidsp builds for wasm32-unknown-unknown (+simd128)"

# Build the visualiser engine (`omnidsp-wasm`, the wasm-bindgen binding
# the browser app consumes). This crate is OUTSIDE the core lint workspace
# (root Cargo.toml `[workspace.exclude]`), so it is built by manifest path, not
# `-p`. If `wasm-pack` is installed it emits the `pkg/` ESM glue for the app;
# otherwise it falls back to a plain target build, which is sufficient proof the
# crate compiles for the browser (+simd128).
wasm-pack:
	@rustup target list --installed | grep -qx wasm32-unknown-unknown \
	  || rustup target add wasm32-unknown-unknown
	@if command -v wasm-pack >/dev/null 2>&1; then \
	  RUSTFLAGS="-C target-feature=+simd128" \
	    wasm-pack build omnidsp-wasm --release --target web; \
	  echo "wasm-pack: omnidsp-wasm packaged to omnidsp-wasm/pkg (+simd128)"; \
	else \
	  echo "wasm-pack not found; falling back to a plain target build"; \
	  RUSTFLAGS="-C target-feature=+simd128" \
	    cargo build --manifest-path omnidsp-wasm/Cargo.toml \
	    --release --target wasm32-unknown-unknown; \
	  echo "wasm-pack: omnidsp-wasm builds for wasm32-unknown-unknown (+simd128); install wasm-pack to emit pkg/ glue"; \
	fi

# Build the wasm engine and run the demo visualiser dev server.
# Requires `wasm-pack` (the app imports the pkg/ glue it emits) and npm. Opens a
# local Vite dev server; Ctrl-C to stop.
demo: wasm-pack
	@if ! command -v wasm-pack >/dev/null 2>&1; then \
	  echo "demo: wasm-pack is required (cargo binstall wasm-pack) — the app imports omnidsp-wasm/pkg"; \
	  exit 1; \
	fi
	@cd demo && npm install && npm run dev

# --- Reference data ---

gen-fir-reference:
	@python3 scripts/gen_fir_reference.py > omnidsp-testdata/src/fir_scipy.rs
	@echo "Generated omnidsp-testdata/src/fir_scipy.rs"

gen-fir-lfilter-reference:
	@python3 scripts/gen_fir_lfilter_reference.py > omnidsp-testdata/src/fir_lfilter_scipy.rs
	@echo "Generated omnidsp-testdata/src/fir_lfilter_scipy.rs"

gen-remez-reference:
	@python3 scripts/gen_remez_reference.py > omnidsp-testdata/src/remez_scipy.rs
	@echo "Generated omnidsp-testdata/src/remez_scipy.rs"

gen-cqt-reference:
	@python3 scripts/gen_cqt_reference.py > omnidsp-testdata/src/cqt_numpy.rs
	@echo "Generated omnidsp-testdata/src/cqt_numpy.rs"

gen-cqt-process-reference:
	@python3 scripts/gen_cqt_process_reference.py > omnidsp-testdata/src/cqt_process_numpy.rs
	@echo "Generated omnidsp-testdata/src/cqt_process_numpy.rs"

gen-cqt-librosa-reference:
	@python3 scripts/gen_cqt_librosa_reference.py > omnidsp-testdata/src/cqt_librosa.rs
	@echo "Generated omnidsp-testdata/src/cqt_librosa.rs"

gen-iir-reference:
	@python3 scripts/gen_iir_reference.py > omnidsp-testdata/src/iir_scipy.rs
	@echo "Generated omnidsp-testdata/src/iir_scipy.rs"

gen-iir-sosfilt-reference:
	@python3 scripts/gen_iir_sosfilt_reference.py > omnidsp-testdata/src/iir_sosfilt_scipy.rs
	@echo "Generated omnidsp-testdata/src/iir_sosfilt_scipy.rs"

gen-resample-reference:
	@python3 scripts/gen_resample_reference.py > omnidsp-testdata/src/resample_scipy.rs
	@echo "Generated omnidsp-testdata/src/resample_scipy.rs"

gen-resample-poly-reference:
	@python3 scripts/gen_resample_poly_reference.py > omnidsp-testdata/src/resample_poly_scipy.rs
	@echo "Generated omnidsp-testdata/src/resample_poly_scipy.rs"

gen-hilbert-reference:
	@python3 scripts/gen_hilbert_reference.py > omnidsp-testdata/src/hilbert_scipy.rs
	@echo "Generated omnidsp-testdata/src/hilbert_scipy.rs"

gen-dct-reference:
	@python3 scripts/gen_dct_reference.py > omnidsp-testdata/src/dct_scipy.rs
	@echo "Generated omnidsp-testdata/src/dct_scipy.rs"

gen-xcorr-reference:
	@python3 scripts/gen_xcorr_reference.py > omnidsp-testdata/src/xcorr_scipy.rs
	@echo "Generated omnidsp-testdata/src/xcorr_scipy.rs"

machete:
	@cargo machete --skip-target-dir

# --- Test ---

mutants:
	@cargo mutants --timeout 60

# Run tests. Pass T= to filter, N= to repeat.
CLEAN_T = $(subst \,,$(subst !,,$(T)))
test:
	@if [ "$(MEMLIMIT_KB)" != unlimited ]; then ulimit -v $(MEMLIMIT_KB); fi; \
	 cargo nextest run --workspace --status-level fail --final-status-level slow --cargo-quiet $(if $(N),--stress-count $(N),) $(if $(T),$(if $(findstring !,$(T)),-E 'not test($(CLEAN_T))',-E 'test($(T))'),)

# --- Release ---

pre-release-check:
	@echo "Checking release prerequisites..."
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Working tree is not clean. Commit or stash changes first."; \
		exit 1; \
	fi
	@if [ "$$(git branch --show-current)" != "main" ]; then \
		echo "Error: Not on main branch."; \
		exit 1; \
	fi
	@git fetch origin main --quiet
	@if [ "$$(git rev-parse HEAD)" != "$$(git rev-parse origin/main)" ]; then \
		echo "Error: Local main is not up to date with origin/main."; \
		exit 1; \
	fi
	@echo "Prerequisites OK."

bump-version:
	@if [ -z "$(V)" ]; then \
		echo "Error: Version not specified. Use V=x.y.z"; \
		exit 1; \
	fi
	@echo "Bumping version: $(CURRENT_VERSION) -> $(V)"
	@sed -i 's/^version = "$(CURRENT_VERSION)"/version = "$(V)"/' omnidsp-core/Cargo.toml
	@cargo check --quiet
	@echo "Version bumped to $(V)"

next-patch:
	$(eval V := $(shell echo $(CURRENT_VERSION) | awk -F. '{print $$1"."$$2"."$$3+1}'))

next-minor:
	$(eval V := $(shell echo $(CURRENT_VERSION) | awk -F. '{print $$1"."$$2+1".0"}'))

next-major:
	$(eval V := $(shell echo $(CURRENT_VERSION) | awk -F. '{print $$1+1".0.0"}'))

release: pre-release-check
	@if [ -z "$(V)" ]; then \
		echo "Error: Version not specified. Use 'make release V=x.y.z' or 'make release-patch'"; \
		exit 1; \
	fi
	@cargo update --quiet
	@$(MAKE) bump-version V=$(V)
	@if ! $(MAKE) check; then \
		echo "Checks failed. Rolling back version bump..."; \
		git checkout HEAD -- omnidsp-core/Cargo.toml Cargo.lock; \
		exit 1; \
	fi
	@git add omnidsp-core/Cargo.toml Cargo.lock
	@if ! git commit -m "chore: Bump version to $(V)"; then \
		echo "Commit failed. Rolling back version bump..."; \
		git checkout HEAD -- omnidsp-core/Cargo.toml Cargo.lock; \
		exit 1; \
	fi
	@git tag -a "v$(V)" -m "Release v$(V)"
	@echo ""
	@echo "Release v$(V) prepared locally."
	@echo "Run 'make publish' to push and create the release."

release-patch: pre-release-check next-patch
	@$(MAKE) release V=$(V)

release-minor: pre-release-check next-minor
	@$(MAKE) release V=$(V)

release-major: pre-release-check next-major
	@$(MAKE) release V=$(V)

publish:
	@echo "Pushing to origin..."
	@git push && git push --tags
	@echo ""
	@echo "Release v$(CURRENT_VERSION) pushed."

tag-current:
	@if git rev-parse "v$(CURRENT_VERSION)" >/dev/null 2>&1; then \
		echo "Tag v$(CURRENT_VERSION) already exists."; \
		exit 1; \
	fi
	@echo "Creating tag v$(CURRENT_VERSION) for current version..."
	@git tag -a "v$(CURRENT_VERSION)" -m "Release v$(CURRENT_VERSION)"
	@echo "Tag created. Run 'make publish' to push and release."

version:
	@echo "Current version: $(CURRENT_VERSION)"
	@echo "Latest tag:      $$(git describe --tags --abbrev=0 2>/dev/null || echo 'none')"
