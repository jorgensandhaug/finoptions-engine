set shell := ["zsh", "-cu"]

sync:
    uv sync --all-groups --locked

native profile="dev-fast": sync
    uv run --no-sync maturin develop --profile {{profile}}

fmt: sync
    cargo fmt --all
    uv run --no-sync ruff format .

check: sync
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test --workspace --all-features --exclude finoptions-python --lib --bins --tests --examples
    cargo test -p finoptions-python --no-default-features --lib
    cargo test --workspace --all-features --doc --exclude finoptions-python
    uv run --no-sync ruff format --check .
    uv run --no-sync ruff check .
    uv run --no-sync ty check --error-on-warning python tests benchmarks

test: native
    uv run --no-sync pytest -q

bench-bootstrap threads="10": native
    uv run --no-sync python benchmarks/stationary_bootstrap.py --threads {{threads}} --mean-block-length 5 --mean-block-length 10 --mean-block-length 20 --mean-block-length 40 --mean-block-length 60

bench-grouped-lsm threads="14" paths="32768" math_backend="rust_std_f64": native
    uv run --no-sync python benchmarks/grouped_lsm_large_paths.py --threads {{threads}} --paths {{paths}} --math-backend {{math_backend}}

bench-polars-boundary rows="1000000" dividends="continuous_yield" iterations="6": native
    uv run --no-sync python benchmarks/polars_boundary.py --rows {{rows}} --dividends {{dividends}} --iterations {{iterations}}

bench-polars-groups groups="100000" strike_rows="1000000" dividends="continuous_yield" iterations="6": native
    uv run --no-sync python benchmarks/polars_grouped_boundary.py --groups {{groups}} --strike-rows {{strike_rows}} --dividends {{dividends}} --iterations {{iterations}}

full: check test
    uv build
