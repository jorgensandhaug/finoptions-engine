set shell := ["zsh", "-cu"]

sync:
    uv sync --all-groups

native profile="dev-fast":
    uv run --no-sync maturin develop --profile {{profile}}

fmt:
    cargo fmt --all
    uv run --no-sync ruff format .

check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test --workspace --all-targets
    uv run --no-sync ruff format --check .
    uv run --no-sync ruff check .
    uv run --no-sync mypy

test: native
    uv run --no-sync pytest -q

full: check test
    uv build

