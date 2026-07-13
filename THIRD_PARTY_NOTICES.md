# Third-party notices

## Rust Rand / rand_distr

The bounded ZIGNOR standard-normal implementation and its generated lookup tables in
`finoptions-sim` are adapted from `rand_distr` 0.6.0 and the Rust Rand project. Those sources are
offered under either Apache License 2.0 or MIT, at the user's choice, and retain their source-file
copyright notices.

- Source: <https://github.com/rust-random/rand>
- Apache license: `third_party/rust-random/LICENSE-APACHE`
- MIT license: `third_party/rust-random/LICENSE-MIT`
- Copyright notice: `third_party/rust-random/COPYRIGHT`

## Random123 Philox

The Philox4x32-10 implementation follows the counter-based algorithm, constants, round count, and
published known-answer vectors from Salmon, Moraes, Dror, and Shaw, “Parallel Random Numbers: As
Easy as 1, 2, 3” (SC11), and was cross-checked against `rand_philox` 0.1.0. The implementation is
covered by this repository's license; the paper/reference attribution and known-answer tests are
retained. Random123 itself is available under a BSD-style license from its authors.

- Project and paper: <https://random123.com/>
- Rust cross-check source: <https://github.com/pawlenartowicz/rand_philox>

## BLAKE3

`blake3` 1.8.5 is used unmodified to identify externally supplied simulation populations so the
same population cannot silently be used for both policy fitting and evaluation. The project is
offered under CC0 1.0, Apache License 2.0, or Apache License 2.0 with the LLVM exception; this
distribution uses it under Apache License 2.0.

- Source: <https://github.com/BLAKE3-team/BLAKE3>
- Apache license: `third_party/rust-random/LICENSE-APACHE`
- Copyright and license choice: `third_party/blake3/COPYRIGHT`
