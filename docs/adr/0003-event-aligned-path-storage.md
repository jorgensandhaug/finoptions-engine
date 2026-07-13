# ADR 0003: Event-aligned time-major path storage

Status: accepted

The first path kernel simulates exact adjacent GBM transitions on the validated event grid. It
stores only states required by backward induction: every configured exercise node and terminal
expiry. Rows are time-major and paths are contiguous within a row, matching regression and payoff
access while avoiding a full node-by-path history.

An observed state is always the pre-event state. At a node shared by exercise and a cash dividend,
the simulator records the exercise state first, then applies the explicitly selected limited-
liability dividend policy before the next transition. Expiry remains a separate terminal payoff
node and never becomes a regression decision.

The initial kernel accepts a contiguous time-major standard-normal buffer. Its entire shape and
finiteness are validated before output mutation. A CPU executor declares an exact positive thread
count and owns an isolated worker pool; one thread selects the serial kernel, while larger counts
select the parallel kernel. It never reads or mutates the global Rayon pool and never falls back to
another worker count or backend. Serial and parallel kernels preserve bitwise-identical path
identity because every path performs the same ordered scalar operations.

Random-number generation, batching, and device backends remain separate components and may not
change path identity or event semantics. There is no implicit backend, precision, policy, or
recovery fallback.
