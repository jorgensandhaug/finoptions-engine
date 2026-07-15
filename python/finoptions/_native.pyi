import numpy as np
import numpy.typing as npt

class InputError(ValueError): ...
class BatchError(ValueError): ...
class ExecutionError(RuntimeError): ...
class SimulationError(RuntimeError): ...
class LsmError(RuntimeError): ...
class QuadratureError(RuntimeError): ...
class AmericanOptionError(RuntimeError): ...

type NativeExerciseDiagnostic = tuple[
    int,
    str,
    bool | None,
    float | None,
    int | None,
    int | None,
    int | None,
    int | None,
    float | None,
    float | None,
    float | None,
]
type NativeBatchLsmResult = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]

class CpuExecutor:
    def __init__(self, threads: int, math_backend: str = "rust_std_f64") -> None: ...
    @property
    def threads(self) -> int: ...
    @property
    def math_backend(self) -> str: ...

class EventAlignedQuadrature:
    def __init__(self, upper_stock: float, stock_points: int, normal_order: int) -> None: ...
    @property
    def upper_stock(self) -> float: ...
    @property
    def stock_points(self) -> int: ...
    @property
    def normal_order(self) -> int: ...

class AndersenLake:
    def __init__(
        self,
        boundary_quadrature_order: int,
        iterations: int,
        collocation_nodes: int,
        price_quadrature_order: int,
        maximum_relative_boundary_residual: float,
        maximum_relative_obstacle_projection: float,
    ) -> None: ...
    @property
    def boundary_quadrature_order(self) -> int: ...
    @property
    def iterations(self) -> int: ...
    @property
    def collocation_nodes(self) -> int: ...
    @property
    def price_quadrature_order(self) -> int: ...
    @property
    def maximum_relative_boundary_residual(self) -> float: ...
    @property
    def maximum_relative_obstacle_projection(self) -> float: ...

def build_info() -> tuple[str, str]: ...
def bin_dividends_f32(
    offsets: npt.NDArray[np.int64],
    times: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    amount_scale: str,
    bins: int,
    horizon: float,
    closure: str,
    maturity_boundary: str,
    executor: CpuExecutor,
    spots: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float32]: ...
def pack_padded_dividends_f32(
    offsets: npt.NDArray[np.int64],
    times: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    amount_scale: str,
    capacity: int,
    maturity_boundary: str,
    executor: CpuExecutor,
    spots: npt.NDArray[np.float64] | None,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.uint32],
]: ...
def validate_cash_dividend_amount(amount: float) -> None: ...
def validate_constant_volatility(annualized_decimal: float) -> None: ...
def validate_cumulative_dividend_yield_curve(
    times: list[float], cumulative_yields: list[float]
) -> None: ...
def validate_event_schedule(
    times: list[float],
    dividend_nodes: list[int],
    dividend_amounts: list[float],
    exercise_nodes: list[int],
    expiry_node: int,
) -> None: ...
def validate_log_discount_curve(times: list[float], log_discounts: list[float]) -> None: ...
def validate_ragged_cash_dividends_f64(
    offsets: npt.NDArray[np.int64],
    times: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
) -> None: ...
def validate_ragged_exercise_times_f64(
    offsets: npt.NDArray[np.int64],
    times: npt.NDArray[np.float64],
) -> None: ...
def simulate_gbm_f64(
    initial_spot: float,
    schedule_times: list[float],
    curve_times: list[float],
    log_discounts: list[float],
    dividend_yield_times: list[float],
    cumulative_dividend_yields: list[float],
    dividend_nodes: list[int],
    dividend_amounts: list[float],
    exercise_nodes: list[int],
    expiry_node: int,
    annualized_volatility: float,
    dividend_policy: str,
    executor: CpuExecutor,
    normals: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def simulate_gbm_seeded_f64(
    initial_spot: float,
    schedule_times: list[float],
    curve_times: list[float],
    log_discounts: list[float],
    dividend_yield_times: list[float],
    cumulative_dividend_yields: list[float],
    dividend_nodes: list[int],
    dividend_amounts: list[float],
    exercise_nodes: list[int],
    expiry_node: int,
    annualized_volatility: float,
    dividend_policy: str,
    executor: CpuExecutor,
    paths: int,
    seed: int,
    domain: int,
) -> npt.NDArray[np.float64]: ...
def price_lsm_call_seeded_f64(
    initial_spot: float,
    strike: float,
    schedule_times: list[float],
    curve_times: list[float],
    log_discounts: list[float],
    dividend_yield_times: list[float],
    cumulative_dividend_yields: list[float],
    dividend_nodes: list[int],
    dividend_amounts: list[float],
    exercise_nodes: list[int],
    expiry_node: int,
    annualized_volatility: float,
    dividend_policy: str,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain: int,
    evaluation_seed: int,
    evaluation_domain: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> tuple[
    float,
    int,
    float | None,
    float | None,
    list[int],
    list[int],
    list[NativeExerciseDiagnostic],
]: ...
def price_event_aligned_call_f64(
    initial_spot: float,
    strike: float,
    schedule_times: list[float],
    curve_times: list[float],
    log_discounts: list[float],
    dividend_yield_times: list[float],
    cumulative_dividend_yields: list[float],
    dividend_nodes: list[int],
    dividend_amounts: list[float],
    exercise_nodes: list[int],
    expiry_node: int,
    annualized_volatility: float,
    dividend_policy: str,
    executor: CpuExecutor,
    method: EventAlignedQuadrature,
) -> tuple[float, float]: ...
def price_andersen_lake_call_grouped_f64(
    group_offsets: npt.NDArray[np.int64],
    group_spots: npt.NDArray[np.float64],
    group_maturities: npt.NDArray[np.float64],
    group_constant_continuously_compounded_rates: npt.NDArray[np.float64],
    group_annualized_dividend_yields: npt.NDArray[np.float64],
    group_constant_volatilities: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    executor: CpuExecutor,
    method: AndersenLake,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]: ...
def price_lsm_call_batch_continuous_yield_f64(
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    annualized_dividend_yields: npt.NDArray[np.float64],
    exercise_mesh: str,
    base_intervals: int | None,
    origin_exercise: str | None,
    explicit_exercise_offsets: npt.NDArray[np.int64] | None,
    explicit_exercise_times: npt.NDArray[np.float64] | None,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain_start: int,
    evaluation_seed: int,
    evaluation_domain_start: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> NativeBatchLsmResult: ...
def price_lsm_call_batch_cash_dividends_f64(
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    dividend_offsets: npt.NDArray[np.int64],
    dividend_times: npt.NDArray[np.float64],
    dividend_amounts: npt.NDArray[np.float64],
    dividend_policy: str,
    maturity_boundary: str,
    exercise_mesh: str,
    base_intervals: int | None,
    origin_exercise: str | None,
    explicit_exercise_offsets: npt.NDArray[np.int64] | None,
    explicit_exercise_times: npt.NDArray[np.float64] | None,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain_start: int,
    evaluation_seed: int,
    evaluation_domain_start: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> NativeBatchLsmResult: ...
def price_lsm_call_batch_escrowed_cash_dividends_f64(
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    dividend_offsets: npt.NDArray[np.int64],
    dividend_times: npt.NDArray[np.float64],
    dividend_amounts: npt.NDArray[np.float64],
    volatility_convention: str,
    maturity_boundary: str,
    exercise_mesh: str,
    base_intervals: int | None,
    origin_exercise: str | None,
    explicit_exercise_offsets: npt.NDArray[np.int64] | None,
    explicit_exercise_times: npt.NDArray[np.float64] | None,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain_start: int,
    evaluation_seed: int,
    evaluation_domain_start: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> NativeBatchLsmResult: ...
def price_lsm_call_grouped_batch_continuous_yield_f64(
    group_offsets: npt.NDArray[np.uint32],
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    annualized_dividend_yields: npt.NDArray[np.float64],
    exercise_mesh: str,
    base_intervals: int | None,
    origin_exercise: str | None,
    explicit_exercise_offsets: npt.NDArray[np.int64] | None,
    explicit_exercise_times: npt.NDArray[np.float64] | None,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain_start: int,
    evaluation_seed: int,
    evaluation_domain_start: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> NativeBatchLsmResult: ...
def price_lsm_call_grouped_batch_cash_dividends_f64(
    group_offsets: npt.NDArray[np.uint32],
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    dividend_offsets: npt.NDArray[np.int64],
    dividend_times: npt.NDArray[np.float64],
    dividend_amounts: npt.NDArray[np.float64],
    dividend_policy: str,
    maturity_boundary: str,
    exercise_mesh: str,
    base_intervals: int | None,
    origin_exercise: str | None,
    explicit_exercise_offsets: npt.NDArray[np.int64] | None,
    explicit_exercise_times: npt.NDArray[np.float64] | None,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain_start: int,
    evaluation_seed: int,
    evaluation_domain_start: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> NativeBatchLsmResult: ...
def price_lsm_call_grouped_batch_escrowed_cash_dividends_f64(
    group_offsets: npt.NDArray[np.uint32],
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    dividend_offsets: npt.NDArray[np.int64],
    dividend_times: npt.NDArray[np.float64],
    dividend_amounts: npt.NDArray[np.float64],
    volatility_convention: str,
    maturity_boundary: str,
    exercise_mesh: str,
    base_intervals: int | None,
    origin_exercise: str | None,
    explicit_exercise_offsets: npt.NDArray[np.int64] | None,
    explicit_exercise_times: npt.NDArray[np.float64] | None,
    executor: CpuExecutor,
    training_paths: int,
    evaluation_paths: int,
    training_seed: int,
    training_domain_start: int,
    evaluation_seed: int,
    evaluation_domain_start: int,
    polynomial_degree: int,
    rank_tolerance: float,
    condition_limit: float,
    tie_rule: str,
    insufficient_exercise_support: str,
) -> NativeBatchLsmResult: ...
