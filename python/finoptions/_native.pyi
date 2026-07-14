import numpy as np
import numpy.typing as npt

class InputError(ValueError): ...
class BatchError(ValueError): ...
class ExecutionError(RuntimeError): ...
class SimulationError(RuntimeError): ...
class LsmError(RuntimeError): ...

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
    npt.NDArray[np.float64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]

class CpuExecutor:
    def __init__(self, threads: int) -> None: ...
    @property
    def threads(self) -> int: ...

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
) -> tuple[
    float,
    int,
    float | None,
    float | None,
    list[int],
    list[int],
    list[NativeExerciseDiagnostic],
]: ...
def price_lsm_call_batch_continuous_yield_f64(
    initial_spots: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    maturities: npt.NDArray[np.float64],
    constant_continuously_compounded_rates: npt.NDArray[np.float64],
    constant_volatilities: npt.NDArray[np.float64],
    annualized_dividend_yields: npt.NDArray[np.float64],
    base_intervals: int,
    origin_exercise: str,
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
    base_intervals: int,
    origin_exercise: str,
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
    base_intervals: int,
    origin_exercise: str,
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
) -> NativeBatchLsmResult: ...
