import numpy as np
import numpy.typing as npt

class InputError(ValueError): ...
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

class CpuExecutor:
    def __init__(self, threads: int) -> None: ...
    @property
    def threads(self) -> int: ...

def build_info() -> tuple[str, str]: ...
def validate_cash_dividend_amount(amount: float) -> None: ...
def validate_constant_volatility(annualized_decimal: float) -> None: ...
def validate_event_schedule(
    times: list[float],
    dividend_nodes: list[int],
    dividend_amounts: list[float],
    exercise_nodes: list[int],
    expiry_node: int,
) -> None: ...
def validate_log_discount_curve(times: list[float], log_discounts: list[float]) -> None: ...
def simulate_gbm_f64(
    initial_spot: float,
    schedule_times: list[float],
    curve_times: list[float],
    log_discounts: list[float],
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
