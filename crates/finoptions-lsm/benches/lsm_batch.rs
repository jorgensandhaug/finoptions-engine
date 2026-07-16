use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use finoptions_batch::{MaturityBoundary, RaggedCashDividends, RaggedExerciseTimes};
use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve, TimeGrid,
};
use finoptions_lsm::{
    BatchDividendModel, BatchLsmCallInputs, BatchLsmCallMethod, BatchRandomStreams,
    CashDividendExerciseMesh, ExerciseTieRule, ExplicitExerciseMesh, GroupedBatchLsmCallInputs,
    InsufficientExerciseSupportPolicy, LsmCallEstimator, OriginExercise, PolynomialDegree,
    PolynomialRegressionSpec, UniformExerciseMesh,
};
use finoptions_sim::{CpuExecutor, CpuMathBackend, GbmPathPlan, RandomStream};

const STEPS: u32 = 150;
const TRAINING_PATHS: usize = 500;
const EVALUATION_PATHS: usize = 500;
const TRAINING_SEED: u64 = 10_000;
const EVALUATION_SEED: u64 = 20_000;
const TRAINING_DOMAIN_START: u32 = 0;
const EVALUATION_DOMAIN_START: u32 = 1_000_000;

struct ScalarModel {
    plan: GbmPathPlan,
    curve: LogDiscountCurve,
}

struct Workload {
    initial_spots: Vec<f64>,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    rates: Vec<f64>,
    volatilities: Vec<f64>,
    dividend_offsets: Vec<i64>,
    dividend_times: Vec<f64>,
    dividend_amounts: Vec<f64>,
    scalar_models: Vec<ScalarModel>,
}

impl Workload {
    fn new(rows: usize) -> Self {
        let mut initial_spots = Vec::with_capacity(rows);
        let mut strikes = Vec::with_capacity(rows);
        let mut maturities = Vec::with_capacity(rows);
        let mut rates = Vec::with_capacity(rows);
        let mut volatilities = Vec::with_capacity(rows);
        let mut dividend_offsets = Vec::with_capacity(rows + 1);
        let mut dividend_times = Vec::with_capacity(rows);
        let mut dividend_amounts = Vec::with_capacity(rows);
        let mut scalar_models = Vec::with_capacity(rows);
        dividend_offsets.push(0);
        for row in 0..rows {
            let bucket = u32::try_from(row % 17).unwrap();
            let rate = 0.03 + f64::from(bucket) * 0.001;
            let volatility = 0.20 + f64::from(bucket) * 0.005;
            let cash_amount = 2.0 + f64::from(bucket) * 0.1;
            initial_spots.push(145.0 + f64::from(bucket));
            strikes.push(100.0 + f64::from(bucket % 5));
            maturities.push(1.0);
            rates.push(rate);
            volatilities.push(volatility);
            dividend_times.push(0.5);
            dividend_amounts.push(cash_amount);
            dividend_offsets.push(i64::try_from(row + 1).unwrap());
            scalar_models.push(scalar_model(rate, volatility, cash_amount));
        }
        Self {
            initial_spots,
            strikes,
            maturities,
            rates,
            volatilities,
            dividend_offsets,
            dividend_times,
            dividend_amounts,
            scalar_models,
        }
    }

    fn rows(&self) -> usize {
        self.initial_spots.len()
    }

    fn dividends(&self) -> RaggedCashDividends<'_> {
        RaggedCashDividends::try_new(
            &self.dividend_offsets,
            &self.dividend_times,
            &self.dividend_amounts,
        )
        .unwrap()
    }

    fn batch_inputs(&self) -> BatchLsmCallInputs<'_> {
        BatchLsmCallInputs::try_new(
            &self.initial_spots,
            &self.strikes,
            &self.maturities,
            &self.rates,
            &self.volatilities,
            BatchDividendModel::CashDividends {
                schedules: self.dividends(),
                policy: DividendPolicy::Liquidator,
                maturity_boundary: MaturityBoundary::StrictlyBefore,
            },
        )
        .unwrap()
    }
}

struct StrikeGroupWorkload {
    group_offsets: Vec<u32>,
    group_initial_spots: Vec<f64>,
    strikes: Vec<f64>,
    group_maturities: Vec<f64>,
    group_rates: Vec<f64>,
    group_volatilities: Vec<f64>,
    group_dividend_offsets: Vec<i64>,
    group_dividend_times: Vec<f64>,
    group_dividend_amounts: Vec<f64>,
    row_initial_spots: Vec<f64>,
    row_maturities: Vec<f64>,
    row_rates: Vec<f64>,
    row_volatilities: Vec<f64>,
    row_dividend_offsets: Vec<i64>,
    row_dividend_times: Vec<f64>,
    row_dividend_amounts: Vec<f64>,
}

impl StrikeGroupWorkload {
    fn new(groups: usize, strikes_per_group: usize) -> Self {
        let rows = groups * strikes_per_group;
        let mut workload = Self {
            group_offsets: Vec::with_capacity(groups + 1),
            group_initial_spots: Vec::with_capacity(groups),
            strikes: Vec::with_capacity(rows),
            group_maturities: Vec::with_capacity(groups),
            group_rates: Vec::with_capacity(groups),
            group_volatilities: Vec::with_capacity(groups),
            group_dividend_offsets: Vec::with_capacity(groups + 1),
            group_dividend_times: Vec::with_capacity(groups),
            group_dividend_amounts: Vec::with_capacity(groups),
            row_initial_spots: Vec::with_capacity(rows),
            row_maturities: Vec::with_capacity(rows),
            row_rates: Vec::with_capacity(rows),
            row_volatilities: Vec::with_capacity(rows),
            row_dividend_offsets: Vec::with_capacity(rows + 1),
            row_dividend_times: Vec::with_capacity(rows),
            row_dividend_amounts: Vec::with_capacity(rows),
        };
        workload.group_offsets.push(0);
        workload.group_dividend_offsets.push(0);
        workload.row_dividend_offsets.push(0);
        for group in 0..groups {
            let bucket = u32::try_from(group % 17).unwrap();
            let spot = 100.0 + f64::from(bucket);
            let maturity = 1.0;
            let rate = 0.03 + f64::from(bucket) * 0.001;
            let volatility = 0.20 + f64::from(bucket) * 0.005;
            let cash_amount = 2.0 + f64::from(bucket) * 0.1;
            workload.group_initial_spots.push(spot);
            workload.group_maturities.push(maturity);
            workload.group_rates.push(rate);
            workload.group_volatilities.push(volatility);
            workload.group_dividend_times.push(0.5);
            workload.group_dividend_amounts.push(cash_amount);
            workload
                .group_dividend_offsets
                .push(i64::try_from(group + 1).unwrap());
            for strike_index in 0..strikes_per_group {
                let strike_fraction = 0.82 + 0.03 * f64::from(u32::try_from(strike_index).unwrap());
                workload.strikes.push(spot * strike_fraction);
                workload.row_initial_spots.push(spot);
                workload.row_maturities.push(maturity);
                workload.row_rates.push(rate);
                workload.row_volatilities.push(volatility);
                workload.row_dividend_times.push(0.5);
                workload.row_dividend_amounts.push(cash_amount);
                workload
                    .row_dividend_offsets
                    .push(i64::try_from(workload.strikes.len()).unwrap());
            }
            workload
                .group_offsets
                .push(u32::try_from(workload.strikes.len()).unwrap());
        }
        workload
    }

    fn rows(&self) -> usize {
        self.strikes.len()
    }

    fn grouped_inputs(&self) -> GroupedBatchLsmCallInputs<'_> {
        let dividends = RaggedCashDividends::try_new(
            &self.group_dividend_offsets,
            &self.group_dividend_times,
            &self.group_dividend_amounts,
        )
        .unwrap();
        GroupedBatchLsmCallInputs::try_new(
            &self.group_offsets,
            &self.group_initial_spots,
            &self.strikes,
            &self.group_maturities,
            &self.group_rates,
            &self.group_volatilities,
            BatchDividendModel::CashDividends {
                schedules: dividends,
                policy: DividendPolicy::Liquidator,
                maturity_boundary: MaturityBoundary::StrictlyBefore,
            },
        )
        .unwrap()
    }

    fn row_inputs(&self) -> BatchLsmCallInputs<'_> {
        let dividends = RaggedCashDividends::try_new(
            &self.row_dividend_offsets,
            &self.row_dividend_times,
            &self.row_dividend_amounts,
        )
        .unwrap();
        BatchLsmCallInputs::try_new(
            &self.row_initial_spots,
            &self.strikes,
            &self.row_maturities,
            &self.row_rates,
            &self.row_volatilities,
            BatchDividendModel::CashDividends {
                schedules: dividends,
                policy: DividendPolicy::Liquidator,
                maturity_boundary: MaturityBoundary::StrictlyBefore,
            },
        )
        .unwrap()
    }
}

fn scalar_model(rate: f64, volatility: f64, cash_amount: f64) -> ScalarModel {
    let times = (0..=STEPS)
        .map(|node| f64::from(node) / f64::from(STEPS))
        .collect::<Vec<_>>();
    let grid = TimeGrid::try_new(times).unwrap();
    let log_discounts = (0..=STEPS)
        .map(|node| {
            if node == 0 {
                0.0
            } else {
                -rate * f64::from(node) / f64::from(STEPS)
            }
        })
        .collect::<Vec<_>>();
    let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
    let dividend_yield_curve =
        CumulativeDividendYieldCurve::try_new(&grid, vec![0.0; grid.len()]).unwrap();
    let dividend = CashDividend::try_new(grid.node_id(75).unwrap(), cash_amount).unwrap();
    let exercise_nodes = (1..usize::try_from(STEPS).unwrap())
        .map(|node| grid.node_id(node).unwrap())
        .collect::<Vec<_>>();
    let expiry = grid.node_id(usize::try_from(STEPS).unwrap()).unwrap();
    let schedule = EventSchedule::try_new(grid, [dividend], exercise_nodes, expiry).unwrap();
    let plan = GbmPathPlan::try_new(
        &schedule,
        &curve,
        &dividend_yield_curve,
        ConstantVolatility::try_new(volatility).unwrap(),
        DividendPolicy::Liquidator,
    )
    .unwrap();
    ScalarModel { plan, curve }
}

fn regression() -> PolynomialRegressionSpec {
    PolynomialRegressionSpec::try_new(PolynomialDegree::try_new(4).unwrap(), 1.0e-12, 1.0e8)
        .unwrap()
}

fn batch_method() -> BatchLsmCallMethod<'static> {
    BatchLsmCallMethod::try_new(
        UniformExerciseMesh::try_new(STEPS, OriginExercise::Excluded).unwrap(),
        TRAINING_PATHS,
        EVALUATION_PATHS,
        regression(),
        ExerciseTieRule::StrictlyGreater,
        InsufficientExerciseSupportPolicy::Fail,
        BatchRandomStreams::new(
            TRAINING_SEED,
            EVALUATION_SEED,
            TRAINING_DOMAIN_START,
            EVALUATION_DOMAIN_START,
        ),
    )
    .unwrap()
}

fn cash_dividend_date_method() -> BatchLsmCallMethod<'static> {
    BatchLsmCallMethod::try_new_cash_dividend_dates(
        CashDividendExerciseMesh::new(),
        TRAINING_PATHS,
        EVALUATION_PATHS,
        regression(),
        ExerciseTieRule::StrictlyGreater,
        InsufficientExerciseSupportPolicy::Fail,
        BatchRandomStreams::new(
            TRAINING_SEED,
            EVALUATION_SEED,
            TRAINING_DOMAIN_START,
            EVALUATION_DOMAIN_START,
        ),
    )
    .unwrap()
}

fn explicit_exercise_method(schedules: RaggedExerciseTimes<'_>) -> BatchLsmCallMethod<'_> {
    BatchLsmCallMethod::try_new_explicit(
        ExplicitExerciseMesh::new(schedules, OriginExercise::Excluded),
        TRAINING_PATHS,
        EVALUATION_PATHS,
        regression(),
        ExerciseTieRule::StrictlyGreater,
        InsufficientExerciseSupportPolicy::Fail,
        BatchRandomStreams::new(
            TRAINING_SEED,
            EVALUATION_SEED,
            TRAINING_DOMAIN_START,
            EVALUATION_DOMAIN_START,
        ),
    )
    .unwrap()
}

fn scalar_repeated(workload: &Workload, executor: &CpuExecutor) -> f64 {
    let mut price_sum = 0.0;
    for row in 0..workload.rows() {
        let model = &workload.scalar_models[row];
        let run = LsmCallEstimator::try_new(
            workload.strikes[row],
            regression(),
            ExerciseTieRule::StrictlyGreater,
            InsufficientExerciseSupportPolicy::Fail,
        )
        .unwrap()
        .price_seeded(
            executor,
            &model.plan,
            &model.curve,
            workload.initial_spots[row],
            TRAINING_PATHS,
            EVALUATION_PATHS,
            RandomStream::new(
                TRAINING_SEED,
                TRAINING_DOMAIN_START + u32::try_from(row).unwrap(),
            ),
            RandomStream::new(
                EVALUATION_SEED,
                EVALUATION_DOMAIN_START + u32::try_from(row).unwrap(),
            ),
        )
        .unwrap();
        price_sum += black_box(run.evaluation().price());
    }
    price_sum
}

fn benchmark_cash_dividend_dates(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("project_grouped_cash_dividend_dates_500_lsm");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    #[cfg(target_os = "macos")]
    let backends = [
        ("rust_std_f64", CpuMathBackend::RustStdF64),
        ("apple_accelerate", CpuMathBackend::AppleAccelerate),
    ];
    #[cfg(not(target_os = "macos"))]
    let backends = [("rust_std_f64", CpuMathBackend::RustStdF64)];
    for groups in [32, 128] {
        let workload = StrikeGroupWorkload::new(groups, 7);
        let explicit_schedules = RaggedExerciseTimes::try_new(
            &workload.group_dividend_offsets,
            &workload.group_dividend_times,
        )
        .unwrap();
        let explicit_method = explicit_exercise_method(explicit_schedules);
        group.throughput(Throughput::Elements(
            u64::try_from(workload.rows()).unwrap(),
        ));
        for threads in [1, 8] {
            for &(backend_name, backend) in &backends {
                let executor = CpuExecutor::try_new_with_math_backend(threads, backend).unwrap();
                group.bench_with_input(
                    BenchmarkId::new(format!("grouped_{backend_name}_{threads}t"), groups),
                    &groups,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(
                                workload
                                    .grouped_inputs()
                                    .price(cash_dividend_date_method(), black_box(&executor))
                                    .unwrap(),
                            );
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("explicit_{backend_name}_{threads}t"), groups),
                    &groups,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(
                                workload
                                    .grouped_inputs()
                                    .price(explicit_method, black_box(&executor))
                                    .unwrap(),
                            );
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("project_150x500_lsm_contracts");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    for rows in [32, 256] {
        let workload = Workload::new(rows);
        group.throughput(Throughput::Elements(u64::try_from(rows).unwrap()));
        for threads in [1, 4] {
            let executor = CpuExecutor::try_new(threads).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("batch_outer_{threads}t"), rows),
                &rows,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            workload
                                .batch_inputs()
                                .price(batch_method(), black_box(&executor))
                                .unwrap(),
                        );
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_repeated_inner_{threads}t"), rows),
                &rows,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(scalar_repeated(black_box(&workload), black_box(&executor)));
                    });
                },
            );
        }
    }
    group.finish();

    let mut group = criterion.benchmark_group("project_grouped_strikes_150x500_lsm");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    #[cfg(target_os = "macos")]
    let backends = [
        ("rust_std_f64", CpuMathBackend::RustStdF64),
        ("apple_accelerate", CpuMathBackend::AppleAccelerate),
    ];
    #[cfg(not(target_os = "macos"))]
    let backends = [("rust_std_f64", CpuMathBackend::RustStdF64)];
    for groups in [32, 128] {
        let workload = StrikeGroupWorkload::new(groups, 7);
        group.throughput(Throughput::Elements(
            u64::try_from(workload.rows()).unwrap(),
        ));
        for threads in [1, 8] {
            for &(backend_name, backend) in &backends {
                let executor = CpuExecutor::try_new_with_math_backend(threads, backend).unwrap();
                group.bench_with_input(
                    BenchmarkId::new(format!("grouped_{backend_name}_{threads}t"), groups),
                    &groups,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(
                                workload
                                    .grouped_inputs()
                                    .price(batch_method(), black_box(&executor))
                                    .unwrap(),
                            );
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("row_{backend_name}_{threads}t"), groups),
                    &groups,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(
                                workload
                                    .row_inputs()
                                    .price(batch_method(), black_box(&executor))
                                    .unwrap(),
                            );
                        });
                    },
                );
            }
        }
    }
    group.finish();

    benchmark_cash_dividend_dates(criterion);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
