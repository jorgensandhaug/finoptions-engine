use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use finoptions_batch::RaggedCashDividends;
use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve, TimeGrid,
};
use finoptions_lsm::{
    BatchDividendModel, BatchLsmCallInputs, BatchLsmCallMethod, BatchRandomStreams,
    ExerciseTieRule, LsmCallEstimator, OriginExercise, PolynomialDegree, PolynomialRegressionSpec,
    UniformExerciseMesh,
};
use finoptions_sim::{CpuExecutor, GbmPathPlan, RandomStream};

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

fn batch_method() -> BatchLsmCallMethod {
    BatchLsmCallMethod::try_new(
        UniformExerciseMesh::try_new(STEPS, OriginExercise::Excluded).unwrap(),
        TRAINING_PATHS,
        EVALUATION_PATHS,
        regression(),
        ExerciseTieRule::StrictlyGreater,
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
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
