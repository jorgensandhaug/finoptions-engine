use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use finoptions_core::{
    CashDividend, ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve, TimeGrid,
};
use finoptions_lsm::{
    ExerciseTieRule, InsufficientExerciseSupportPolicy, LsmCallEstimator, PolynomialDegree,
    PolynomialRegressionSpec,
};
use finoptions_sim::{CpuExecutor, GbmPathPlan, RandomStream};

struct Workload {
    name: &'static str,
    initial_spot: f64,
    paths: usize,
    plan: GbmPathPlan,
    curve: LogDiscountCurve,
}

fn workload(
    name: &'static str,
    initial_spot: f64,
    steps: u32,
    paths: usize,
    dividend_node: u32,
    exercise_nodes: Vec<u32>,
) -> Workload {
    let times = (0..=steps)
        .map(|node| f64::from(node) / f64::from(steps))
        .collect::<Vec<_>>();
    let grid = TimeGrid::try_new(times).unwrap();
    let log_discounts = (0..=steps)
        .map(|node| {
            if node == 0 {
                0.0
            } else {
                -0.06 * f64::from(node) / f64::from(steps)
            }
        })
        .collect::<Vec<_>>();
    let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
    let dividend_yield_curve =
        CumulativeDividendYieldCurve::try_new(&grid, vec![0.0; grid.len()]).unwrap();
    let dividend = CashDividend::try_new(
        grid.node_id(usize::try_from(dividend_node).unwrap())
            .unwrap(),
        7.0,
    )
    .unwrap();
    let exercise_nodes = exercise_nodes
        .into_iter()
        .map(|node| grid.node_id(usize::try_from(node).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let expiry = grid.node_id(usize::try_from(steps).unwrap()).unwrap();
    let schedule = EventSchedule::try_new(grid, [dividend], exercise_nodes, expiry).unwrap();
    let plan = GbmPathPlan::try_new(
        &schedule,
        &curve,
        &dividend_yield_curve,
        ConstantVolatility::try_new(0.30).unwrap(),
        DividendPolicy::Liquidator,
    )
    .unwrap();
    Workload {
        name,
        initial_spot,
        paths,
        plan,
        curve,
    }
}

fn estimator() -> LsmCallEstimator {
    let regression =
        PolynomialRegressionSpec::try_new(PolynomialDegree::try_new(4).unwrap(), 1.0e-12, 1.0e8)
            .unwrap();
    LsmCallEstimator::try_new(
        100.0,
        regression,
        ExerciseTieRule::StrictlyGreater,
        InsufficientExerciseSupportPolicy::Fail,
    )
    .unwrap()
}

fn benchmark(c: &mut Criterion) {
    let workloads = [
        workload("project_150x500", 150.0, 150, 500, 75, (0..150).collect()),
        workload("reference_32x65536", 100.0, 32, 65_536, 16, vec![16]),
    ];
    let mut group = c.benchmark_group("lsm_call_fit_and_independent_evaluation");
    for workload in workloads {
        let transitions = u64::try_from(workload.plan.intervals())
            .unwrap()
            .checked_mul(u64::try_from(workload.paths).unwrap())
            .and_then(|value| value.checked_mul(2))
            .unwrap();
        group.throughput(Throughput::Elements(transitions));
        for threads in [1, 4] {
            let executor = CpuExecutor::try_new(threads).unwrap();
            group.bench_with_input(
                BenchmarkId::new(workload.name, threads),
                &threads,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            estimator()
                                .price_seeded(
                                    &executor,
                                    black_box(&workload.plan),
                                    black_box(&workload.curve),
                                    workload.initial_spot,
                                    workload.paths,
                                    workload.paths,
                                    RandomStream::new(10_000, 1),
                                    RandomStream::new(20_000, 2),
                                )
                                .unwrap(),
                        );
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
