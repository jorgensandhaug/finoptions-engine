use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use finoptions_core::{
    ConstantVolatility, CumulativeDividendYieldCurve, DividendPolicy, EventSchedule,
    LogDiscountCurve, TimeGrid,
};
use finoptions_sim::{CpuExecutor, GbmPathPlan, RandomStream};

const INTERVALS: usize = 32;
const PATH_COUNTS: [usize; 2] = [1_000, 100_000];
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

fn plan() -> GbmPathPlan {
    let denominator = f64::from(u32::try_from(INTERVALS).unwrap());
    let times = (0..=INTERVALS)
        .map(|index| f64::from(u32::try_from(index).unwrap()) / denominator)
        .collect::<Vec<_>>();
    let grid = TimeGrid::try_new(times).unwrap();
    let log_discounts = grid
        .as_slice()
        .iter()
        .map(|&time| if time == 0.0 { 0.0 } else { -0.04 * time })
        .collect::<Vec<_>>();
    let curve = LogDiscountCurve::try_new(&grid, log_discounts).unwrap();
    let dividend_yield_curve =
        CumulativeDividendYieldCurve::try_new(&grid, vec![0.0; grid.len()]).unwrap();
    let exercise_nodes = (4..INTERVALS)
        .step_by(4)
        .map(|index| grid.node_id(index).unwrap())
        .collect::<Vec<_>>();
    let expiry = grid.node_id(INTERVALS).unwrap();
    let schedule = EventSchedule::try_new(grid, [], exercise_nodes, expiry).unwrap();
    GbmPathPlan::try_new(
        &schedule,
        &curve,
        &dividend_yield_curve,
        ConstantVolatility::try_new(0.25).unwrap(),
        DividendPolicy::Liquidator,
    )
    .unwrap()
}

fn varied_normals(values: usize) -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..values)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let high = u32::try_from(state >> 32).unwrap();
            4.0 * (f64::from(high) / f64::from(u32::MAX) - 0.5)
        })
        .collect()
}

fn benchmark_gbm_paths(criterion: &mut Criterion) {
    let plan = plan();
    let mut external = criterion.benchmark_group("gbm_paths_external_f64");
    for paths in PATH_COUNTS {
        let values = INTERVALS.checked_mul(paths).unwrap();
        let normals = varied_normals(values);
        external.throughput(Throughput::Elements(u64::try_from(values).unwrap()));
        for threads in THREAD_COUNTS {
            let executor = CpuExecutor::try_new(threads).unwrap();
            external.bench_function(
                BenchmarkId::new(format!("{threads}_threads"), paths),
                |bencher| {
                    bencher.iter(|| {
                        plan.simulate_with(&executor, black_box(100.0), paths, black_box(&normals))
                            .unwrap()
                    });
                },
            );
        }
    }
    external.finish();

    let stream = RandomStream::new(0x01d6_328f_7a4c_915b, 0x1729_0042);
    let mut seeded = criterion.benchmark_group("gbm_paths_seeded_f64");
    for paths in PATH_COUNTS {
        let transitions = INTERVALS.checked_mul(paths).unwrap();
        seeded.throughput(Throughput::Elements(u64::try_from(transitions).unwrap()));
        for threads in THREAD_COUNTS {
            let executor = CpuExecutor::try_new(threads).unwrap();
            seeded.bench_function(
                BenchmarkId::new(format!("{threads}_threads"), paths),
                |bencher| {
                    bencher.iter(|| {
                        plan.simulate_seeded_with(
                            &executor,
                            black_box(100.0),
                            paths,
                            black_box(stream),
                        )
                        .unwrap()
                    });
                },
            );
        }
    }
    seeded.finish();
}

criterion_group!(benches, benchmark_gbm_paths);
criterion_main!(benches);
