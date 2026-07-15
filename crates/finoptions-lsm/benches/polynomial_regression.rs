use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use finoptions_lsm::{
    PolynomialDegree, PolynomialRegressionSpec, RegressionBenchmark, RegressionPredictionBenchmark,
};
use finoptions_sim::CpuMathBackend;
use nalgebra::{DMatrix, DVector};

fn population(observations: usize) -> (Vec<f64>, Vec<f64>) {
    let observation_count = u32::try_from(observations).unwrap();
    let predictors = (0..observations)
        .map(|index| {
            let index = u32::try_from(index).unwrap();
            let permuted = (u64::from(index) * 104_729) % u64::from(observation_count);
            let unit = f64::from(u32::try_from(permuted).unwrap()) / f64::from(observation_count);
            0.82 + 0.56 * unit + 0.015 * (17.0 * unit).sin()
        })
        .collect::<Vec<_>>();
    let responses = predictors
        .iter()
        .map(|&x| {
            2.0 - 0.7 * x + 0.35 * x.powi(2) - 0.08 * x.powi(3)
                + 0.015 * x.powi(4)
                + 0.002 * (31.0 * x).cos()
        })
        .collect();
    (predictors, responses)
}

fn benchmark(c: &mut Criterion) {
    let spec =
        PolynomialRegressionSpec::try_new(PolynomialDegree::try_new(4).unwrap(), 1.0e-12, 1.0e8)
            .unwrap();
    let mut group = c.benchmark_group("polynomial_regression_degree_4");

    for observations in [500, 65_536] {
        let (predictors, responses) = population(observations);
        let observation_count = u32::try_from(observations).unwrap();
        let mut production = RegressionBenchmark::new(spec);
        group.throughput(Throughput::Elements(u64::from(observation_count)));
        group.bench_with_input(
            BenchmarkId::new("empirical_orthogonal_rust_std_f64", observations),
            &observations,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        production
                            .fit(black_box(&predictors), black_box(&responses))
                            .unwrap(),
                    );
                });
            },
        );

        #[cfg(target_os = "macos")]
        {
            let mut accelerated =
                RegressionBenchmark::new_with_math_backend(spec, CpuMathBackend::AppleAccelerate);
            group.bench_with_input(
                BenchmarkId::new("empirical_orthogonal_apple_accelerate", observations),
                &observations,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(
                            accelerated
                                .fit(black_box(&predictors), black_box(&responses))
                                .unwrap(),
                        );
                    });
                },
            );
        }

        let mean = predictors.iter().sum::<f64>() / f64::from(observation_count);
        let scale = (predictors
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / f64::from(observation_count))
        .sqrt();
        let design = DMatrix::from_fn(observations, 5, |row, column| {
            ((predictors[row] - mean) / scale).powi(i32::try_from(column).unwrap())
        });
        let response = DVector::from_column_slice(&responses);
        group.bench_with_input(
            BenchmarkId::new("independent_svd", observations),
            &observations,
            |bencher, _| {
                bencher.iter(|| {
                    let svd = black_box(design.clone()).svd(true, true);
                    black_box(svd.solve(black_box(&response), 1.0e-12).unwrap());
                });
            },
        );
    }
    group.finish();

    let (predictors, responses) = population(500);
    let prediction = RegressionPredictionBenchmark::fit(&predictors, &responses, spec).unwrap();
    let mut group = c.benchmark_group("polynomial_prediction_degree_4");
    group.throughput(Throughput::Elements(500));
    for (name, predict) in [
        (
            "runtime_degree_recurrence_500",
            RegressionPredictionBenchmark::predict
                as fn(&RegressionPredictionBenchmark, f64) -> Result<f64, _>,
        ),
        (
            "hoisted_degree_four_recurrence_500",
            RegressionPredictionBenchmark::predict_degree_four
                as fn(&RegressionPredictionBenchmark, f64) -> Result<f64, _>,
        ),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                let model = black_box(&prediction);
                let predictors = black_box(&predictors);
                let mut sum = 0.0;
                for &predictor in predictors {
                    sum += predict(model, black_box(predictor)).unwrap();
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
