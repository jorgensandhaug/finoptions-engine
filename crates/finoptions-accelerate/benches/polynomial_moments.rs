use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
#[cfg(target_os = "macos")]
use finoptions_accelerate::polynomial_moments_f64;

fn scalar_moments(predictors: &[f64], responses: &[f64], basis: &[f64]) -> (f64, f64, f64) {
    let mut norm = 0.0;
    let mut projection = 0.0;
    let mut weighted_predictor = 0.0;
    for ((&predictor, &response), &basis_value) in predictors.iter().zip(responses).zip(basis) {
        let squared_basis = basis_value * basis_value;
        norm += squared_basis;
        projection = response.mul_add(basis_value, projection);
        weighted_predictor = predictor.mul_add(squared_basis, weighted_predictor);
    }
    (norm, projection, weighted_predictor)
}

fn benchmark(c: &mut Criterion) {
    let observations = 500_usize;
    let predictors = (0..observations)
        .map(|index| f64::from(u32::try_from(index).unwrap()) / 499.0)
        .collect::<Vec<_>>();
    let responses = predictors
        .iter()
        .map(|predictor| predictor.mul_add(0.75, 2.0))
        .collect::<Vec<_>>();
    let basis = predictors
        .iter()
        .map(|predictor| predictor.mul_add(*predictor, -0.25))
        .collect::<Vec<_>>();
    #[cfg(target_os = "macos")]
    let mut scratch = vec![0.0; observations];
    let mut group = c.benchmark_group("polynomial_moments_f64");
    group.throughput(Throughput::Elements(u64::try_from(observations).unwrap()));
    group.bench_function("rust_scalar", |bencher| {
        bencher.iter(|| {
            black_box(scalar_moments(
                black_box(&predictors),
                black_box(&responses),
                black_box(&basis),
            ));
        });
    });
    #[cfg(target_os = "macos")]
    {
        group.bench_function("apple_vdsp", |bencher| {
            bencher.iter(|| {
                black_box(
                    polynomial_moments_f64(
                        black_box(&predictors),
                        black_box(&responses),
                        black_box(&basis),
                        black_box(&mut scratch),
                    )
                    .unwrap(),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
