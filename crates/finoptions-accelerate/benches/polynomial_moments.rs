use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
#[cfg(target_os = "macos")]
use finoptions_accelerate::{polynomial_moments_f64, polynomial_recurrence_moments_f64};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn vDSP_dotprD(
        first: *const f64,
        first_stride: isize,
        second: *const f64,
        second_stride: isize,
        result: *mut f64,
        count: usize,
    );
    fn vDSP_vsqD(
        input: *const f64,
        input_stride: isize,
        output: *mut f64,
        output_stride: isize,
        count: usize,
    );
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn vdsp_moments_reference(
    predictors: &[f64],
    responses: &[f64],
    basis: &[f64],
    squared_basis: &mut [f64],
) -> (f64, f64, f64) {
    assert_eq!(predictors.len(), responses.len());
    assert_eq!(predictors.len(), basis.len());
    assert_eq!(predictors.len(), squared_basis.len());
    if basis.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut norm = 0.0;
    let mut projection = 0.0;
    let mut weighted_predictor = 0.0;
    // SAFETY: the assertions prove all contiguous slices contain `basis.len()` elements. Mutable
    // scratch is disjoint from the immutable operands, every stride is one, and scalar outputs live
    // through their respective calls. This benchmark-only reference is the accepted pre-candidate
    // production implementation.
    unsafe {
        vDSP_vsqD(
            basis.as_ptr(),
            1,
            squared_basis.as_mut_ptr(),
            1,
            basis.len(),
        );
        vDSP_dotprD(
            basis.as_ptr(),
            1,
            basis.as_ptr(),
            1,
            &raw mut norm,
            basis.len(),
        );
        vDSP_dotprD(
            responses.as_ptr(),
            1,
            basis.as_ptr(),
            1,
            &raw mut projection,
            basis.len(),
        );
        vDSP_dotprD(
            predictors.as_ptr(),
            1,
            squared_basis.as_ptr(),
            1,
            &raw mut weighted_predictor,
            basis.len(),
        );
    }
    (norm, projection, weighted_predictor)
}

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

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn separate_recurrence_moments_reference(
    predictors: &[f64],
    responses: &[f64],
    previous: &[f64],
    current: &[f64],
    next: &mut [f64],
    squared: &mut [f64],
    alpha: f64,
    beta: f64,
) -> (f64, f64, f64) {
    for (((&predictor, &previous), &current), next) in
        predictors.iter().zip(previous).zip(current).zip(&mut *next)
    {
        *next = (predictor - alpha).mul_add(current, -beta * previous);
    }
    polynomial_moments_f64(predictors, responses, next, squared).unwrap()
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
        group.bench_function("apple_production", |bencher| {
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
        group.bench_function("apple_vdsp_reference", |bencher| {
            bencher.iter(|| {
                black_box(vdsp_moments_reference(
                    black_box(&predictors),
                    black_box(&responses),
                    black_box(&basis),
                    black_box(&mut scratch),
                ));
            });
        });
    }
    group.finish();

    #[cfg(target_os = "macos")]
    {
        let previous = predictors
            .iter()
            .map(|predictor| predictor.mul_add(-0.2, 0.3))
            .collect::<Vec<_>>();
        let current = predictors
            .iter()
            .map(|predictor| predictor.mul_add(*predictor, 0.7))
            .collect::<Vec<_>>();
        let mut next = vec![0.0; observations];
        let mut squared = vec![0.0; observations];
        let alpha = 0.17;
        let beta = 0.23;
        let mut group = c.benchmark_group("polynomial_recurrence_moments_f64");
        group.throughput(Throughput::Elements(u64::try_from(observations).unwrap()));
        group.bench_function("separate_recurrence_then_moments", |bencher| {
            bencher.iter(|| {
                black_box(separate_recurrence_moments_reference(
                    black_box(&predictors),
                    black_box(&responses),
                    black_box(&previous),
                    black_box(&current),
                    black_box(&mut next),
                    black_box(&mut squared),
                    alpha,
                    beta,
                ));
            });
        });
        group.bench_function("fused_apple_production", |bencher| {
            bencher.iter(|| {
                black_box(
                    polynomial_recurrence_moments_f64(
                        black_box(&predictors),
                        black_box(&responses),
                        black_box(&previous),
                        black_box(&current),
                        black_box(&mut next),
                        alpha,
                        beta,
                    )
                    .unwrap(),
                );
            });
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
