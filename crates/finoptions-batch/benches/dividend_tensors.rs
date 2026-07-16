use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use finoptions_batch::{
    CashAmountScale, DividendBinSpec, MaturityBoundary, PaddedSequenceSpec, RaggedCashDividends,
    TimeBinClosure, bin_dividends_f32, pack_padded_dividends_f32,
};
use finoptions_runtime::CpuExecutor;

const ROWS: usize = 1_000_000;
const THREAD_COUNTS: [usize; 3] = [1, 4, 10];

fn synthetic_schedules(rows: usize) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let mut offsets = Vec::with_capacity(rows + 1);
    let mut times = Vec::with_capacity(rows * 2);
    let mut amounts = Vec::with_capacity(rows * 2);
    offsets.push(0);
    for row in 0..rows {
        let events = row % 5;
        for event in 0..events {
            let event = u32::try_from(event).unwrap();
            let events = u32::try_from(events).unwrap();
            times.push(f64::from(event + 1) / f64::from(events + 1));
            amounts.push(0.25 + (f64::from(event) * 0.125));
        }
        offsets.push(i64::try_from(times.len()).unwrap());
    }
    (offsets, times, amounts)
}

fn benchmark_dividend_tensors(criterion: &mut Criterion) {
    let (offsets, times, amounts) = synthetic_schedules(ROWS);
    let schedules = RaggedCashDividends::try_new(&offsets, &times, &amounts).unwrap();
    let maturities = vec![1.0; ROWS];
    let spots = vec![100.0; ROWS];
    let sequence = PaddedSequenceSpec::try_new(4, MaturityBoundary::AtOrBefore).unwrap();
    let bins = DividendBinSpec::try_new(
        12,
        1.0,
        TimeBinClosure::LeftOpenRightClosed,
        MaturityBoundary::AtOrBefore,
    )
    .unwrap();

    let mut group = criterion.benchmark_group("ragged_dividend_tensors_f32");
    group.throughput(Throughput::Elements(u64::try_from(ROWS).unwrap()));
    for threads in THREAD_COUNTS {
        let executor = CpuExecutor::try_new(threads).unwrap();
        group.bench_with_input(
            BenchmarkId::new("padded", threads),
            &threads,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        pack_padded_dividends_f32(
                            schedules,
                            black_box(&maturities),
                            CashAmountScale::DivideBySpot(black_box(&spots)),
                            sequence,
                            &executor,
                        )
                        .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("binned", threads),
            &threads,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        bin_dividends_f32(
                            schedules,
                            black_box(&maturities),
                            CashAmountScale::DivideBySpot(black_box(&spots)),
                            bins,
                            &executor,
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_dividend_tensors);
criterion_main!(benches);
