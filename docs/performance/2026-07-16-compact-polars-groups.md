# Compact Polars grouped boundary

Date: 2026-07-16  
Machine: Apple arm64  
Python: 3.14.6  
Polars: 1.42.1  
NumPy: 2.5.1

## Question

What does the explicit two-axis Polars boundary cost when one million strike rows share 100,000
independently simulated market states?

This benchmark isolates lazy projection, explicit collection, and validation/native preparation.
The synthetic Polars frames are already materialized before timing, and no pricing estimator runs.
It therefore does not measure Parquet scanning, group construction, simulation, or LSM fitting.

## Contract

- 100,000 state groups.
- 1,000,000 strike rows.
- Exact `UInt32` group offsets supplied by the caller.
- Six iterations; the first is reported separately and the remaining five use the median.
- `BufferPolicy.REQUIRE_ZERO_COPY`.
- Two Polars threads and two vector-library threads at most.
- Scalar and offset buffer sharing is asserted on every iteration.
- Cash groups contain two chronological events each.

Commands:

```shell
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=2 nice -n 10 \
  .venv/bin/python benchmarks/polars_grouped_boundary.py \
  --groups 100000 --strike-rows 1000000 \
  --dividends continuous_yield --iterations 6

POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=2 nice -n 10 \
  .venv/bin/python benchmarks/polars_grouped_boundary.py \
  --groups 100000 --strike-rows 1000000 \
  --dividends cash --iterations 6
```

## Results

| Dividend input | Plan median | Collect median | Prepare median | Native request | Incremental peak RSS | State rows avoided |
|---|---:|---:|---:|---:|---:|---:|
| Continuous yield | 0.081 ms | 0.105 ms | 0.138 ms | 12.40 MB | 4.18 MB | 900,000 |
| Two cash events | 0.113 ms | 0.211 ms | 2.911 ms | 15.60 MB | 4.95 MB | 900,000 |

The native request sizes count the exact borrowed arrays visible to the estimator. Compared with
repeating the same state on every strike row, the compact representation reduces that array surface
from 48.00 MB to 12.40 MB for continuous yield (74.2%) and from 80.00 MB to 15.60 MB for two cash
events (80.5%). These are layout reductions, not whole-process memory claims.

The steady preparation medians correspond to 724.2 million state groups per second for continuous
yield and 34.4 million for cash. The cash cost is dominated by validating paired list lengths,
exploding the two event children, and constructing the one explicit `Int64` event-offset axis. No
financial value is cast, sorted, regrouped, inferred, or copied implicitly.

## Retained reports

Continuous yield:

```json
{"cold_collect_seconds":0.0004996250208932906,"cold_plan_seconds":0.0004364999767858535,"cold_preparation_seconds":0.0002976669929921627,"collect_seconds":0.00010529198334552348,"collect_strike_rows_per_second":9497399215.270033,"dividend_representation":"continuous_yield","events":0,"format":"finoptions-polars-grouped-boundary-benchmark-v1","groups":100000,"incremental_peak_rss_bytes":4177920,"iterations":6,"native_request_bytes":12400004,"numpy_version":"2.5.1","peak_max_rss_bytes":108003328,"plan_seconds":8.116700337268412e-05,"platform_machine":"arm64","polars_version":"1.42.1","preparation_groups_per_second":724202072.6332154,"preparation_seconds":0.00013808300718665123,"preparation_strike_rows_per_second":7242020726.332154,"python_version":"3.14.6","scalar_and_offset_zero_copy_verified":true,"state_rows_avoided":900000,"steady_collect_seconds_minimum":9.420799324288964e-05,"steady_plan_seconds_minimum":6.129097891971469e-05,"steady_preparation_seconds_minimum":0.0001194580108858645,"strike_rows":1000000}
```

Cash:

```json
{"cold_collect_seconds":0.0002771250146906823,"cold_plan_seconds":0.00027020799461752176,"cold_preparation_seconds":0.003219332982553169,"collect_seconds":0.00021116601419635117,"collect_strike_rows_per_second":4735610528.075589,"dividend_representation":"cash","events":200000,"format":"finoptions-polars-grouped-boundary-benchmark-v1","groups":100000,"incremental_peak_rss_bytes":4947968,"iterations":6,"native_request_bytes":15600012,"numpy_version":"2.5.1","peak_max_rss_bytes":117882880,"plan_seconds":0.00011337499017827213,"platform_machine":"arm64","polars_version":"1.42.1","preparation_groups_per_second":34355406.83830582,"preparation_seconds":0.002910749986767769,"preparation_strike_rows_per_second":343554068.38305825,"python_version":"3.14.6","scalar_and_offset_zero_copy_verified":true,"state_rows_avoided":900000,"steady_collect_seconds_minimum":0.00012091698590666056,"steady_plan_seconds_minimum":7.450001430697739e-05,"steady_preparation_seconds_minimum":0.002508875011699274,"strike_rows":1000000}
```
