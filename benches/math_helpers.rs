use criterion::{Criterion, criterion_group, criterion_main};
use rs_physics::utils::{AtanLookupTable, fast_atan, fastest_atan, minimax_atan};


pub fn bench_atan(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan_methods");
    group.measurement_time(std::time::Duration::from_secs(10));
    const ITERATIONS: usize = 1_000;
    let lookup_table = AtanLookupTable::new();

    // Test values
    let test_values: Vec<f32> = (-100..=100).map(|i| i as f32 / 10.0).collect();

    group.bench_function("truth", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += x.atan();
            }
        }
    }));

    group.bench_function("fast_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += fast_atan(x);
            }
        }
    }));

    group.bench_function("fastest_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += fastest_atan(x);
            }
        }
    }));

    group.bench_function("minimax_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += minimax_atan(x);
            }
        }
    }));

    group.bench_function("lookup_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += lookup_table.atan(x);
            }
        }
    }));
}

criterion_group!(benches, bench_atan);
criterion_main!(benches);