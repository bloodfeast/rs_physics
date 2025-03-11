use criterion::{Criterion, criterion_group, criterion_main};
use rs_physics::utils::{AtanLookupTable, fast_atan, fastest_atan, minimax_atan};


pub fn bench_atan(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan_methods");
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);
    let lookup_table = AtanLookupTable::new();

    // Test values
    let test_values: Vec<f32> = (-100..=100).map(|i| i as f32 / 10.0).collect();

    group.bench_function("truth", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for &x in &test_values {
            sum += x.atan();
        }
    }));

    group.bench_function("fast_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for &x in &test_values {
            sum += fast_atan(x);
        }
    }));

    group.bench_function("fastest_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for &x in &test_values {
            sum += fastest_atan(x);
        }
    }));

    group.bench_function("minimax_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for &x in &test_values {
            sum += minimax_atan(x);
        }
    }));

    group.bench_function("lookup_atan", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for &x in &test_values {
            sum += lookup_table.atan(x);
        }
    }));
}

pub fn bench_fast_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_sqrt");
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);

    group.bench_function("truth", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for i in 0..100 {
            sum += (i as f32).sqrt();
        }
    }));

    group.bench_function("fast_sqrt", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for i in 0..100 {
            sum += rs_physics::utils::fast_sqrt(i as f32);
        }
    }));

    group.bench_function("fast_sqrt_f64_truth" , |b| b.iter(|| {
        let mut sum = 0.0_f64;
        for i in 0..100 {
            sum += (i as f64).sqrt();
        }
    }));

    group.bench_function("fast_sqrt_f64", |b| b.iter(|| {
        let mut sum = 0.0_f64;
        for i in 0..100 {
            sum += rs_physics::utils::fast_sqrt_f64(i as f64);
        }
    }));

    group.bench_function("inverse_sqrt_truth", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for i in 0..100 {
            sum += 1.0 / (i as f32).sqrt();
        }
    }));

    group.bench_function("fast_inverse_sqrt", |b| b.iter(|| {
        let mut sum = 0.0_f32;
        for i in 0..100 {
            sum += rs_physics::utils::fast_inverse_sqrt(i as f32);
        }
    }));
}

criterion_group!(benches, bench_atan, bench_fast_sqrt);
criterion_main!(benches);