use criterion::{Criterion, criterion_group, criterion_main};
use log::debug;
use rs_physics::utils::{AtanLookupTable, fast_atan, fastest_atan, minimax_atan, simd_atan_f32x8};


pub fn bench_atan(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan_methods");
    group.measurement_time(std::time::Duration::from_secs(5));
    group.sample_size(100);
    let lookup_table = AtanLookupTable::new();

    // Test values
    let test_values: Vec<f32> = (-400..400).map(|i| i as f32 / 10.0).collect();

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

    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2_atan", |b| b.iter(|| {
            let mut sum = 0.0_f32;
            let count = test_values.len();
            let mut i = 0;
            while i < count {
                let args: [f32; 8] = test_values[i..i + 8].try_into().unwrap();
                simd_atan_f32x8(args)
                    .iter().for_each(|&atan| sum += atan);
                i += 8;
            }
        }));
    }
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
        for i in 1..101 {
            sum += unsafe{rs_physics::utils::fast_inverse_sqrt(i as f32)};
        }
    }));

    group.bench_function("fast_inverse_sqrt_f64", |b| b.iter(|| {
        let mut sum = 0.0_f64;
        for i in 1..101 {
            sum += unsafe{rs_physics::utils::fast_inverse_sqrt_f64(i as f64)};
        }
    }));
}

criterion_group!(benches, bench_atan, bench_fast_sqrt);
criterion_main!(benches);