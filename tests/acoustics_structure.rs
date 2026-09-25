//! **L8. No blocking, structurally.** The GPU acoustics never poll the device, never pull
//! in a blocking executor, and take the device only when they are built. There is no timing
//! assertion: a pass that blocks can be fast on a quiet machine, so the test reads the code.

const SOURCES: &[(&str, &str)] = &[
    ("mod.rs", include_str!("../src/gpu/acoustics/mod.rs")),
    ("pack.rs", include_str!("../src/gpu/acoustics/pack.rs")),
    (
        "readback.rs",
        include_str!("../src/gpu/acoustics/readback.rs"),
    ),
    (
        "records.rs",
        include_str!("../src/gpu/acoustics/records.rs"),
    ),
    ("shader.rs", include_str!("../src/gpu/acoustics/shader.rs")),
    ("oracle.rs", include_str!("../src/gpu/acoustics/oracle.rs")),
    (
        "query.wgsl",
        include_str!("../src/gpu/acoustics/query.wgsl"),
    ),
    (
        "build.wgsl",
        include_str!("../src/gpu/acoustics/build.wgsl"),
    ),
];

/// Lines of code, doc and line comments dropped: the docs talk about polling on purpose.
fn code(src: &str) -> impl Iterator<Item = (usize, &str)> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
}

#[test]
fn l8_the_acoustics_never_poll_and_take_the_device_only_to_build() {
    for (name, src) in SOURCES {
        for (n, line) in code(src) {
            let l = line.to_ascii_lowercase();
            assert!(
                !l.contains(".poll("),
                "{name}:{}: polls the device: {line}",
                n + 1
            );
            assert!(
                !l.contains("poll_all"),
                "{name}:{}: polls the instance: {line}",
                n + 1
            );
            assert!(
                !l.contains("pollster"),
                "{name}:{}: blocks on a future: {line}",
                n + 1
            );
            assert!(
                !l.contains("block_on"),
                "{name}:{}: blocks on a future: {line}",
                n + 1
            );
            assert!(
                !l.contains("on_submitted_work_done"),
                "{name}:{}: waits on the queue: {line}",
                n + 1
            );
        }
    }
    // Every function signature that takes a device: public ones must be `new`; the private
    // ones are the constructors' helpers.
    let mut public_takers = Vec::new();
    for (name, src) in SOURCES.iter().filter(|(n, _)| n.ends_with(".rs")) {
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            if !(t.starts_with("pub fn ")
                || t.starts_with("fn ")
                || t.starts_with("pub(crate) fn "))
            {
                continue;
            }
            // The signature runs to the line that opens the body.
            let mut sig = String::new();
            for l in &lines[i..] {
                sig.push_str(l);
                if l.contains('{') || l.trim_end().ends_with(';') {
                    break;
                }
            }
            let fn_name = t
                .split("fn ")
                .nth(1)
                .unwrap()
                .split(['(', '<'])
                .next()
                .unwrap()
                .to_string();
            if sig.contains("wgpu::Device") && !sig.contains("-> wgpu::Device") {
                if t.starts_with("pub fn ") {
                    public_takers.push(format!("{name}: {fn_name}"));
                } else {
                    assert!(
                        ["groups", "pipeline", "new"].contains(&fn_name.as_str()),
                        "{name}: private fn {fn_name} takes the device and is not a constructor helper"
                    );
                }
            }
        }
    }
    for f in &public_takers {
        assert!(
            f.ends_with(": new"),
            "a public function other than new takes the device: {f}"
        );
    }
    // Two constructors: the pass, and the law probe the oracles use.
    assert_eq!(public_takers.len(), 2, "{public_takers:?}");
}
