//! Calibrix core math in Rust — MIT.
//!
//! Clean-room reimplementation of the hot paths (no external kernels):
//!   * kernel gain materialization (bit-exact with Python within f64 noise)
//!   * kernel spec-string parsing (the ComfyUI node grammar)
//!   * Pareto fronts + knee selection (NSGA-II math)
//!   * differential evolution (Storn & Price 1997)
//!
//! Built as a cdylib for FFI/accel use and an rlib for tests/benchmarks.

use serde::{Deserialize, Serialize};

/// Parameters of the modulation kernel for one component. Field order and
/// semantics are fixed by the Python `KernelParams` — the two sides must
/// stay in lockstep (parity is test-enforced).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KernelParams {
    pub weight: f64,
    pub position: f64, // normalized 0..1
    pub focus: f64,    // normalized layer distance
    pub floor: f64,
    pub ripple: f64,
    pub phase: f64,
}

/// Python-parity ripple: base + ripple*cos(2*pi*l/n + 2*pi*phase)
///
/// NOTE: this intentionally replicates a *quirk* of the Python
/// implementation (phase inside the cosine, no per-binade pulse). When the
/// Python side ships the correct per-binade cosine pulse, the Rust side
/// must be updated in the same commit — parity is test-enforced.
fn python_parity_ripple(l: usize, n: usize, ripple: f64, phase: f64) -> f64 {
    let t = 2.0 * std::f64::consts::PI * (l as f64) / (n.max(1) as f64)
        + 2.0 * std::f64::consts::PI * phase;
    ripple * t.cos()
}

impl KernelParams {
    /// Materialize per-site gains for `n_sites`. Bit-exact with Python
    /// `ModulationSpec.gains()` within f64 noise.
    pub fn gains(&self, n_sites: usize) -> Vec<f64> {
        let last = std::cmp::max(n_sites as i64 - 1, 1) as f64;
        let pos = self.position * last;
        let f = self.focus * last;
        let n = n_sites.max(1) as f64;
        (0..n_sites)
            .map(|l| {
                let dist = (l as f64 - pos).abs();
                let decay = if f > 0.0 { (dist / f).min(1.0) } else { 1.0 };
                let mut g = self.weight - (self.weight - self.floor) * decay;
                if self.ripple != 0.0 {
                    g += python_parity_ripple(l, n_sites, self.ripple, self.phase);
                }
                g
            })
            .collect()
    }

    /// Validate the same invariants the Python side enforces.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("weight", self.weight),
            ("position", self.position),
            ("focus", self.focus),
            ("floor", self.floor),
            ("ripple", self.ripple),
            ("phase", self.phase),
        ] {
            if !v.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.focus <= 0.0 {
            return Err("focus must be > 0".into());
        }
        if self.weight < 0.0 || self.floor < 0.0 {
            return Err("weight and floor must be >= 0".into());
        }
        if self.position < 0.0 || self.position > 1.0 {
            return Err("position must be in [0, 1]".into());
        }
        Ok(())
    }

    /// Parse one channel of the ComfyUI node grammar:
    /// `comp:weight@position:floor:focus[:ripple:phase]`
    pub fn parse_channel(part: &str) -> Result<(String, KernelParams), String> {
        let part = part.trim();
        let (head, rest) = part
            .split_once(':')
            .ok_or_else(|| format!("bad segment (missing ':'): {part:?}"))?;
        let comp = head.trim().to_string();
        if comp.is_empty() {
            return Err(format!("empty component in {part:?}"));
        }
        let (w_str, tail) = rest
            .split_once('@')
            .ok_or_else(|| format!("bad segment (missing '@'): {part:?}"))?;
        let weight: f64 = w_str
            .trim()
            .parse()
            .map_err(|_| format!("bad weight in {part:?}"))?;
        let nums: Vec<f64> = tail
            .split(':')
            .map(|t| t.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| format!("bad numbers in {part:?}"))?;
        let get = |i: usize, default: f64| nums.get(i).copied().unwrap_or(default);
        let p = KernelParams {
            weight,
            position: get(0, 0.5),
            floor: get(1, 1.0),
            focus: get(2, 1e6),
            ripple: get(3, 0.0),
            phase: get(4, 0.0),
        };
        p.validate()?;
        Ok((comp, p))
    }

    /// Parse a full spec string `attn:...|mlp:...` into channels.
    pub fn parse_spec(spec: &str) -> Result<Vec<(String, KernelParams)>, String> {
        spec.split('|')
            .filter(|s| !s.trim().is_empty())
            .map(KernelParams::parse_channel)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Multi-objective: Pareto fronts (Deb 2002 non-dominated sorting)
// ---------------------------------------------------------------------------

fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

/// Non-dominated sorting, minimization. Returns fronts as index lists.
pub fn pareto_fronts(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    let mut dom_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&objectives[i], &objectives[j]) {
                dominated_by[i].push(j);
                dom_count[j] += 1;
            } else if dominates(&objectives[j], &objectives[i]) {
                dominated_by[j].push(i);
                dom_count[i] += 1;
            }
        }
    }
    let mut fronts = Vec::new();
    let mut current: Vec<usize> = (0..n).filter(|&i| dom_count[i] == 0).collect();
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominated_by[i] {
                dom_count[j] -= 1;
                if dom_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        fronts.push(current);
        current = next;
    }
    fronts
}

/// Knee point of a front: min-max achievement scalarization on the
/// min-max normalized objectives (same math as the Python `knee_point`).
pub fn knee_point(front: &[usize], objectives: &[Vec<f64>]) -> Option<usize> {
    if front.is_empty() {
        return None;
    }
    let m = objectives.first().map(|r| r.len()).unwrap_or(0);
    if m == 0 {
        return None;
    }
    let pts: Vec<Vec<f64>> = front.iter().map(|&i| objectives[i].clone()).collect();
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for p in &pts {
        for k in 0..m {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let span = |k: usize| if hi[k] - lo[k] < 1e-12 { 1.0 } else { hi[k] - lo[k] };
    let mut best_i = 0usize;
    let mut best_key = f64::INFINITY;
    for (i, p) in pts.iter().enumerate() {
        let scaled: Vec<f64> = (0..m).map(|k| (p[k] - lo[k]) / span(k)).collect();
        // achievement = max normalized objective, ties broken by the sum
        // (identical to the Python knee_point key)
        let key = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1000.0
            + scaled.iter().sum::<f64>() * 1e-3;
        if key < best_key {
            best_key = key;
            best_i = i;
        }
    }
    Some(front[best_i])
}

// ---------------------------------------------------------------------------
// Differential evolution (DE/rand/1/bin), f64, seeded PRNG (xorshift64*)
// ---------------------------------------------------------------------------

pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Minimizes `f`. Same contract as the Python DifferentialEvolution.
pub struct DifferentialEvolution<'a> {
    pub pop: Vec<Vec<f64>>,
    pub fit: Vec<f64>,
    func: &'a dyn Fn(&[f64]) -> f64,
    dim: usize,
    lo: f64,
    hi: f64,
    f: f64,
    cr: f64,
    rng: XorShift64,
}

impl<'a> DifferentialEvolution<'a> {
    pub fn new(func: &'a dyn Fn(&[f64]) -> f64, dim: usize, lo: f64, hi: f64,
               popsize: usize, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let pop: Vec<Vec<f64>> = (0..popsize)
            .map(|_| (0..dim).map(|_| rng.uniform(lo, hi)).collect())
            .collect();
        let fit = pop.iter().map(|x| func(x)).collect();
        Self { pop, fit, func, dim, lo, hi, f: 0.8, cr: 0.9, rng }
    }

    pub fn step(&mut self) -> (f64, Vec<f64>) {
        let popsize = self.pop.len();
        for i in 0..popsize {
            // sample 3 distinct indices != i (rejection sampling, popsize >= 4)
            let mut pick = |rng: &mut XorShift64| (rng.next_u64() % popsize as u64) as usize;
            let mut a = pick(&mut self.rng);
            let mut b = pick(&mut self.rng);
            let mut c = pick(&mut self.rng);
            while a == i { a = pick(&mut self.rng); }
            while b == i || b == a { b = pick(&mut self.rng); }
            while c == i || c == a || c == b { c = pick(&mut self.rng); }
            let mut trial = self.pop[i].clone();
            let mut cross = self.rng.next_f64() < self.cr;
            let jrand = (self.rng.next_u64() % self.dim as u64) as usize;
            for j in 0..self.dim {
                if j == jrand || cross {
                    let mut v = self.pop[a][j] + self.f * (self.pop[b][j] - self.pop[c][j]);
                    if !v.is_finite() {
                        v = self.lo;
                    }
                    trial[j] = v.clamp(self.lo, self.hi);
                }
                cross = self.rng.next_f64() < self.cr;
            }
            let tf = (self.func)(&trial);
            if tf <= self.fit[i] {
                self.pop[i] = trial;
                self.fit[i] = tf;
            }
        }
        let bi = min_index(&self.fit);
        (self.fit[bi], self.pop[bi].clone())
    }
}

fn min_index(v: &[f64]) -> usize {
    let mut bi = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[bi] {
            bi = i;
        }
    }
    bi
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gains_match_reference_shape() {
        let k = KernelParams { weight: 1.5, position: 0.5, focus: 0.3, floor: 0.4, ripple: 0.0, phase: 0.0 };
        let g = k.gains(21);
        assert_eq!(g.len(), 21);
        let peak = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((peak - 1.5).abs() < 1e-12);
        assert!((g[10] - 1.5).abs() < 1e-12); // odd n => exact center peak
    }

    #[test]
    fn parses_comfyui_spec() {
        let chans = KernelParams::parse_spec("attn:1.2@0.5:0.9:0.3|mlp:0.8@0.25:1.0:1e6").unwrap();
        assert_eq!(chans.len(), 2);
        assert_eq!(chans[0].0, "attn");
        assert!((chans[0].1.weight - 1.2).abs() < 1e-12);
        assert!((chans[1].1.floor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pareto_fronts_are_layered() {
        let objs = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.5, 0.5], vec![2.0, 0.0]];
        let fronts = pareto_fronts(&objs);
        assert_eq!(fronts[0], vec![0]);
        assert!(fronts[1].contains(&1) || fronts[1].contains(&2) || fronts[1].contains(&3));
    }

    #[test]
    fn de_minimizes_sphere() {
        let f = |x: &[f64]| x.iter().map(|v| v * v).sum();
        let mut de = DifferentialEvolution::new(&f, 4, -2.0, 2.0, 20, 7);
        for _ in 0..30 {
            de.step();
        }
        assert!(de.fit.iter().cloned().fold(f64::INFINITY, f64::min) < 0.1);
    }
}

// ---------------------------------------------------------------------------
// C ABI for the Python accel bridge (ctypes, no pyo3 dependency)
// ---------------------------------------------------------------------------

/// Write gains into caller-provided buffer. Returns 0 on success.
#[no_mangle]
pub extern "C" fn calibrix_gains(
    weight: f64, position: f64, focus: f64, floor: f64, ripple: f64, phase: f64,
    n_sites: usize, out: *mut f64,
) -> i32 {
    let p = KernelParams { weight, position, focus, floor, ripple, phase };
    if p.validate().is_err() || out.is_null() {
        return -1;
    }
    let g = p.gains(n_sites);
    unsafe {
        for (i, v) in g.iter().enumerate() {
            *out.add(i) = *v;
        }
    }
    0
}

/// Parse a spec string; on success writes the flat params (6 per channel)
/// into `out` and the channel count into `n_channels`. Returns 0 on success.
#[no_mangle]
pub extern "C" fn calibrix_parse_spec(
    spec: *const std::os::raw::c_char,
    out: *mut f64,
    out_cap: usize,
    n_channels: *mut usize,
) -> i32 {
    if spec.is_null() || out.is_null() || n_channels.is_null() {
        return -1;
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(spec) };
    let s = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let chans = match KernelParams::parse_spec(s) {
        Ok(c) => c,
        Err(_) => return -3,
    };
    if chans.len() * 6 > out_cap {
        return -4;
    }
    unsafe {
        let mut i = 0usize;
        for (_, p) in &chans {
            for v in [p.weight, p.position, p.focus, p.floor, p.ripple, p.phase] {
                *out.add(i) = v;
                i += 1;
            }
        }
        *n_channels = chans.len();
    }
    0
}

/// Runs to a fixed evaluation count; returns final best fitness. The workhorse
/// for accelerating long searches: the fitness callback stays on the Python
/// side is NOT possible via this ABI without pyo3 — so this entry point
/// optimizes the built-in sphere benchmark (used for benchmarking the
/// interpreter overhead delta, and as a smoke test of the FFI path).
#[no_mangle]
pub extern "C" fn calibrix_bench_de(
    dim: usize, popsize: usize, generations: usize, seed: u64,
    lo: f64, hi: f64, out_best: *mut f64,
) -> i32 {
    if out_best.is_null() {
        return -1;
    }
    let f = |x: &[f64]| x.iter().map(|v| v * v).sum();
    let mut de = DifferentialEvolution::new(&f, dim, lo, hi, popsize, seed);
    for _ in 0..generations {
        de.step();
    }
    unsafe {
        *out_best = de.fit.iter().cloned().fold(f64::INFINITY, f64::min);
    }
    0
}
