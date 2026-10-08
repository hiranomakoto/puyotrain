//! 1秒あたりの手数を計測する。`cargo bench -p puyo-core`

use std::hint::black_box;
use std::time::{Duration, Instant};

use puyo_core::{Action, ActionMask, Game, Rng, Versus};

const BUDGET: Duration = Duration::from_secs(2);

#[inline]
fn random_legal(mask: ActionMask, rng: &mut Rng) -> Action {
    let mut k = rng.below(mask.count_ones());
    let mut m = mask;
    while k > 0 {
        m &= m - 1;
        k -= 1;
    }
    Action::new(m.trailing_zeros() as usize).unwrap()
}

fn report(name: &str, n: u64, elapsed: Duration) {
    let rate = n as f64 / elapsed.as_secs_f64();
    println!("{:<40} {:>14.0} /s  ({} in {:.2?})", name, rate, n, elapsed);
}

fn bench_solo() {
    let mut rng = Rng::new(1);
    let mut seed = 0;
    let mut g = Game::new(seed);
    let mut n = 0u64;
    let t = Instant::now();
    while t.elapsed() < BUDGET {
        for _ in 0..1000 {
            if g.is_over() {
                seed += 1;
                g = Game::new(seed);
            }
            let a = random_legal(g.legal_actions(), &mut rng);
            black_box(g.step(a).unwrap());
            n += 1;
        }
    }
    report("Game: legal_actions + step (random)", n, t.elapsed());
}

fn bench_versus() {
    let mut rng = Rng::new(2);
    let mut seed = 0;
    let mut v = Versus::new(seed);
    let mut n = 0u64;
    let t = Instant::now();
    while t.elapsed() < BUDGET {
        for _ in 0..1000 {
            if v.is_over() {
                seed += 1;
                v = Versus::new(seed);
            }
            let a0 = random_legal(v.legal_actions(0), &mut rng);
            let a1 = random_legal(v.legal_actions(1), &mut rng);
            black_box(v.step(a0, a1).unwrap());
            n += 1;
        }
    }
    report("Versus: step(a1, a2) (random)", n, t.elapsed());
}

/// 探索を想定: 中盤の局面から全合法手を clone + step で展開する。
fn bench_expand() {
    let mut rng = Rng::new(3);
    let mut positions = Vec::new();
    for s in 0..256 {
        let mut g = Game::new(s);
        for _ in 0..15 {
            if g.is_over() {
                break;
            }
            let a = random_legal(g.legal_actions(), &mut rng);
            g.step(a).unwrap();
        }
        if !g.is_over() {
            positions.push(g);
        }
    }
    let mut n = 0u64;
    let t = Instant::now();
    while t.elapsed() < BUDGET {
        for g in &positions {
            let mut m = g.legal_actions();
            while m != 0 {
                let a = Action::new(m.trailing_zeros() as usize).unwrap();
                m &= m - 1;
                let mut c = *black_box(g);
                black_box(c.step(a).unwrap());
                n += 1;
            }
        }
    }
    report("Game: clone + step (all children)", n, t.elapsed());
}

fn bench_legal() {
    let positions: Vec<Game> = (0..64).map(Game::new).collect();
    let mut n = 0u64;
    let t = Instant::now();
    while t.elapsed() < BUDGET {
        for g in &positions {
            black_box(black_box(g).legal_actions());
            n += 1;
        }
    }
    report("Game: legal_actions", n, t.elapsed());
}

fn main() {
    println!("puyo-core throughput (single thread)");
    bench_solo();
    bench_versus();
    bench_expand();
    bench_legal();
}
