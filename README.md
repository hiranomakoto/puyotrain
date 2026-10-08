# puyotrain

ぷよぷよ通ルール準拠の対戦シミュレータ(Rust)と、Python から使う強化学習環境(Gymnasium 互換)。

| 構成 | 内容 |
|---|---|
| `puyo-core/` | ルール実装。実行時の外部依存なし、`#![forbid(unsafe_code)]` |
| `puyo-py/` | PyO3 + maturin による Python バインディング(`puyotrain._puyo`) |
| `python/puyotrain/` | `PuyoSoloEnv` / `PuyoVersusEnv` |
| `python/tests/` | pytest |

ルール定数は `puyo-core/src/rules.rs` に集約しています。

## ビルド

必要なもの: Rust stable(1.83 以上)、Python 3.9 以上。

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop --release     # puyotrain を venv にインストール(numpy, gymnasium も入る)

cargo test --workspace        # Rust のテスト
pytest                        # Python のテスト
```

### Windows 11(PowerShell)

[rustup](https://rustup.rs/) で Rust を、python.org または winget で Python を入れておきます。
Rust の MSVC ツールチェーンには Visual Studio Build Tools(「C++ によるデスクトップ開発」)が必要です。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install maturin pytest
maturin develop --release

cargo test --workspace
pytest
```

配布用の wheel は `maturin build --release` で `target/wheels/` に作られます(abi3 なので Python 3.9 以上で共通)。

## 使い方

### Python

```python
import numpy as np
from puyotrain import PuyoSoloEnv, PuyoVersusEnv

# 1人用(Gymnasium 準拠)。報酬はその手の得点。
env = PuyoSoloEnv()
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
done = False
while not done:
    action = rng.choice(np.flatnonzero(obs["action_mask"]))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print("score:", info["total_score"])

# 2人対戦。報酬は勝ち +1 / 負け -1 / 引き分け 0。
env = PuyoVersusEnv()
(obs1, obs2), info = env.reset(seed=0)
while True:
    a1 = rng.choice(np.flatnonzero(obs1["action_mask"]))
    a2 = rng.choice(np.flatnonzero(obs2["action_mask"]))
    (obs1, obs2), (r1, r2), terminated, truncated, info = env.step(a1, a2)
    if terminated or truncated:
        break
print("winner:", info["winner"], "rewards:", (r1, r2))
```

`gymnasium.make("puyotrain/PuyoSolo-v0")` でも作れます。`max_steps` / `max_turns` を渡すと
その手数で `truncated=True` になります(既定は無制限)。

探索などで直接使う低レベル API:

```python
from puyotrain import Game, Versus

g = Game(seed=1)
mask = g.legal_actions()                 # int のビットマスク
score, chains, all_clear, dead = g.step(2)
child = g.clone()                        # 高速コピー(copy.copy / deepcopy も可)
print(g.field_str())

v = Versus(seed=1)
p1, p2 = v.step(0, 5)                    # 各プレイヤーの結果 dict
print(v.pending_ojama(), v.winner, v.is_draw)
```

### Rust

```rust
use puyo_core::{Action, Game, Versus};

let mut g = Game::new(42);
let mask = g.legal_actions();                     // u32 ビットマスク
let a = Action::new(mask.trailing_zeros() as usize).unwrap();
let r = g.step(a)?;                               // 合法でなければ Err
println!("{} 連鎖 {} 点", r.placement.chain.chains, r.placement.chain.score);

let snapshot = g;                                 // Game / Versus / Field は Copy
let mut v = Versus::new(42);
let res = v.step(a, a)?;
if let Some(outcome) = res.outcome { println!("{:?}", outcome); }
```

## 観測

`dict` で、各プレイヤー視点(自分が先)です。

| キー | 形状 / dtype | 内容 |
|---|---|---|
| `board` | `(2, 6, 13, 6)` uint8 | [自分, 相手] × 平面[空, 赤, 緑, 青, 黄, おじゃま] × 段(0 = 1段目, 12 = 13段目)× 列(0 = 左端)。1人用では相手側はすべて0 |
| `pieces` | `(3, 2, 4)` uint8 | [現在, ネクスト, ネクネク] × [軸, 子] × 色のワンホット |
| `ojama` | `(2,)` int32 | [自分, 相手] の予告おじゃま数 |
| `action_mask` | `(22,)` int8 | 合法手なら1 |

`PuyoSoloEnv.action_masks()` は sb3-contrib の `MaskablePPO` と互換です。

## 行動(22通り)

| 番号 | 置き方 | 軸の列 |
|---|---|---|
| 0–5 | 縦、子が上 | 0–5 |
| 6–11 | 縦、子が下 | 0–5 |
| 12–16 | 横、子が右 | 0–4 |
| 17–21 | 横、子が左 | 1–5 |

出現列(左から3列目)から配置する列までの間(両端を含む)に、高さ12以上の列があればその手は非合法です。
軸と子が同色でも、重複する手はマスクから外しません。

## ルール

- 盤面は6列×13段。13段目には置けますが、連結判定には含めません(13段目のおじゃまも消えません)。
- 色は4色 + おじゃま。4個以上の連結で消え、隣接するおじゃまも消えます。消去数におじゃまは含めません。
- 得点 = 消去数 × 10 × max(1, 連鎖 + 連結 + 色数ボーナス)。連鎖ボーナスは 0, 8, 16, 32, 64, 96, 128, 以降 +32(上限なし)。落下ボーナスはありません。
- おじゃまは70点で1個、端数は持ち越します。全消しすると、次の連鎖のおじゃまに30個を上乗せします(得点には加えません)。
- 1回に降るおじゃまは最大30個です。6個ごとに全列へ1段、端数は重複しないランダムな列に落とします(シード由来の乱数)。13段を超えた分は消えます。
- 左から3列目の12段目が埋まると窒息です。両者同時に窒息すると引き分けです。
- ツモはシードから決まり、両プレイヤーに同じ列を配ります。現在・ネクスト・ネクネクの3組を公開します。初手の色制限はありません。

### 同期ターンの進行(`puyo-core/src/versus.rs`)

1. 両者が同じ組ぷよを同時に置き、それぞれ連鎖を最後まで解決する。
2. このターンに連鎖しなかったプレイヤーには、**前のターンまでに**届いていた予告おじゃまが最大30個降る。
   そのターンに届いた分はまだ降らないので、受けた側には相殺する猶予が1手ある。
3. 攻撃を授受する。まず自分の予告を相殺し、残った攻撃同士を打ち消して、差分を相手の予告に加える。
4. 窒息判定。

設置・連鎖・授受・降下・窒息判定の各部品は `puyo-core/src/phase.rs` に分けてあります。
将来フレーム時間モデルを足すときは、これらを時間付きで呼ぶ別のスケジューラを書けば済むようにしています。
`ChainResolver` は1連鎖ずつ進むイテレータです。

## ベンチマーク

```bash
cargo bench -p puyo-core                     # Rust
python python/benchmarks/bench_env.py        # Python
```

参考値(Intel Xeon 2.8GHz、4コアの Linux VM、1スレッド):

| 計測 | 手/秒 |
|---|---|
| Rust `Game`: legal_actions + step(ランダム) | 約 1,970,000 |
| Rust `Versus`: step(2人分) | 約 1,130,000 |
| Rust `Game`: clone + step(全合法手を展開) | 約 2,870,000 |
| Rust `legal_actions` のみ | 約 20,300,000 |
| Python `Game.step`(観測なし) | 約 224,000 |
| Python `PuyoSoloEnv.step`(観測あり) | 約 136,000 |
| Python `PuyoVersusEnv.step`(観測2人分) | 約 54,000 |

## テスト

- `puyo-core/tests/rules.rs`: 得点(4個消し40点、2連鎖360点、連結・色数ボーナス)、おじゃまの換算・相殺・30個上限・全消しボーナス、猶予1手、窒息・引き分け
- `puyo-core/tests/simulation.rs`: 落下後に浮いたぷよが無いこと(proptest)、同じシードでの再現、ランダムエージェント同士1000試合
- `python/tests/test_envs.py`: Gymnasium の `check_env`、観測レイアウト、再現性、1000試合
