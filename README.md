# puyotrain

ぷよぷよ通ルール準拠の対戦シミュレータ(Rust)と、Python から使う強化学習環境(Gymnasium 互換)。

| 構成 | 内容 |
|---|---|
| `puyo-core/` | ルール実装。実行時の外部依存なし、`#![forbid(unsafe_code)]` |
| `puyo-py/` | PyO3 + maturin による Python バインディング(`puyotrain._puyo`) |
| `python/puyotrain/` | `PuyoSoloEnv` / `PuyoVersusEnv` |
| `python/puyotrain/rl/` | 1人プレイの学習基盤(ベースライン、MaskablePPO、リプレイ、動画) |
| `configs/solo_ppo.yaml` | 学習設定(1ファイルに集約) |
| `scripts/` | `train.py` / `replay.py` / `render.py` / `render_all.py` / `baseline.py` / `bench_train.py`、起動用の `start_training.sh` / `check_env.sh` / `*.ps1` |
| `docker/`, `docker-compose.yml` | Docker で学習するためのイメージと設定 |
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

## 学習基盤(1人プレイ)

### インストール

```bash
pip install -e ".[rl]"     # torch, sb3-contrib, tensorboard, pyyaml, pillow, imageio-ffmpeg
```

`imageio-ffmpeg` は ffmpeg のバイナリを同梱しているので、Windows でも ffmpeg を別に入れる必要はありません。

**GPU(Windows / Linux)を使う場合**は、先に PyTorch 公式の CUDA 版を入れてから `.[rl]` を入れます
(CUDA のバージョンは https://pytorch.org/get-started/locally/ で自分の環境に合うものを選ぶ)。

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[rl]"
python -c "import torch; print(torch.cuda.is_available())"   # True なら OK
```

設定の `device: auto` で、GPU があれば自動で使います。GPU で速くなるかは構成次第なので、
次のコマンドで CPU と GPU の学習速度(環境ステップ/秒)を比べてから決めてください。

```bash
python scripts/bench_train.py --devices cpu cuda --steps 30000
python scripts/bench_train.py --devices cpu cuda --set env.n_envs=32   # 並列数を変えて比較
```

### ベースライン

```bash
python scripts/baseline.py random greedy --seeds 100     # 固定シード 0..99、上限500手
```

- `random`: 合法手から一様に選ぶ。
- `greedy`: 2手読み(現在 + ネクストの 22×22 通り)で合計得点が最大の手を選ぶ。同点の場合は
  「窒息しない → 同色の隣接数が多い → 最大の高さが低い → 乱数」の順に決める(`puyo-core/src/bot.rs`)。

| エージェント | 平均得点 | 平均最大連鎖数 | 平均手数 | 窒息率 |
|---|---|---|---|---|
| random | 1,271 | 1.77 | 42 | 100% |
| greedy | 50,985 | 4.11 | 500(上限) | 0% |

### 学習

```bash
python scripts/train.py                                    # configs/solo_ppo.yaml の設定で学習
python scripts/train.py --set ppo.total_timesteps=200000 --set checkpoint.every_steps=50000
tensorboard --logdir runs
```

- アルゴリズムは sb3-contrib の `MaskablePPO`。行動マスクは環境の `action_masks()` から取ります。
- 方策ネット(`python/puyotrain/rl/features.py`): 自分の盤面(6平面 × 13 × 6)を 3×3 畳み込み2層に通し、
  ネクスト・ネクネクと予告おじゃまを結合して全結合層へ。相手盤面は使わないので、同じ重みで対戦もできます。
- 環境は `env.n_envs` 個を並列に進めます(既定は1プロセス内でまとめて進める `DummyVecEnv`)。
  環境の1手は Rust で非常に速いため、プロセスを分ける `SubprocVecEnv`(`env.vec_env: subproc`)より
  プロセス間通信の分だけ遅くなりにくく、計測では 8並列で 1,581 対 1,128 ステップ/秒でした。
  報酬は 得点 / 1000。`env.death_penalty` で窒息した手から罰を引き、`env.survival_bonus` で
  生き残った手ごとにボーナスを足せます(どちらも既定 0)。
- 大連鎖を組ませるための設定(どちらも既定はオフ):
  - `env.potential_coef`: 潜在連鎖(どこかの列に同じ色を1〜2個足したときに起きる最大の連鎖数)を
    ポテンシャル Φ とする報酬整形の係数。毎手 `gamma × Φ(後) − Φ(前)` を足すので、連鎖の種を
    育てると即座に報酬が入り、小さく消して種を壊すと減る。最適方策は変えない形の整形です。
  - `env.safe_mask`: 置いた直後に窒息する手と、ネクストをどう置いても窒息する手を行動マスクで禁止する。
    評価・リプレイでもチェックポイントの記録からこの設定を読み取り、同じマスクで打ちます。
- `policy.action_features: true` にすると、22通りの手それぞれについて置いた結果(起きる連鎖・
  置いた後の潜在連鎖・窒息・高さ・ネクストで起こせる最大の連鎖など、`ACTION_FEATURE_NAMES` の13個)を
  Rust で計算して観測に加え、方策は全手共通の小さな MLP で手ごとに点数をつけてそのままロジットにします
  (`PuyoActionFeatures` / `PuyoActionPolicy`)。価値は盤面 CNN の特徴と手ごとの点数から計算します。
  学習速度はおよそ6割になります。
- 評価には `big_chain_rate`(10連鎖以上を撃てたゲームの割合)と `mean_max_potential`
  (1局中の潜在連鎖の最大値の平均)も出ます。
- 設定はすべて `configs/solo_ppo.yaml` にあり、`--set キー=値` で上書きできます。

TensorBoard に記録する指標:

| タグ | 内容 |
|---|---|
| `eval/mean_score`, `eval/mean_max_chain` | 固定シード(既定100本)での平均得点・平均最大連鎖数(チェックポイントごと) |
| `eval/score_vs_random`, `eval/score_vs_greedy`, `eval/chain_vs_*` | ベースラインとの比(1.0 で同等) |
| `baseline/*` | ベースラインの値(比較用の水平線) |
| `eval/mean_moves`, `eval/death_rate` | 平均手数・窒息率 |
| `rollout/ep_score_mean`, `rollout/ep_max_chain_mean` | 学習中のエピソードの得点・最大連鎖 |

### Docker で学習する(Windows ではこちらを推奨)

Docker Desktop(WSL2 バックエンド)の Linux コンテナ内でビルドと学習を行います。
Windows 側に Rust や Python を入れる必要がなく、Windows 11 の「スマート アプリ コントロール」に
ビルドを止められることもありません。リポジトリのフォルダをそのままコンテナにマウントするので、
チェックポイントや動画はエクスプローラーから見られます。

前提: Docker Desktop(Settings → General の「Use the WSL 2 based engine」がオン)と、
GPU を使う場合は Windows 側の NVIDIA ドライバ(WSL 用の追加インストールは不要)。

```powershell
cd C:\Users\<ユーザー名>\dev\puyotrain
git pull
docker compose build                                         # 初回のみ。数分〜十数分
docker compose run --rm train bash scripts/check_env.sh      # GPU 確認・テスト・速度計測
docker compose run --rm train                                # 学習開始(2回目以降は自動で再開)
docker compose up -d tensorboard                             # http://localhost:6006
```

- `docker compose run --rm train` は `scripts/start_training.sh` を実行し、`runs/solo_main` に学習します。
  オプションは後ろに付けます:
  `docker compose run --rm train bash scripts/start_training.sh --device cuda --n-envs 16 --total-steps 100000000`
- **止め方**: Ctrl+C(バックグラウンドで動かしている場合は `docker stop <コンテナ名>`)。
  どちらでもその時点の重みを保存してから終了し、同じコマンドで再開できます。
- 長時間回すときは `docker compose run -d --name puyo-train train` でバックグラウンド実行し、
  `docker logs -f puyo-train` でログを見られます(止めるときは `docker stop puyo-train`、
  その後 `docker rm puyo-train`)。
- **PyTorch の CUDA 版**: 既定は CUDA 12.8 版(`cu128`)です。GPU ドライバが新しければ
  `$env:TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"; docker compose build`、
  `check_env.sh` で `CUDA available: False` になる場合は `cu126` で作り直してください。
- コードを更新したら `git pull` するだけです(起動時に Rust 拡張を差分ビルドします)。
  依存パッケージを変えたときだけ `docker compose build` し直します。
- GPU の無い PC では `docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm train` のように
  CPU 用の設定を重ねます。

### Windows に直接環境を作る(スマート アプリ コントロールがオフの場合)

Windows 11 の「スマート アプリ コントロール」がオンだと、Rust のビルド中に作られる実行ファイルが
ブロックされます(`os error 4551`)。その場合は上の Docker を使ってください。
オフの PC では、次のスクリプトで Windows 上に直接環境を作れます。

前提: Git、Rust(rustup)、Visual Studio Build Tools(C++)、Python 3.10〜3.13。無いものは
セットアップスクリプトが `winget` のインストールコマンドを表示します。

```powershell
cd C:\Users\<ユーザー名>\dev
git clone https://github.com/hiranomakoto/puyotrain.git
cd puyotrain
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1     # 環境構築・テスト・速度計測
powershell -ExecutionPolicy Bypass -File scripts\start_training.ps1    # 学習開始(2回目以降は自動で再開)
.venv\Scripts\tensorboard.exe --logdir runs                            # 別のターミナルで進捗を見る
```

- `setup_windows.ps1` は、NVIDIA GPU があれば PyTorch の CUDA 版を新しい順(cu130 → cu128 → cu126)に試し、
  GPU が使えることを確認します(どれも使えなければ CPU 版に切り替えて警告)。
- `start_training.ps1` のオプション: `-Device cuda`、`-NEnvs 16`、`-TotalSteps 100000000`、
  `-RunName 名前`、`-Set "ppo.learning_rate=0.0001"`。
- Linux / WSL に直接入れる場合は、README 冒頭の「ビルド」の手順のあと
  `bash scripts/check_env.sh` と `bash scripts/start_training.sh` が使えます。

### 長時間の学習と再開

```bash
# 例: 目標 5,000万ステップ、並列 16、チェックポイント 50万ごと、動画は 4 回に 1 回
python scripts/train.py --set ppo.total_timesteps=50000000 --set env.n_envs=16 \
    --set checkpoint.every_steps=500000 --set video.every_checkpoints=4

# 止めた(または目標に到達した)ランを、最新のチェックポイントから続ける
python scripts/train.py --resume runs/<run>
python scripts/train.py --resume runs/<run> --set ppo.total_timesteps=100000000   # 目標を延ばす
python scripts/train.py --resume runs/<run> --set ppo.learning_rate=0.0001         # 設定を変えて続ける

# 古いチェックポイントから別の方針で分岐する(元のランはそのまま残る)
python scripts/train.py --resume runs/<run> --from runs/<run>/checkpoints/step_0010000000.zip \
    --run-dir runs/<run>-lr1e-4 --set ppo.learning_rate=0.0001
```

- **Ctrl+C で止めても失われません。** その時点の重みをチェックポイントとして保存してから終了します
  (評価は行い、動画は省略。後で `render_all.py` で作れます)。
- `--resume` はランの `config.yaml` を引き継ぎ、`--set` で変えた分だけ上書きします。
  `ppo.total_timesteps` は「通算でここまで」の目標値です。
- 再開しても、ステップ番号、TensorBoard の曲線、`metrics.jsonl`、経過時間は続きから記録されます。
  再開の履歴は `resume_history.jsonl` に残ります。
- オプティマイザの状態はチェックポイントに含まれるので、そのまま続きから学習します。
  ロールアウトの途中のデータと、直近エピソードの統計(`rollout/*` の平均の元)は引き継がれません。
- 変えられないのは `policy`(ネットワークの形)だけです。並列数、学習率、バッチサイズなどは変えられます。
- 古いチェックポイントから同じランに続けると後のチェックポイントを上書きしてしまうため、
  その場合は `--run-dir` で分岐先を指定する必要があります。

**ディスク使用量の目安**: チェックポイント1つ約 17 MB(重み + オプティマイザの状態)、
500手の動画1本 約 2.4 MB。チェックポイントは消さない仕様なので、
5,000万ステップを 50万ごとに保存すると約 1.7 GB、動画を毎回3本作るとさらに最大約 0.7 GB です。
間隔は `checkpoint.every_steps` と `video.every_checkpoints` で調整してください。

**並列数の目安**: `env.n_envs` は CPU の論理コア数程度から始め、`bench_train.py` で
`--set env.n_envs=...` を変えながら一番速い値を選ぶのがおすすめです。

### 出力(チェックポイント・指標・動画)

```
runs/<run_name>-<日時>/
  config.yaml, config.json          使った設定
  baseline.json                     ベースラインの評価結果(固定シード)
  metrics.jsonl                     チェックポイントごとの指標(1行1件)
  checkpoints/step_0000050000.zip   重み(checkpoint.every_steps ごと。古いものも消さない。
                                    学習終了時の重みも保存する。PPO はロールアウト単位で
                                    終わるので、最後の1つは total_timesteps を少し超えた番号になる)
  checkpoints/step_0000050000.json  その時点の指標・ベースライン比較・設定・経過時間
  replays/step_0000050000_seed0.json リプレイ
  videos/step_0000050000_seed0.mp4  動画(video.seeds の3本。保存のたびに自動生成)
  videos/baseline_greedy_seed0.mp4  ベースラインの同じシードの動画(学習開始時)
  tb/                               TensorBoard
  resume_history.jsonl              再開の履歴(再開したときだけ)
```

### 短い学習の実行例

`--set ppo.total_timesteps=200000 --set checkpoint.every_steps=50000`(他は既定値)で、
CPU 4コアの Linux VM 上で約9分(評価・動画生成込み)。固定シード100本での評価:

| ステップ | 平均得点 | 平均最大連鎖数 | 平均手数 | random 比 | greedy 比 |
|---|---|---|---|---|---|
| 50,000 | 2,289 | 2.19 | 51 | 1.80 | 0.045 |
| 100,000 | 4,729 | 2.89 | 70 | 3.72 | 0.093 |
| 150,000 | 1,909 | 2.10 | 41 | 1.50 | 0.037 |
| 200,000 | 2,295 | 2.29 | 53 | 1.81 | 0.045 |
| 200,704(終了時) | 2,529 | 2.29 | 52 | 1.99 | 0.050 |

この長さではランダムは上回るものの、まだすぐ窒息する段階です(窒息率 100%)。
同じ VM での学習速度は CPU で約 820 環境ステップ/秒(`bench_train.py`、8並列)。

### リプレイと動画

```bash
# チェックポイント(または random / greedy)で固定シードを打ってリプレイ JSON を保存し、動画も作る
python scripts/replay.py --agent runs/<run>/checkpoints/step_0000200000.zip --seeds 0 1 2 --out-dir out --render

# 対戦: 2つ指定すると盤面を左右に並べた動画になる(左 = --agent、右 = --agent2)
python scripts/replay.py --agent runs/<run>/checkpoints/step_0000200000.zip \
    --agent2 runs/<run>/checkpoints/step_0000050000.zip --seeds 0 --out-dir out --render

# リプレイ JSON から MP4 を作る
python scripts/render.py out/step_0000200000_seed0.json -o out/seed0.mp4

# ランの全チェックポイント(とベースライン)の動画をまとめて作り直す
python scripts/render_all.py runs/<run>            # 未作成のものだけ
python scripts/render_all.py runs/<run> --force    # すべて作り直す(リプレイ JSON は再利用)
```

リプレイ形式(`python/puyotrain/rl/replay.py`):

```json
{"format": "puyotrain-replay", "version": 1, "mode": "solo", "seed": 0,
 "actions": [3, 12, 7],
 "meta": {"agents": [{"type": "ppo", "checkpoint": "...", "step": 200000}],
          "labels": ["step_0000200000"], "training_steps": [200000],
          "final_scores": [12340], "max_chains": [3], "moves": 500,
          "result": "max_moves", "max_moves": 500, "created_at": "..."}}
```

対戦の `actions` は `[[左, 右], ...]` です。シミュレータは決定的なので、シードと行動列だけで対局を完全に再現できます。

動画は、盤面、ネクスト・ネクネク、得点、現在の連鎖数、最大連鎖数、手数、学習ステップ数を描画します。
連鎖は、`puyo-core` のトレース付き step(`Game.step_traced` / `Versus.step_traced`)が返す中間盤面
(設置後 → 消去前(消えるぷよを強調)→ 消去後 → 落下後、対戦ではおじゃま降下後も)を順に再生します。
コマ数や fps は設定の `video` で変えられます。

## 学習基盤(対戦)

方針と経緯は [document/versus_plan.md](document/versus_plan.md) と [document/versus_log.md](document/versus_log.md)。

### 勝率の測定

```bash
# 2つのエージェントを、各シードで左右を入れ替えて2局ずつ打たせる(100シード = 200局)
python scripts/match.py runs/<run>/checkpoints/step_XXXXXXXXXX.zip greedy --seeds 100 --out out/match.json
```

### 自己対戦の学習

```bash
python scripts/train_versus.py                                   # configs/versus_ppo.yaml の設定で学習
python scripts/train_versus.py --run-dir runs/versus_main        # 出力先を指定
python scripts/train_versus.py --resume runs/versus_main         # 再開(--from / --run-dir で分岐も可)

# Docker
docker compose run --rm train python scripts/train_versus.py --run-dir runs/versus_main
```

- `init_from` の1人プレイの重み(`policy.action_features`)から始める。形が同じ部分はコピーし、
  追加した部分(相手の情報、予告おじゃまの特徴量)は影響ゼロで始めるので、学習開始時点の方策は
  1人プレイの重みと同じ。価値ヘッドは引き継がない。
- 観測: 手ごとの特徴量は対戦用の20個(`VERSUS_ACTION_FEATURE_NAMES`。1人用の13個に、この手で相殺・送る
  おじゃまの数、降ってくる数、降った後に窒息するか・残る潜在連鎖などを足したもの)。対局全体の情報
  `versus_info`(双方の予告おじゃま、潜在連鎖、相手の高さ)。相手の盤面は自分と同じ CNN に通す。
- 相手: 対局ごとに `pool.weights` の重みで greedy / 1人プレイの重み(`pool.solo`)/ 過去の自分
  (`pool.snapshot_every` ごとに追加、最大 `pool.max_snapshots`)から選ぶ。学習側の席も対局ごとにランダム。
- 報酬: 勝ち +1 / 負け −1。補助報酬(生んだおじゃま × `env.attack_coef`、潜在連鎖の整形
  × `env.potential_coef`)は `env.anneal_steps` で 0 まで直線的に減らす。
- 評価: チェックポイントごとに `eval.opponents`(既定は greedy と1人プレイの重み)と
  `eval.seeds` × 左右入れ替えで対局し、勝率を `metrics.jsonl` と TensorBoard(`eval/win_rate_vs_*`)に出す。
  学習中の勝率は `rollout/win_rate`(相手別は `rollout/win_rate_greedy` など)。
- 動画: greedy との対局を左右入れ替えて録画する(`videos/step_..._vs_greedy_p1_seed0.mp4` など)。

## テスト

- `puyo-core/tests/rules.rs`: 得点(4個消し40点、2連鎖360点、連結・色数ボーナス)、おじゃまの換算・相殺・30個上限・全消しボーナス、猶予1手、窒息・引き分け
- `puyo-core/tests/simulation.rs`: 落下後に浮いたぷよが無いこと(proptest)、同じシードでの再現、ランダムエージェント同士1000試合
- `puyo-core/tests/trace.rs`: トレース付き step が通常の step と同じ結果になること、中間盤面の順序・得点・連鎖数
- `python/tests/test_envs.py`: Gymnasium の `check_env`、観測レイアウト、再現性、1000試合
- `python/tests/test_rl.py`: トレースのバインディング、ベースライン、リプレイの往復、動画生成、短い学習(チェックポイント・指標 JSON・動画・作り直し)。RL 用の依存が無い環境ではスキップ
