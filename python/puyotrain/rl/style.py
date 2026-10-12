"""勝ち方(魅せ方)の評価: 決め手の連鎖の大きさと「返し勝ち」。

時間のモデルの学習で「大連鎖で魅せて勝つ」ことを目指すための報酬(``reward: style``)と、
評価の指標に使う。連鎖の記録は ``TimedVersus.chain_stats`` / ``TimedBatch.stats`` から取る。

- 決め手の連鎖: 勝った局で、勝った側が撃った最大の連鎖の連鎖数。
  (大連鎖で相手を埋めたあと、小さな連鎖でとどめを刺すことが多いので、最後に撃った連鎖ではなく
  最大の連鎖で数える)
- 大連鎖で決めた勝ち: 決め手の連鎖が ``big_chain`` 以上。
- 返し勝ち: 相手が ``counter_min_chain`` 以上の連鎖を撃ち、それより後に撃ち始めた
  それより大きい連鎖を撃って勝った。
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class StyleConfig:
    big_chain: int = 8             # 大連鎖とみなす連鎖数
    counter_min_chain: int = 5     # 返し勝ちに数える、相手の連鎖の最小
    win_low_chain: int = 3         # 勝ちの報酬が最小になる決め手の連鎖(これ以下)
    win_high_chain: int = 10       # 勝ちの報酬が最大になる決め手の連鎖(これ以上)
    win_min: float = 0.3           # 勝ちの報酬の最小(小さい連鎖で決めた勝ち)
    win_max: float = 1.5           # 勝ちの報酬の最大(大連鎖で決めた勝ち)
    counter_coef: float = 0.1      # 返しの報酬: 自分の連鎖で相殺したおじゃま30個あたり
    potential_floor: float = 0.2   # 潜在連鎖の整形を、減らしてもこの割合までは残す

    @classmethod
    def from_dict(cls, d: dict | None) -> "StyleConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d or {}) - names
        if unknown:
            raise KeyError(f"unknown style keys: {sorted(unknown)}")
        return cls(**(d or {}))


def win_reward(finish_chain: int, style: StyleConfig) -> float:
    """勝ったときの報酬。決め手の連鎖が大きいほど大きい(win_min 〜 win_max の間を直線で)。"""
    span = max(1, style.win_high_chain - style.win_low_chain)
    t = min(1.0, max(0.0, (finish_chain - style.win_low_chain) / span))
    return style.win_min + (style.win_max - style.win_min) * t


def finish_chain(winner: dict) -> int:
    """決め手の連鎖(勝った側の最大連鎖)。``winner`` は連鎖の記録(max_chain などを持つ dict)。"""
    return int(winner["max_chain"])


def is_counter_win(winner: dict, loser: dict, style: StyleConfig) -> bool:
    """返し勝ちか。``winner`` / ``loser`` は連鎖の記録(max_chain・max_chain_start を持つ dict)。"""
    return (loser["max_chain"] >= style.counter_min_chain
            and winner["max_chain"] > loser["max_chain"]
            and winner["max_chain_start"] >= loser["max_chain_start"])


@dataclass(frozen=True)
class DrillConfig:
    """組む練習の対局(相手は何もしない)の設定。

    学習中の対局のうち ``fraction`` の割合を練習の対局にする。練習の対局では、``moves`` 手のうちに
    撃った連鎖ごとに ``drill_reward`` の報酬を出し、窒息は −1、``moves`` 手に達したら終わり
    (撃たずに終われば報酬なし。打ち切りではなく終了として扱う)。
    """

    fraction: float = 0.0          # 練習の対局の割合(0 なら使わない)
    moves: int = 32                # 1局の手数
    low_chain: int = 3             # 報酬が0になる連鎖数(これ以下)
    high_chain: int = 10           # 報酬が1になる連鎖数(これ以上)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DrillConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d or {}) - names
        if unknown:
            raise KeyError(f"unknown drill keys: {sorted(unknown)}")
        return cls(**(d or {}))


def drill_reward(chain: int, drill: DrillConfig) -> float:
    """練習の対局で、撃った連鎖1回の報酬: ((連鎖数 − low) / (high − low))² を 0〜1 に収めたもの。
    大きな連鎖を1回撃つほうが、小さな連鎖を何回も撃つよりずっと得になる。"""
    span = max(1, drill.high_chain - drill.low_chain)
    t = min(1.0, max(0.0, (chain - drill.low_chain) / span))
    return t * t
