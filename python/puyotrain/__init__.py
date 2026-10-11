"""ぷよぷよ通ルール準拠の対戦シミュレータ。"""

from gymnasium.envs.registration import register

from ._puyo import HEIGHT, NUM_ACTIONS, NUM_COLORS, WIDTH, Game, TimedVersus, Versus
from .envs import PuyoSoloEnv, PuyoVersusEnv

register(id="puyotrain/PuyoSolo-v0", entry_point="puyotrain.envs:PuyoSoloEnv")

__all__ = [
    "Game",
    "Versus",
    "TimedVersus",
    "PuyoSoloEnv",
    "PuyoVersusEnv",
    "NUM_ACTIONS",
    "WIDTH",
    "HEIGHT",
    "NUM_COLORS",
]
