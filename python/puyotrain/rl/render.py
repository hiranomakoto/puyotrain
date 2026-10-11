"""リプレイ JSON から MP4 を作る。

リプレイを再シミュレーションし、トレース付き step の中間盤面
(設置後・消去前・消去後・落下後・おじゃま降下後)を順に描画する。

時間のモデル(mode = timed)のリプレイは、対局の時刻に沿って一定間隔で盤面を描く
(既定は実時間。設定 ``video.timed_speed`` で速さを変えられる)。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from collections import deque

from .._puyo import Game, TimedVersus, Versus
from .replay import load_replay

# ---------------------------------------------------------------- 見た目

CELL = 30
BOARD_W, BOARD_H = 6 * CELL, 13 * CELL
PANEL_W, PANEL_H = 384, 480          # 16 の倍数(H.264 のマクロブロック)
BOARD_X, BOARD_Y = 20, 64
SIDE_X = BOARD_X + BOARD_W + 20

BG = (24, 26, 33)
BOARD_BG = (40, 44, 56)
HIDDEN_BG = (30, 32, 41)             # 13段目
GRID = (52, 57, 72)
TEXT = (235, 235, 240)
MUTED = (150, 155, 170)
ACCENT = (255, 214, 90)
PUYO_COLORS = [
    (230, 70, 70),     # 赤
    (70, 190, 90),     # 緑
    (70, 120, 230),    # 青
    (235, 205, 60),    # 黄
    (165, 165, 175),   # おじゃま
]

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


# ---------------------------------------------------------------- シーン


@dataclass
class PlayerView:
    board: np.ndarray                 # (13, 6) int8
    cleared: Optional[np.ndarray]     # (13, 6) bool
    chain: int
    score: int
    max_chain: int
    next_pairs: list                  # [(axis, child), (axis, child)]
    label: str
    training_step: Optional[int]
    pending: int = 0
    banner: str = ""
    moves: Optional[int] = None        # 手数(プレイヤーごとに違うとき。None なら Scene.move)
    pending_text: str = ""             # 予告おじゃまの表示(空なら pending)
    falling: list = field(default_factory=list)  # 落下中のぷよ [(段(小数), 列, 色), ...]


@dataclass
class Scene:
    players: list[PlayerView]
    move: int
    frames: int
    footer: str = ""
    extra: dict = field(default_factory=dict)


def _duration(kind: str, cfg: dict) -> int:
    return cfg["frames_per_move"] if kind == "placed" else cfg["frames_per_chain_stage"]


def _step_text(step: Optional[int]) -> str:
    return f"STEP {step:,}" if step is not None else ""


def solo_scenes(replay: dict, cfg: dict):
    meta = replay["meta"]
    label = meta["labels"][0]
    step = meta["training_steps"][0]
    g = Game(int(replay["seed"]))
    max_chain = 0
    vp = g.visible_pairs()
    yield Scene([PlayerView(g.board(), None, 0, 0, 0, vp[:2], label, step)], 0,
                cfg["frames_per_move"])
    for a in replay["actions"]:
        _, frames = g.step_traced(int(a))
        vp = g.visible_pairs()
        for f in frames:
            max_chain = max(max_chain, f["chain"])
            cleared = f["cleared"] if f["kind"] == "before_clear" else None
            pv = PlayerView(f["board"], cleared, f["chain"], f["score"], max_chain,
                            vp[:2], label, step)
            yield Scene([pv], g.turn, _duration(f["kind"], cfg))
    end = "GAME OVER" if g.is_over else f"END ({g.turn} MOVES)"
    pv = PlayerView(g.board(), None, 0, g.score, max_chain, vp[:2], label, step, banner=end)
    yield Scene([pv], g.turn, cfg["end_hold_frames"])


def timed_scenes(replay: dict, cfg: dict):
    """時間のモデルのリプレイを、時刻に沿って一定間隔で描く(1シーン = 動画の1フレーム)。

    1フレームあたりの対局のフレーム数は 60 / fps × ``timed_speed``(既定1 = 実時間)。
    手を決めた時刻はリプレイの記録どおりに再現する。
    """
    meta = replay["meta"]
    labels, steps = meta["labels"], meta["training_steps"]
    g = TimedVersus(int(replay["seed"]), replay.get("timing"))
    events = deque(replay["actions"])
    step = max(1, round(60 / cfg["fps"] * float(cfg.get("timed_speed", 1.0))))
    t = 0

    def decide() -> None:
        # 時刻 t までの出来事を処理し、操作待ちになったプレイヤーにはリプレイの手を決めさせる
        while True:
            g.advance_until(t)
            acted = False
            for p in (0, 1):
                if g.needs_action()[p]:
                    when, who, a = events.popleft()
                    if who != p or when != g.now:
                        raise ValueError(f"replay mismatch at frame {g.now}: "
                                         f"expected player {p}, got {who} at {when}")
                    g.act(p, int(a))
                    acted = True
            if not acted:
                return

    def views(banners=("", "")) -> list[PlayerView]:
        scores, chains, moves = g.scores(), g.max_chains(), g.moves()
        pending = g.pending_ojama()
        out = []
        for p in (0, 1):
            d = g.display(p)
            confirmed, unconfirmed = pending[p]
            text = f"{confirmed}+{unconfirmed}" if unconfirmed else str(confirmed)
            out.append(PlayerView(d["board"], d["cleared"], d["chain"], scores[p], chains[p],
                                  g.visible_pairs(p)[1:3], labels[p], steps[p],
                                  confirmed + unconfirmed, banners[p], moves[p], text,
                                  d["falling"]))
        return out

    while True:
        decide()
        yield Scene(views(), 0, 1, footer=f"{g.now / 60:5.1f} s")
        if g.is_over:
            break
        t += step
    if g.timed_out:
        banners = ("END", "END")
    elif g.is_draw:
        banners = ("DRAW", "DRAW")
    else:
        banners = ("WIN", "LOSE") if g.winner == 0 else ("LOSE", "WIN")
    yield Scene(views(banners), 0, cfg["end_hold_frames"], footer=f"{g.now / 60:5.1f} s")


def versus_scenes(replay: dict, cfg: dict):
    meta = replay["meta"]
    labels, steps = meta["labels"], meta["training_steps"]
    v = Versus(int(replay["seed"]))
    max_chain = [0, 0]

    def view(i, board, cleared, chain, score, vp, pending, banner=""):
        return PlayerView(board, cleared, chain, score, max_chain[i], vp[:2], labels[i],
                          steps[i], pending, banner)

    vp = v.visible_pairs()
    yield Scene([view(i, v.board(i), None, 0, 0, vp, 0) for i in range(2)], 0,
                cfg["frames_per_move"])
    for a0, a1 in replay["actions"]:
        _, frames = v.step_traced(int(a0), int(a1))
        vp = v.visible_pairs()
        pending = v.pending_ojama()
        n = max(len(frames[0]), len(frames[1]))
        for k in range(n):
            views, dur = [], 0
            for i in range(2):
                fr = frames[i][min(k, len(frames[i]) - 1)]
                if k < len(frames[i]):
                    dur = max(dur, _duration(fr["kind"], cfg))
                    max_chain[i] = max(max_chain[i], fr["chain"])
                active = k < len(frames[i]) and fr["kind"] == "before_clear"
                views.append(view(i, fr["board"], fr["cleared"] if active else None,
                                  fr["chain"], fr["score"], vp, pending[i]))
            yield Scene(views, v.turn, dur)
    if not v.is_over:
        banners = ["END", "END"]
    elif v.is_draw:
        banners = ["DRAW", "DRAW"]
    else:
        banners = ["WIN", "LOSE"] if v.winner == 0 else ["LOSE", "WIN"]
    scores = v.scores()
    pending = v.pending_ojama()
    yield Scene([view(i, v.board(i), None, 0, scores[i], vp, pending[i], banners[i])
                 for i in range(2)], v.turn, cfg["end_hold_frames"])


# ---------------------------------------------------------------- 描画


def _puyo(d: ImageDraw.ImageDraw, x: int, y: int, size: int, color: int, highlight=False):
    pad = 2
    box = [x + pad, y + pad, x + size - pad, y + size - pad]
    rgb = PUYO_COLORS[color]
    if highlight:
        rgb = tuple(min(255, c + 70) for c in rgb)
    d.ellipse(box, fill=rgb, outline=(255, 255, 255) if highlight else None,
              width=3 if highlight else 1)
    if color == 4 and not highlight:
        d.ellipse(box, outline=(110, 110, 120), width=2)


def _draw_panel(img: Image.Image, ox: int, pv: PlayerView, move: int) -> None:
    d = ImageDraw.Draw(img)
    # タイトル
    d.text((ox + BOARD_X, 14), pv.label, font=_font(16), fill=TEXT)
    st = _step_text(pv.training_step)
    if st:
        d.text((ox + BOARD_X, 36), st, font=_font(14), fill=ACCENT)
    # 盤面
    bx, by = ox + BOARD_X, BOARD_Y
    d.rectangle([bx, by, bx + BOARD_W, by + BOARD_H], fill=BOARD_BG)
    d.rectangle([bx, by, bx + BOARD_W, by + CELL], fill=HIDDEN_BG)
    for c in range(7):
        d.line([bx + c * CELL, by, bx + c * CELL, by + BOARD_H], fill=GRID)
    d.line([bx, by + CELL, bx + BOARD_W, by + CELL], fill=GRID, width=2)
    # 窒息点(3列目12段目)
    dx, dy = bx + 2 * CELL, by + CELL
    d.line([dx + 6, dy + 6, dx + CELL - 6, dy + CELL - 6], fill=(120, 60, 60), width=2)
    d.line([dx + CELL - 6, dy + 6, dx + 6, dy + CELL - 6], fill=(120, 60, 60), width=2)
    for r in range(13):
        for c in range(6):
            v = int(pv.board[r, c])
            if v < 0:
                continue
            hl = pv.cleared is not None and bool(pv.cleared[r, c])
            _puyo(d, bx + c * CELL, by + (12 - r) * CELL, CELL, v, hl)
    for row, col, color in pv.falling:
        _puyo(d, bx + col * CELL, by + round((12 - row) * CELL), CELL, color)
    # 右側の情報
    sx = ox + SIDE_X
    y = BOARD_Y
    for title, pair in zip(("NEXT", "NEXT2"), pv.next_pairs):
        d.text((sx, y), title, font=_font(14), fill=MUTED)
        size = 26 if title == "NEXT" else 20
        axis, child = pair
        _puyo(d, sx + 4, y + 20, size, child)
        _puyo(d, sx + 4, y + 20 + size, size, axis)
        y += 30 + 2 * size
    y += 6
    rows = [("SCORE", f"{pv.score:,}"), ("CHAIN", str(pv.chain) if pv.chain else "-"),
            ("MAX CHAIN", str(pv.max_chain)),
            ("MOVE", str(pv.moves if pv.moves is not None else move))]
    if pv.pending:
        rows.append(("OJAMA", pv.pending_text or str(pv.pending)))
    for title, val in rows:
        d.text((sx, y), title, font=_font(12), fill=MUTED)
        color = ACCENT if title == "CHAIN" and pv.chain else TEXT
        d.text((sx, y + 14), val, font=_font(18), fill=color)
        y += 42
    if pv.chain and pv.cleared is not None:
        d.text((sx, PANEL_H - 52), f"{pv.chain} CHAIN!", font=_font(22), fill=ACCENT)
    if pv.banner:
        w = d.textlength(pv.banner, font=_font(24))
        cy = by + BOARD_H // 2
        d.rectangle([bx, cy - 22, bx + BOARD_W, cy + 22], fill=(0, 0, 0))
        d.text((bx + (BOARD_W - w) / 2, cy - 14), pv.banner, font=_font(24), fill=ACCENT)


def draw_scene(scene: Scene) -> np.ndarray:
    img = Image.new("RGB", (PANEL_W * len(scene.players), PANEL_H), BG)
    for i, pv in enumerate(scene.players):
        _draw_panel(img, i * PANEL_W, pv, scene.move)
    d = ImageDraw.Draw(img)
    if len(scene.players) == 2:
        d.line([PANEL_W, 8, PANEL_W, PANEL_H - 8], fill=GRID, width=2)
    if scene.footer:
        w = d.textlength(scene.footer, font=_font(16))
        d.text(((img.width - w) / 2, PANEL_H - 24), scene.footer, font=_font(16), fill=MUTED)
    return np.asarray(img)


def render_replay(replay: dict, out: Path, cfg: dict) -> Path:
    """リプレイ(dict)を MP4 にする。"""
    import imageio.v2 as imageio

    if replay["mode"] == "solo":
        scenes = solo_scenes(replay, cfg)
    elif replay["mode"] == "timed":
        scenes = timed_scenes(replay, cfg)
    else:
        scenes = versus_scenes(replay, cfg)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    with imageio.get_writer(str(tmp), fps=cfg["fps"], codec="libx264", quality=7,
                            pixelformat="yuv420p", macro_block_size=16,
                            ffmpeg_log_level="error") as w:
        for sc in scenes:
            frame = draw_scene(sc)
            for _ in range(max(1, int(sc.frames))):
                w.append_data(frame)
    tmp.replace(out)
    return out


def main(argv=None) -> None:
    from .config import load_config

    p = argparse.ArgumentParser(description="リプレイ JSON から MP4 を作る")
    p.add_argument("replays", type=Path, nargs="+")
    p.add_argument("-o", "--out", type=Path, help="出力先(リプレイ1つのときのみ。既定: 同名 .mp4)")
    p.add_argument("--config", type=Path, help="動画設定を読む YAML(既定: configs/solo_ppo.yaml)")
    p.add_argument("--fps", type=int)
    args = p.parse_args(argv)
    if args.out and len(args.replays) > 1:
        p.error("--out は リプレイ1つのときだけ使える")
    cfg = load_config(args.config)["video"]
    if args.fps:
        cfg["fps"] = args.fps
    for rp in args.replays:
        out = args.out or rp.with_suffix(".mp4")
        print(render_replay(load_replay(rp), out, cfg))


if __name__ == "__main__":
    main()
