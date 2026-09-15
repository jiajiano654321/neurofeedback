"""IT4 反馈会话状态机（纯逻辑：无 I/O、无线程、无真实时钟来源）。

相位（spec §5.1）：
    calibration → freeze_card → block ↔ rest ×6 → finished
异常终态 aborted：user_abort / freeze_failed / incomplete_data / wall_cap / engine_error。
相位推进权只在引擎同步循环（按已发射段硬推进），页面无继续/延长/跳过控件（C7）。
rest/freeze_card 只走墙钟、不推进任何数据侧状态（§4.4）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import config
from .theta_metric import FreezeFailed


@dataclass
class _Counts:
    n_valid: int = 0
    n_invalid: int = 0
    n_hit: int = 0


_ABORT_REASONS = (
    "user_abort", "freeze_failed", "incomplete_data", "wall_cap", "engine_error",
)


class FeedbackSession:
    """计数语义（spec §4.3/4.5，C5）：
    - valid 且 θ 有限：n_valid/n_hit 才动；无效段：只动 n_invalid；
    - valid 但 θ 非有限（实测不可达的守卫分支）：三者都不动；
    - hit_rate 分母只含 valid 段；n_valid=0 → 0.0，不除零。
    """

    def __init__(
        self,
        *,
        calib_segs: int | None = None,
        n_blocks: int | None = None,
        block_segs: int | None = None,
        wall_cap_sec: float | None = None,
        min_calib_segs: int | None = None,
    ):
        # 默认值在构造时从 config 读（不用 def-time 默认参数），便于测试 monkeypatch
        self.calib_segs = int(round(config.QC_CALIBRATION_SEC / config.SEGMENT_SEC)) \
            if calib_segs is None else calib_segs
        per_block = int(round(config.FEEDBACK_BLOCK_SEC / config.SEGMENT_SEC))
        self.n_blocks = config.FEEDBACK_N_BLOCKS if n_blocks is None else n_blocks
        self.block_segs = per_block if block_segs is None else block_segs
        self.wall_cap_sec = config.SESSION_WALL_CAP_SEC if wall_cap_sec is None else wall_cap_sec
        self.min_calib_segs = (
            config.THETA_MIN_CALIB_SEGS if min_calib_segs is None else min_calib_segs
        )

        self.phase = "calibration"
        self.status: str | None = None   # None=运行中；"ok" / "aborted"
        self.reason: str | None = None
        self.block_idx = -1
        self.seg_in_block = 0
        self.blocks_completed = 0
        self.block_rates: list[float] = []
        self.summaries: list[dict] = []
        self.freeze_info: dict | None = None
        self._block_t1_start: float | None = None
        self._cur = _Counts()
        self.total = _Counts()

    # ---- 冻结 ----
    def note_freeze(self, theta_freeze: dict) -> None:
        n = int(theta_freeze["n_thr_segs"])
        if n < self.min_calib_segs:
            raise FreezeFailed(
                f"S3 成熟有效段 {n} < 最小要求 {self.min_calib_segs}，冻结失败"
            )
        thr = float(theta_freeze["theta_thr_uV2"])
        if not math.isfinite(thr):
            raise FreezeFailed("S3 θ 阈值非有限值，冻结失败")
        self.freeze_info = {
            "b_fp1": float(theta_freeze["b_fp1"]),
            "b_fp2": float(theta_freeze["b_fp2"]),
            "theta_thr_uV2": thr,
            "n_thr_segs": n,
            "calib_hit_rate": float(theta_freeze["calib_hit_rate"]),
        }
        self.phase = "freeze_card"

    # ---- block / rest ----
    def begin_block(self, t1: float) -> dict:
        self.block_idx += 1
        if not (0 <= self.block_idx < self.n_blocks):
            raise RuntimeError(f"block 越界: {self.block_idx}")
        self.seg_in_block = 0
        self._block_t1_start = float(t1)
        self.phase = "block"
        return {
            "type": "phase", "phase": "block_start",
            "block_idx": self.block_idx, "t1": round(float(t1), 6),
        }

    def record_segment(self, *, valid: bool, power_finite: bool, hit: bool | None) -> None:
        if valid and power_finite:
            self._cur.n_valid += 1
            self.total.n_valid += 1
            if hit:
                self._cur.n_hit += 1
                self.total.n_hit += 1
        elif not valid:
            self._cur.n_invalid += 1
            self.total.n_invalid += 1
        self.seg_in_block += 1

    @staticmethod
    def _rate(c: _Counts) -> float:
        return round(c.n_hit / c.n_valid, 6) if c.n_valid else 0.0

    def end_block_if_due(self, t1_end: float) -> dict | None:
        if self.seg_in_block < self.block_segs:
            return None
        row = {
            "type": "block_summary",
            "block_idx": self.block_idx,
            "t1_start": round(float(self._block_t1_start), 6),
            "t1_end": round(float(t1_end), 6),
            "n_valid": self._cur.n_valid,
            "n_invalid": self._cur.n_invalid,
            "n_hit": self._cur.n_hit,
            "hit_rate": self._rate(self._cur),
        }
        self.summaries.append(row)
        self.block_rates.append(row["hit_rate"])
        self.blocks_completed += 1
        self._cur = _Counts()
        self.seg_in_block = 0
        return row

    def begin_rest(self, rest_sec: float) -> dict:
        rest_idx = self.blocks_completed - 1
        if not (0 <= rest_idx < self.n_blocks - 1):
            raise RuntimeError(f"rest 越界: {rest_idx}")
        self.phase = "rest"
        return {
            "type": "phase", "phase": "rest_start",
            "rest_idx": rest_idx, "rest_sec": float(rest_sec),
        }

    def end_rest(self, rest_sec: float) -> dict:
        rest_idx = self.blocks_completed - 1
        return {
            "type": "phase", "phase": "rest_end",
            "rest_idx": rest_idx, "rest_sec": float(rest_sec),
        }

    def finish(self) -> dict:
        if self.blocks_completed != self.n_blocks:
            raise RuntimeError(
                f"未完成全部 block 不能 finish：{self.blocks_completed}/{self.n_blocks}"
            )
        self.phase = "finished"
        self.status = "ok"
        return {
            "type": "final",
            "status": "ok",
            "n_valid": self.total.n_valid,
            "n_invalid": self.total.n_invalid,
            "n_hit": self.total.n_hit,
            "overall_hit_rate": self._rate(self.total),
            "block_rates": list(self.block_rates),
        }

    # ---- 异常终态 ----
    def terminate(self, reason: str, *, k: int, t1: float, error: str | None = None) -> dict:
        if reason not in _ABORT_REASONS:
            raise ValueError(f"未知 abort reason: {reason!r}")
        self.phase = "aborted"
        self.status = "aborted"
        self.reason = reason
        row = {
            "type": "aborted",
            "status": "aborted",
            "reason": reason,
            "k": int(k),
            "t1": round(float(t1), 6),
            "blocks_completed": self.blocks_completed,
        }
        if error is not None:
            row["error"] = str(error)
        return row

    def wall_cap_terminate(self, wall_elapsed: float, *, k: int, t1: float) -> dict | None:
        if self.status is not None:
            return None
        if wall_elapsed > self.wall_cap_sec:
            return self.terminate("wall_cap", k=k, t1=t1)
        return None
