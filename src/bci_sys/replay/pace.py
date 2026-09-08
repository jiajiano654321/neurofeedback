"""1x 节拍器。deadline 锚定起点、自校正，不累加漂移。时钟/sleep 可注入（测试用假时钟）。"""
from __future__ import annotations

import time

from .. import config

GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def rate_color(ratio: float) -> str:
    if ratio >= config.RATE_GREEN:
        return GREEN
    if ratio >= config.RATE_YELLOW:
        return YELLOW
    return RED


class Pacer:
    def __init__(self, segment_sec: float = config.SEGMENT_SEC, sleep_fn=time.sleep,
                 clock=time.perf_counter):
        self.segment_sec = segment_sec
        self._sleep = sleep_fn
        self._clock = clock
        self._wall0 = None

    def start(self):
        self._wall0 = self._clock()

    def wait_after_segment(self, seg_index: int) -> None:
        """第 seg_index（0 起）段处理完后调用：睡到 wall0 + (k+1)*段长。"""
        target = self._wall0 + (seg_index + 1) * self.segment_sec
        dt = target - self._clock()
        if dt > 0:
            self._sleep(dt)

    def rate_ratio(self, data_elapsed: float) -> float:
        wall = self.wall_elapsed()
        if wall <= 0:
            return 1.0
        return data_elapsed / wall

    def wall_elapsed(self) -> float:
        return self._clock() - self._wall0
