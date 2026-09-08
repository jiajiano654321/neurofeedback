from bci_sys.replay import pace


class FakeClock:
    """假时钟：sleep 多久时钟就走多久，确定性测试用（不走真睡眠）。"""

    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, dt):
        self.slept.append(dt)
        self.t += dt


def test_pacer_sleeps_until_segment_deadline():
    clk = FakeClock()
    p = pace.Pacer(segment_sec=0.2, sleep_fn=clk.sleep, clock=clk.now)
    p.start()
    p.wait_after_segment(0)  # 第 0 段结束：应睡到 0.2
    assert abs(clk.slept[-1] - 0.2) < 1e-9
    # 模拟处理耗了 0.05s（时钟已随 sleep 走过 0.2，再手动加 0.05）
    clk.t += 0.05
    p.wait_after_segment(1)  # 目标 0.4，当前 0.25 → 睡 0.15
    assert abs(clk.slept[-1] - 0.15) < 1e-9


def test_pacer_does_not_sleep_when_behind():
    clk = FakeClock()
    p = pace.Pacer(segment_sec=0.2, sleep_fn=clk.sleep, clock=clk.now)
    p.start()
    clk.t += 1.0  # 已经落后 1 秒
    p.wait_after_segment(0)
    assert clk.slept == []  # 落后时不睡负时间


def test_rate_ratio():
    clk = FakeClock()
    p = pace.Pacer(segment_sec=0.2, sleep_fn=clk.sleep, clock=clk.now)
    p.start()
    p.wait_after_segment(0)  # 墙钟走 0.2，数据走 0.2 → 1.0x
    assert abs(p.rate_ratio(data_elapsed=0.2) - 1.0) < 1e-9


def test_rate_color_thresholds():
    assert pace.rate_color(0.99) == pace.GREEN
    assert pace.rate_color(0.98) == pace.GREEN
    assert pace.rate_color(0.97) == pace.YELLOW
    assert pace.rate_color(0.95) == pace.YELLOW
    assert pace.rate_color(0.94) == pace.RED
