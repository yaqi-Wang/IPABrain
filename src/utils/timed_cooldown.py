"""Persist task cooldowns when an individual task runs for a long time."""
import json
import time
from pathlib import Path

from src.utils.staged_queue import atomic_json, local_time


class TimedCooldown:
    def __init__(self, directory, threshold=3600, seconds=120, *,
                 clock=time.time, monotonic=time.monotonic, sleeper=time.sleep, announce=print):
        self.path = Path(directory) / 'timed_cooldown.json'
        self.threshold, self.seconds = threshold, seconds
        self.clock, self.monotonic, self.sleeper, self.announce = clock, monotonic, sleeper, announce
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def wait(self):
        deadline = self.state.get('rest_until', 0)
        while self.clock() < deadline:
            remaining = deadline - self.clock()
            self.announce(f'[休息] 剩余 {remaining:.0f} 秒；恢复时间 {local_time(deadline)}', flush=True)
            self.sleeper(min(30, remaining))

    def run(self, name, action):
        self.wait()
        self.announce(f'[{local_time(self.clock())}] 开始 {name}', flush=True)
        start = self.monotonic()
        result = action()
        duration = max(0, self.monotonic() - start)
        now = self.clock()
        should_rest = duration >= self.threshold
        self.state = {'last_stage': name, 'duration_seconds': duration,
                      'threshold_seconds': self.threshold, 'cooldown_seconds': self.seconds,
                      'rest_until': now + self.seconds if should_rest else 0}
        atomic_json(self.state, self.path)
        self.announce(f'[{local_time(now)}] 完成 {name}，运行 {duration:.1f} 秒；'
                      f'{"休息 " + str(self.seconds) + " 秒" if should_rest else "无需休息"}', flush=True)
        return result
