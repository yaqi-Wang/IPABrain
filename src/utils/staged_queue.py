"""Persistent stage completion and interruptible rest between GPU tasks."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def local_time(epoch):
    return datetime.fromtimestamp(epoch, timezone(timedelta(hours=8))).isoformat(timespec='seconds')


class StageQueue:
    def __init__(self, directory, seconds=600, clock=time.time, sleeper=time.sleep, announce=print):
        self.directory = Path(directory)
        self.path = self.directory / 'stage_ledger.json'
        self.state_path = self.directory / 'queue_state.json'
        self.seconds, self.clock, self.sleeper, self.announce = seconds, clock, sleeper, announce
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {'stages': {}, 'rest_until': 0}

    def state(self, **values):
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        state.update(values, updated_at=local_time(self.clock()))
        atomic_json(state, self.state_path)

    def wait(self):
        deadline = self.ledger.get('rest_until', 0)
        while self.clock() < deadline:
            remaining = max(0, deadline - self.clock())
            self.state(status='cooling', cooldown_remaining_seconds=round(remaining),
                       resume_at=local_time(deadline), last_completed_stage=self.ledger.get('last_stage'))
            self.announce(f'[休息] 剩余 {remaining:.0f} 秒；恢复时间 {local_time(deadline)}', flush=True)
            self.sleeper(min(30, remaining))

    def run(self, key, action, verify, *, rest=True):
        if key in self.ledger['stages']:
            if not verify():
                raise RuntimeError(f'Completed stage lost its verified artifacts: {key}')
            self.announce(f'[复用阶段] {key}', flush=True)
            return False
        self.wait()
        self.state(status='running', stage=key, cooldown_remaining_seconds=0, resume_at=None)
        self.announce(f'[{local_time(self.clock())}] 开始 {key}', flush=True)
        action()
        if not verify():
            raise RuntimeError(f'Stage did not produce complete verified artifacts: {key}')
        ended = self.clock()
        self.ledger['stages'][key] = {'completed_at': local_time(ended), 'completed_epoch': ended}
        self.ledger.update(last_stage=key, rest_until=ended + self.seconds if rest else 0)
        atomic_json(self.ledger, self.path)
        self.announce(f'[{local_time(ended)}] 完成 {key}', flush=True)
        self.state(status='stage_complete', last_completed_stage=key,
                   resume_at=local_time(self.ledger['rest_until']) if rest else None)
        return True
