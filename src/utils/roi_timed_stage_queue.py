"""ROI stage ledger with a cooldown based on each completed stage's runtime."""
import time

from src.utils.staged_queue import StageQueue, atomic_json, local_time


class ROITimedStageQueue(StageQueue):
    def __init__(self, directory, seconds=180, clock=time.time, sleeper=time.sleep,
                 announce=print, monotonic=time.monotonic, threshold=3600):
        # Accept the legacy queue's positional 120 argument without keeping its old policy.
        super().__init__(directory, seconds=180, clock=clock, sleeper=sleeper, announce=announce)
        self.monotonic = monotonic
        self.threshold = threshold

    def run(self, key, action, verify, *, rest=True):
        if key in self.ledger['stages']:
            if not verify():
                raise RuntimeError(f'Completed stage lost its verified artifacts: {key}')
            self.announce(f'[复用阶段] {key}', flush=True)
            return False
        self.wait()
        self.state(status='running', stage=key, cooldown_remaining_seconds=0, resume_at=None)
        self.announce(f'[{local_time(self.clock())}] 开始 {key}', flush=True)
        start = self.monotonic()
        action()
        duration = max(0, self.monotonic() - start)
        if not verify():
            raise RuntimeError(f'Stage did not produce complete verified artifacts: {key}')
        ended = self.clock()
        should_rest = rest and duration >= self.threshold
        self.ledger['stages'][key] = {'completed_at': local_time(ended),
                                      'completed_epoch': ended,
                                      'duration_seconds': duration}
        self.ledger.update(last_stage=key, rest_until=ended + self.seconds if should_rest else 0)
        atomic_json(self.ledger, self.path)
        self.announce(f'[{local_time(ended)}] 完成 {key}，运行 {duration:.1f} 秒；'
                      f'{"休息 180 秒" if should_rest else "无需休息"}', flush=True)
        self.state(status='stage_complete', last_completed_stage=key,
                   resume_at=local_time(self.ledger['rest_until']) if should_rest else None)
        return True
