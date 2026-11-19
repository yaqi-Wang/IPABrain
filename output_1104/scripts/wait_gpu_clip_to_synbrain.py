#!/usr/bin/env python3
"""Wait for the full multisubject queue and an idle GPU before CLIP-to-SynBrain."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'output_1104/scripts/train_clip_to_synbrain.py'
DEFAULT_OUTPUT = 'output_1104/clip_to_synbrain_conditional_v1'
DEFAULT_PREDECESSOR = ROOT / 'output_1104/multisubject_linear_imagery_v1'
EXPECTED_PROTOCOL = 'subj01_08_w010_mu_mse_linear_3samples_v1'
METRICS = ('PixCorr', 'SSIM', 'Alex2_2way', 'Alex5_2way', 'Incep_2way',
           'CLIP_2way', 'Eff_corr_distance', 'SwAV_corr_distance')
CHILD = None


def query_smi(arguments):
    result = subprocess.run(['nvidia-smi', *arguments], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=15, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or 'nvidia-smi failed').strip())
    return result.stdout.strip()


def gpu_status(index, max_used_mib, max_utilization):
    """Return observed status; unavailable or unparseable information is not idle."""
    info = query_smi([f'--id={index}', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'])
    fields = [field.strip() for field in info.split(',')]
    if len(fields) != 4 or not fields[0].isdigit() or int(fields[0]) != index:
        raise ValueError(f'Unexpected GPU information: {info!r}')
    uuid = fields[1]
    if not uuid.startswith('GPU-'):
        raise ValueError(f'Unexpected GPU UUID: {uuid!r}')
    used, utilization = int(fields[2]), int(fields[3])
    if used < 0 or not 0 <= utilization <= 100:
        raise ValueError(f'Invalid GPU readings: {info!r}')
    processes = query_smi(['--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                           '--format=csv,noheader,nounits'])
    active = []
    for line in processes.splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].startswith('GPU-'):
            raise ValueError(f'Unexpected GPU process information: {line!r}')
        if parts[1] == uuid:
            active.append({'pid': int(parts[0]), 'used_mib': parts[2]})
    idle = not active and used <= max_used_mib and utilization <= max_utilization
    return {'index': index, 'uuid': uuid, 'used_mib': used, 'utilization_percent': utilization,
            'compute_processes': active, 'idle': idle}


def observe(index, max_used_mib, max_utilization):
    try:
        return gpu_status(index, max_used_mib, max_utilization)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        return {'index': index, 'idle': False, 'error': str(error)}


def wait_for_idle(args, *, clock=time.monotonic, sleep=time.sleep, query=observe):
    while True:
        status = query(args.gpu_index, args.max_used_mib, args.max_utilization)
        print('[GPU 检查] ' + json.dumps(status, ensure_ascii=False), flush=True)
        if status['idle']:
            uuid, start = status['uuid'], clock()
            while True:
                elapsed = clock() - start
                if elapsed >= args.idle_confirm_seconds:
                    print(f'[启动] GPU {uuid} 连续确认空闲 {elapsed:.0f} 秒', flush=True)
                    return uuid
                sleep(min(args.confirm_poll_seconds, args.idle_confirm_seconds - elapsed))
                status = query(args.gpu_index, args.max_used_mib, args.max_utilization)
                print('[空闲确认] ' + json.dumps(status, ensure_ascii=False), flush=True)
                if not status['idle'] or status.get('uuid') != uuid:
                    break
        print(f'[等待] {args.poll_seconds} 秒后再次检查', flush=True)
        sleep(args.poll_seconds)


def predecessor_status(directory):
    """Fail closed: completion needs the expected full queue and complete summary."""
    directory = Path(directory)
    state_path = directory / 'queue_state.json'
    identity_path = directory / 'queue_identity.json'
    summary_path = directory / 'summary.json'
    result = {'directory': str(directory), 'ready': False}
    try:
        identity_bytes = identity_path.read_bytes()
        identity = json.loads(identity_bytes)
        result['identity_sha256'] = hashlib.sha256(identity_bytes).hexdigest()
        if (identity.get('protocol') != EXPECTED_PROTOCOL or identity.get('subjects') != list(range(1, 9))
                or set(identity.get('methods', [])) != {'w010_mse', 'mu_mse'}
                or set(identity.get('tasks', [])) != {'vision', 'imagery'} or identity.get('samples') != 3):
            result['reason'] = '前序队列身份或八人、两方法、两条件、三次采样的范围不匹配'
            return result
        state = json.loads(state_path.read_text(encoding='utf-8'))
        result.update(status=state.get('status'), stage=state.get('stage'),
                      last_completed_stage=state.get('last_completed_stage'))
        if state.get('status') != 'complete':
            result['reason'] = ('前序队列异常或中断；请检查其日志并续跑，等待器不会提前启动'
                                if state.get('status') in {'failed', 'interrupted'} else '前序队列尚未完整完成')
            return result
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary.get('status') != 'complete' or summary.get('subjects') != list(range(1, 9))
                or summary.get('samples') != 3 or not (directory / 'summary.csv').is_file()):
            result['reason'] = '前序结果汇总缺失或范围不完整'
            return result
        rows = summary.get('rows')
        expected = set(itertools.product(range(1, 9), ('w010_mse', 'mu_mse'),
                       ('vision', 'imagery'), ('initial', 'refined'),
                       ('all', 'natural', 'geometry'), ('sample00', 'sample01', 'sample02', 'mean')))
        if not isinstance(rows, list) or len(rows) != len(expected):
            result['reason'] = '前序汇总行数不完整'
            return result
        actual = set()
        for row in rows:
            key = tuple(row[k] for k in ('subject', 'method', 'task', 'phase', 'group', 'sample'))
            if key in actual or key not in expected or any(not isinstance(row[k], (int, float))
                    or isinstance(row[k], bool) or not math.isfinite(row[k]) for k in METRICS):
                result['reason'] = '前序汇总组合重复、范围错误或指标无效'
                return result
            actual.add(key)
        if actual != expected:
            result['reason'] = '前序汇总缺少受试者或采样组合'
            return result
        result.update(ready=True, reason='前序队列和完整评估汇总已完成', summary_rows=len(rows))
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        result['reason'] = f'前序状态暂不可验证：{error}'
        return result


def acquire_predecessor_lock(directory):
    """The predecessor's queue holds this lock throughout CPU, GPU and cooldown stages."""
    try:
        handle = (Path(directory) / '.mapping.lock').open('r')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            handle.close()
            return None
        return handle
    except OSError:
        return None


def wait_for_predecessor_and_gpu(args, *, sleep=time.sleep):
    """Return idle GPU UUID and held predecessor lock; caller releases after spawning."""
    expected_identity = None
    while True:
        status = predecessor_status(args.after_queue_dir)
        print('[前序队列检查] ' + json.dumps(status, ensure_ascii=False), flush=True)
        observed = status.get('identity_sha256')
        if observed and expected_identity is None:
            expected_identity = observed
        if observed and expected_identity != observed:
            print('[等待] 前序队列身份发生变化，拒绝自动启动；请核查并重新启动等待器', flush=True)
        elif status['ready']:
            lock = acquire_predecessor_lock(args.after_queue_dir)
            if lock is None:
                print('[等待] 前序队列完成标记已出现，但队列进程尚未释放锁', flush=True)
            else:
                try:
                    again = predecessor_status(args.after_queue_dir)
                finally:
                    lock.close()
                if again['ready'] and again.get('identity_sha256') == expected_identity:
                    print('[前序完成] 已核验全部结果并确认前序队列锁已释放，开始确认 GPU 空闲', flush=True)
                    uuid = wait_for_idle(args)
                    # Reacquire immediately before starting the child: a resumed
                    # predecessor must not slip into its CPU/cooldown phase here.
                    lock = acquire_predecessor_lock(args.after_queue_dir)
                    if lock is not None:
                        try:
                            final = predecessor_status(args.after_queue_dir)
                            if final['ready'] and final.get('identity_sha256') == expected_identity:
                                return uuid, lock
                        except BaseException:
                            lock.close()
                            raise
                        lock.close()
                print('[等待] 前序状态或队列锁发生变化，重新检查', flush=True)
        print(f'[等待前序队列] {args.poll_seconds} 秒后再次检查', flush=True)
        sleep(args.poll_seconds)


def forward_signal(signum, _frame):
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signum)
    else:
        raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--gpu-index', type=int, default=0, help='Physical nvidia-smi GPU index')
    parser.add_argument('--poll-seconds', type=int, default=1200)
    parser.add_argument('--idle-confirm-seconds', type=int, default=180)
    parser.add_argument('--confirm-poll-seconds', type=int, default=10)
    parser.add_argument('--max-used-mib', type=int, default=1024)
    parser.add_argument('--max-utilization', type=int, default=5)
    parser.add_argument('--after-queue-dir', type=Path, default=DEFAULT_PREDECESSOR,
                        help='Predecessor eight-subject linear queue directory; completion is mandatory')
    parser.add_argument('--check-once', action='store_true', help='Read-only predecessor and GPU status check')
    args, extra = parser.parse_known_args()
    if extra and extra[0] == '--': extra = extra[1:]
    if any(x.split('=', 1)[0] in {'--device','--subject','--output','--preflight'} for x in extra):
        parser.error('Use wrapper --subject / --output; --device and --preflight cannot be forwarded')
    if (args.gpu_index < 0 or args.poll_seconds <= 0 or args.idle_confirm_seconds <= 0
        or args.confirm_poll_seconds <= 0 or args.max_used_mib < 0 or not 0 <= args.max_utilization <= 100):
        parser.error('Invalid GPU waiting settings')
    if args.check_once:
        predecessor = predecessor_status(args.after_queue_dir)
        gpu = observe(args.gpu_index, args.max_used_mib, args.max_utilization)
        lock = acquire_predecessor_lock(args.after_queue_dir)
        predecessor['lock_available'] = lock is not None
        if lock is not None: lock.close()
        print(json.dumps({'predecessor': predecessor, 'gpu': gpu}, indent=2, ensure_ascii=False), flush=True)
        return 0 if 'error' not in gpu else 1
    root = Path(args.output)
    root = (root if root.is_absolute() else ROOT / root) / f'subj{args.subject:02d}'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.gpu_wait.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'已有相同输出目录的等待队列：{root}') from error
        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)
        uuid, predecessor_lock = wait_for_predecessor_and_gpu(args)
        try:
            command = [sys.executable, '-u', str(WORKER), '--subject', str(args.subject),
                       '--output', args.output, '--device', 'cuda:0', *extra]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = uuid
            print('[执行] ' + json.dumps(command, ensure_ascii=False), flush=True)
            print('[执行] CUDA_VISIBLE_DEVICES=' + uuid, flush=True)
            global CHILD
            CHILD = subprocess.Popen(command, cwd=ROOT, env=env)
        finally:
            predecessor_lock.close()
        code = CHILD.wait()
        CHILD = None
        print(f'[结束] 训练队列退出码={code}', flush=True)
        return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
