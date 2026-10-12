"""Run two or three native speech agents as one bounded batch experiment."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=90)
    args = parser.parse_args()
    agents = json.loads(Path(args.models).read_text())['agents']
    if len(agents) not in (2, 3):
        raise ValueError('Expected two or three model agents')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    processes, logs = [], []
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
    manifest = {'job_id': os.environ.get('SLURM_JOB_ID'), 'agents': agents,
                'gpu_assignment': {str(i): a['id'] for i, a in enumerate(agents)},
                'arena_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'stimulus': 'none; agents hear only the acoustic simulation'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        pending = set()
        for gpu, agent in enumerate(agents):
            backend = agent['backend']
            if backend not in ('personaplex', 'moshi'):
                raise ValueError(f'Unsupported native backend: {backend}')
            url = urlparse(agent['url'])
            if url.hostname != '127.0.0.1' or not url.port:
                raise ValueError('Workers must use an explicit loopback port')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', url.port))
            log = (root / f"{agent['id']}.log").open('w')
            logs.append(log)
            processes.append(subprocess.Popen(['bash', f'workers/launch_{backend}.sh', str(gpu), str(url.port)], stdout=log, stderr=subprocess.STDOUT))
            pending.add(url.port)
        deadline = time.monotonic() + 1200
        while pending:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('A model worker exited; inspect its log')
            for port in list(pending):
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        pending.remove(port)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'Model startup timed out: {pending}')
            time.sleep(1)
        subprocess.run([sys.executable, '-m', 'duplex_arena', 'evaluate', '--models', args.models,
                        '--configuration', args.configuration, '--seconds', str(args.seconds),
                        '--output', str(root)], check=True)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
