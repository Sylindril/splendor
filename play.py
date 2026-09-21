"""Sit down at a Splendor table against the trained agents.

    python play.py                      # asks for the player count and your seat
    python play.py -p 4 -s 2            # 4 players, you are seat 2
    python play.py -p 2 --model experiments/latest.pt
    python play.py -p 2 --mcts 64       # opponents search before every move
    python play.py -p 4 --watch         # no human seat, just watch

The opponents are picked for the table size: the best shipped model for that
player count (models/splendor_<P>p_*.pt, the 2-player or the 4-player league
run), else the newest checkpoint under experiments/ trained for that many
players, else the greedy bot. `--latest` prefers your newest trained
checkpoint over the shipped one, and `--model PATH` uses exactly that file
(its player count is read from the checkpoint).
"""
import argparse
import glob
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)   # torch's pynvml notice

HERE = os.path.dirname(os.path.abspath(__file__))


def model_players(path):
    """Player count a checkpoint was trained for, from its input width."""
    import torch
    from splendor import layout as L
    sd = torch.load(path, map_location='cpu', weights_only=False)
    while isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    width = int(sd['encoder.0.weight'].shape[1])
    for p in (2, 3, 4):
        if L.obs_size(p) == width:
            return p
    raise ValueError(f'{path}: input width {width} matches no player count')


def shipped_model(num_players):
    """Best shipped model for this table size: the one trained the longest."""
    paths = glob.glob(os.path.join(HERE, 'models', f'splendor_{num_players}p_*.pt'))
    steps = lambda p: int((re.findall(r'_(\d+)M', os.path.basename(p)) or [0])[-1])
    return max(paths, key=steps) if paths else None


def pick_model(num_players, prefer_latest=False):
    """(spec for gui.py, human-readable reason)."""
    shipped = shipped_model(num_players)
    if shipped and not prefer_latest:
        return shipped, 'best shipped model'
    from splendor.agents import latest_checkpoint
    try:
        os.chdir(HERE)
        return latest_checkpoint(num_players), 'newest trained checkpoint'
    except FileNotFoundError:
        if shipped:
            return shipped, 'best shipped model'
        return 'greedy', f'no {num_players}-player checkpoint found, using the greedy bot'


def ask(prompt, default, valid):
    while True:
        raw = input(f'{prompt} [{default}]: ').strip().lower() or str(default)
        if raw in valid:
            return raw
        print(f'  choose one of: {", ".join(valid)}')


def free_port(start):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise SystemExit('no free port found')


def main():
    ap = argparse.ArgumentParser(description='Play Splendor against the trained agents')
    ap.add_argument('-p', '--players', type=int, choices=(2, 3, 4), default=None)
    ap.add_argument('-s', '--seat', type=int, default=None, help='your seat, 0 .. players-1')
    ap.add_argument('--watch', action='store_true', help='no human seat')
    ap.add_argument('--model', default=None, help='a checkpoint path instead of the default')
    ap.add_argument('--latest', action='store_true',
                    help='prefer the newest checkpoint under experiments/')
    ap.add_argument('--mcts', type=int, default=None, metavar='SIMS',
                    help='opponents run an MCTS of this many simulations')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, HERE)

    model, why = None, 'given with --model'
    if args.model:
        model = args.model if os.path.isfile(args.model) else os.path.join(HERE, args.model)
        if not os.path.isfile(model):
            raise SystemExit(f'no such checkpoint: {args.model}')
        trained = model_players(model)
        if args.players is None:
            args.players = trained
            print(f'{os.path.basename(model)} is a {trained}-player model.')
        elif args.players != trained:
            raise SystemExit(f'{args.model} was trained for {trained} players, '
                             f'not {args.players}. Drop -p or pass -p {trained}.')

    if args.players is None:
        args.players = int(ask('How many players (2, 3 or 4)?', 2, ('2', '3', '4')))
    if args.watch:
        seat = 'none'
    else:
        seats = [str(i) for i in range(args.players)]
        if args.seat is None:
            seat = ask(f'Your seat (0-{args.players - 1}; the first player is random)?', 0, seats)
        elif str(args.seat) in seats:
            seat = str(args.seat)
        else:
            raise SystemExit(f'--seat must be between 0 and {args.players - 1}')

    if model is None:
        model, why = pick_model(args.players, args.latest)
    shown = model if model == 'greedy' else os.path.relpath(model, HERE)
    print(f'Opponents: {shown} ({why})' + (f', MCTS {args.mcts} sims' if args.mcts else ''),
          flush=True)

    port = free_port(args.port)
    cmd = [sys.executable, os.path.join(HERE, 'gui.py'), '--num-players', str(args.players),
           '--human', seat, '--load-model-path', model, '--port', str(port)]
    if args.mcts:
        cmd += ['--mcts', str(args.mcts)]
    url = f'http://127.0.0.1:{port}'
    log = tempfile.TemporaryFile(mode='w+')        # the GUI's own chatter, shown only on failure
    gui = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # stop the GUI with us
    try:
        for _ in range(120):                       # the first import of torch is slow
            if gui.poll() is not None:
                log.seek(0)
                raise SystemExit('the GUI exited early:\n' + ''.join(log.readlines()[-15:]))
            try:
                urllib.request.urlopen(url + '/state', timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        print(f'Table ready: {url}   (Ctrl-C to stop)', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        gui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        gui.terminate()


if __name__ == '__main__':
    main()
