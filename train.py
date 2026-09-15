"""Train (or evaluate) a Splendor policy.

    python train.py [--train.device mps] [--env.num_players 2] ...
    python train.py --config league [--league.save-every 30] ...
    python train.py eval --load-model-path experiments/xxx.pt

`--config NAME` picks `config/NAME.ini` (on top of pufferlib's default.ini);
`[base] env_class` selects the env (Splendor or the self-play League).
"""
import argparse
import ast
import configparser
import os
import shutil
import sys
import time
from collections import defaultdict

import torch

import pufferlib
import pufferlib.pufferl
import pufferlib.vector

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_INI = os.path.join(
    os.path.dirname(os.path.realpath(pufferlib.__file__)), 'config/default.ini')


def config_files(name):
    """pufferlib's default.ini plus config/NAME.ini."""
    path = os.path.join(HERE, 'config', f'{name}.ini')
    if not os.path.exists(path):
        raise SystemExit(f'No such config: {path}')
    return [DEFAULT_INI, path]


def load_config():
    """pufferlib.pufferl.load_config, but reading our own ini files."""
    # --config has to be resolved before the rest of the parser can be built,
    # since the ini files decide which --section.key flags exist.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default='splendor')
    config_name = pre.parse_known_args()[0].config

    parser = argparse.ArgumentParser(description='Splendor training',
        add_help=False)
    parser.add_argument('--config', type=str, default=config_name,
        help='config/NAME.ini to read (default: splendor)')
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint ("latest" for the newest)')
    parser.add_argument('--load-id', type=str, default=None)
    parser.add_argument('--render-mode', type=str, default='auto')
    for flag in ('--wandb', '--neptune'):
        parser.add_argument(flag, action='store_true')
    for flag, default in (('--save-frames', 0), ('--fps', 15), ('--max-runs', 200),
            ('--local-rank', 0), ('--gif-path', 'eval.gif'), ('--tag', None),
            ('--wandb-project', 'pufferlib'), ('--wandb-group', 'debug'),
            ('--neptune-name', 'pufferai'), ('--neptune-project', 'ablations')):
        parser.add_argument(flag, type=type(default) if default is not None else str,
            default=default)
    parser.add_argument('--games', type=int, default=1000, help='eval games')
    parser.add_argument('--render', action='store_true', help='eval rendering')

    p = configparser.ConfigParser()
    p.read(config_files(config_name))

    def auto_type(value):
        # As in pufferl.load_config, plus a string fallback so that
        # --train.device mps works (device defaults to 'auto' here)
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    for section in p.sections():
        for key in p[section]:
            try:
                value = ast.literal_eval(p[section][key])
            except Exception:
                value = p[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            if value == 'auto':
                arg_type = auto_type
            else:
                # str for None defaults (e.g. rnn_name) so they stay overridable
                arg_type = str if value is None else type(value)
            parser.add_argument(fmt.replace('_', '-'), default=value, type=arg_type)

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    if args['rnn_name'] == 'None':
        args['rnn_name'] = None
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args


def resolve_device(device):
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def save_state_dict(policy, path):
    """Atomically write a CPU state dict that agents.load_policy can read."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    state = {k: v.detach().to('cpu')
             for k, v in policy.state_dict().items()}
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def league_train(args, vecenv, policy, env_name='puffer_splendor'):
    """pufferlib.pufferl.train's loop, plus the two saves the league needs.

    Every `[league] save_every` seconds the current policy is written
    atomically to `[env] latest_path` (the workers notice the new mtime and
    reload it as the "latest" opponent), and every `[league] pool_every`
    global steps a snapshot is frozen into `[env] pool_dir`.
    """
    latest_path = args['env'].get('latest_path', 'experiments/latest.pt')
    pool_dir = args['env'].get('pool_dir', 'experiments/pool')
    save_every = float(args['league'].get('save_every', 30))
    pool_every = int(args['league'].get('pool_every', 20_000_000))
    os.makedirs(pool_dir, exist_ok=True)
    # Written before the workers reset, so they start with a real opponent.
    save_state_dict(policy, latest_path)

    train_config = dict(**args['train'], env=env_name)
    total = train_config['total_timesteps']
    pufferl = pufferlib.pufferl.PuffeRL(train_config, vecenv, policy)
    last_save, next_pool = time.time(), 0

    all_logs = []
    while pufferl.global_step < total:
        step = pufferl.global_step
        if step >= next_pool:
            save_state_dict(pufferl.uncompiled_policy, latest_path)
            last_save = time.time()
            shutil.copyfile(latest_path,
                            os.path.join(pool_dir, f'step_{step:011d}.pt'))
            next_pool = step + pool_every
        elif time.time() - last_save >= save_every:
            save_state_dict(pufferl.uncompiled_policy, latest_path)
            last_save = time.time()

        pufferl.evaluate()
        logs = pufferl.train()
        if logs is not None and step > 0.20 * total:
            all_logs.append(logs)

    save_state_dict(pufferl.uncompiled_policy, latest_path)

    i, stats = 0, {}
    while i < 32 or not stats:
        stats = pufferl.evaluate()
        i += 1

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs


def main():
    mode = 'train'
    if len(sys.argv) > 1 and sys.argv[1] in ('train', 'eval'):
        mode = sys.argv.pop(1)

    args = load_config()
    device = resolve_device(args['train']['device'])
    args['train']['device'] = device

    if mode == 'eval':
        from evaluate import evaluate
        evaluate(args['load_model_path'], games=args['games'],
                 render=args['render'], device=device,
                 policy_name=args['policy_name'], rnn_name=args['rnn_name'],
                 policy_kwargs=args['policy'], rnn_kwargs=args['rnn'],
                 env_kwargs=args['env'])
        return

    import splendor.policy

    env_class = args.get('env_class', 'Splendor')
    if env_class == 'League':
        from splendor.league import League as env_cls
    elif env_class == 'Splendor':
        from splendor.splendor import Splendor as env_cls
    else:
        raise SystemExit(f'Unknown [base] env_class: {env_class}')

    vecenv = pufferlib.vector.make(env_cls, env_kwargs=args['env'], **args['vec'])
    policy = getattr(splendor.policy, args['policy_name'])(
        vecenv.driver_env, **args['policy'])
    if args['rnn_name'] is not None:
        policy = getattr(splendor.policy, args['rnn_name'])(
            vecenv.driver_env, policy, **args['rnn'])
    policy = policy.to(device)

    if args['load_model_path'] is not None:
        state = torch.load(args['load_model_path'], map_location=device)
        policy.load_state_dict({k.replace('module.', ''): v
                                for k, v in state.items()})

    if env_class == 'League':
        league_train(args, vecenv, policy)
    else:
        pufferlib.pufferl.train('puffer_splendor', args=args,
                                vecenv=vecenv, policy=policy)


if __name__ == '__main__':
    main()
