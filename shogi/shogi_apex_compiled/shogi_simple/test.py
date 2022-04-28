# from shogi_pettingzoo_env import get_env
# from pettingzoo.test import api_test, seed_test

# environ = get_env()

# environ.reset()

# api_test(environ, num_cycles=2000, verbose_progress=False)
# #seed_test(get_env, num_cycles=2000, test_kept_state=True)


import pprint

from shogi_ts import get_args, train_agent, watch

def test_tic_tac_toe(args=get_args()):
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    assert result["best_reward"] >= args.win_rate

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent)


if __name__ == '__main__':
    test_tic_tac_toe(get_args())
