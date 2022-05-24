from src.backtest.parallel_backtest import run_parallel_backtest


def main(from_config):
    run_parallel_backtest(from_config=from_config, base_seed=12345)


if __name__ == '__main__':
    config_name = 'tuned_config_multi'
    main(config_name)
