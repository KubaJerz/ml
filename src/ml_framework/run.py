import argparse
from .ExperimentRunner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    runner = ExperimentRunner(args.config_path)
    runner.run()

if __name__ == '__main__':
    main()