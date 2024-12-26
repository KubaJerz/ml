import argparse
from .src.ExperimentRunner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    runner = ExperimentRunner(args.config_path)
    runner.validate_core_config_structure()
    runner.run()

if __name__ == '__main__':
    main()