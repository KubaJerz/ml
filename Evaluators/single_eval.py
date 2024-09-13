import argparse
import matplotlib as plt
import os
import glob
import json


def extract_metrics(sudir):
    metrics_data = []
    # Search for files matching the pattern *_FULL_metrics.json 
    file_pattern = os.path.join(sudir, '*_Full_metrics.json')
    files = glob.glob(file_pattern)

    if not files:
        print(f"No previous metrics files found in {sudir}")
        return {}
    
    file_path = files[0]

    try:
        with open(file_path, 'r') as file:
            metrics = json.load(file)
        return metrics
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {}


def single_plot(data):
    #make a plot and add stuff
    return plot

def main(eval_dir_path):
    metrics = extract_metrics(subdir)
    plot = single_plot(data=metrics)


    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir_path,f"_fullEval.png"))
    plt.close()


def arg_pars():
    parser = argparse.ArgumentParser(description="single model evaluator")
    parser.add_argument("eval_dir_path", type=str, help="Path to the dir with model")

    args = parser.parse_args()

    main(eval_dir_path=args.eval_dir_path)

if __name__ == "__main__":
    arg_pars()