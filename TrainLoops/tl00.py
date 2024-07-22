import torch
import sys
import os

def load_data(data_path, batch_size):
    


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("WRONG Usage: python script.py <model_path> <data_path> <batch_size> <epochs>(OPTIONAL DEF=50)")
        sys.exit(1)
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    if len(sys.argv) == 5:
        epochs = int(sys.argv[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model_path = os.path.abspath(model_path)
    data_path = os.path.abspath(data_path)
    print(data_path)
    print(batch_size)
    print(epochs)
    
    model=torch.load(model_path)
    loader = load_data(data_path, batch_size)








if __name__ == "__main__":
    main()