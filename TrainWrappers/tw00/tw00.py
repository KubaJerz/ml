from tqdm import tqdm
import random

def def_model(hyperparams):
    


def main():
    parser = argparse.ArgumentParser(description="train wrapper")
    parser.add_argument("id", type=str, help="ID")
    parser.add_argument("--num_models", type=int, default=5, help="Number of models to train (default: 5)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")


    args = parser.parse_args()
    id = args.id
    num_models = args.num_models
    epochs = args.epochs

    hyper_ranges = {
        'train_batch_size': [32,64,128,256,512],
        'learning_rate': (0.0001, 0.1),
        'hidden_blocks': 4,
        'layer_depth': [[4, 8, 16, 32],[2,4,8,16]],
        'dropout_rate': (0.1, 0.5),
        'activation': ['relu'], #['relu', 'tanh', 'sigmoid'],
        'normalization': ['batch', 'layer']#['none', 'batch', 'layer']
    }
    test_batch_size = -1

    for i in tqdm(range(num_models)):
        #random pick
        hyperparams = {
            'id': f"{id}_{i}",
            'train_batch_size': random.choice(hyper_ranges['train_batch_size']),
            'test_batch_size': test_batch_size,
            'epochs': epochs,
            'learning_rate': random.uniform(*hyper_ranges['learning_rate']),
            'hidden_blocks': hyper_ranges['hidden_blocks'],
            'neurons_per_layer': random.choice(hyper_ranges['layer_depth']),
            'dropout_rate': random.uniform(*hyper_ranges['dropout_rate']),
            'activation': random.choice(hyper_ranges['activation']),
            'normalization': random.choice(hyper_ranges['normalization'])
        }


        def_model(hyperparams)
        train_model(hyperparams['id'])

if __name__ == "__main__":
    main()