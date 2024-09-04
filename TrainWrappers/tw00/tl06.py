import torch
import argparse
import torch.nn  as nn
import torch.optim as optim
import sys
import json
import os


from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score
import importlib.util
import glob

from tensor_builder import getDataSet


'''
IMPORTANT NOTES about this training script
-we expect the forward pass to return logits
-we pass logits into the criterion
'''
def plot_combined_metrics(lossi, devlossi, f1i, devf1i, best_f1_dev, best_loss_dev):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(lossi, label='Train Loss')
    plt.plot(devlossi, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axhline(best_loss_dev, color='g', linestyle='--', label=f'Best Dev Loss: {best_loss_dev:.3f}')
    #plt.text(0, best_loss_dev - 0.02, f'Best Loss: {best_loss_dev:.4f}', color='g', fontsize=10)
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(f1i, label='Train F1')
    plt.plot(devf1i, label='Validation F1')
    plt.title('F1 Score vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.axhline(best_f1_dev, color='g', linestyle='--', label=f'Best Dev F1: {best_f1_dev:.3f}')
    #plt.text(0, best_f1_dev - 0.02, f'Best F1: {best_f1_dev:.4f}', color='g', fontsize=10)
    plt.legend()

    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,f"{MODEL_NAME}_{TRAIN_ID}_metrics.png"))
    plt.close()


def save_metrics(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs, metrics, best_dev_f1, best_dev_loss):

    model = model.to(device)
    best_dev_f1 = best_dev_f1 # we will use this to save the model if its better than the best
    best_dev_loss = best_dev_loss # we will use this to save the model if its better than the best
    
    model.train()
    for epoch in tqdm(range(epochs),desc=f'Progress: '):
        epoch_loss = 0
        epoch_f1 = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model.forward(X_batch)

            loss = criterion(logits, y_batch)
            f1 = multiclass_f1_score(logits, torch.argmax(y_batch, dim=1), num_classes=model.num_classes).item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_f1 += f1


        metrics['train_loss'].append(epoch_loss / len(train_loader))
        metrics['train_f1'].append(epoch_f1 / len(train_loader))  # append avg f1 per epoch


        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_f1 = 0
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                devlogits = model.forward(X_batch)
                dev_loss = criterion(devlogits, y_batch).item()
                dev_f1 = multiclass_f1_score(devlogits, torch.argmax(y_batch, dim=1), num_classes=model.num_classes).item() 
                test_epoch_loss += dev_loss
                test_epoch_f1 += dev_f1

            metrics['dev_loss'].append(test_epoch_loss / len(test_loader))
            metrics['dev_f1'].append(test_epoch_f1 / len(test_loader))



            #check if the model has best F1 or Best Loss
            is_loss_best = False
            is_f1_best = False
            if metrics['dev_loss'][-1] < best_dev_loss:
                is_loss_best = True
                best_dev_loss = metrics['dev_loss'][-1]
                metrics['best_loss_dev'] = best_dev_loss #save to metrics for resuming and loading model

            if metrics['dev_f1'][-1] > best_dev_f1:
                is_f1_best = True
                best_dev_f1 = metrics['dev_f1'][-1]
                metrics['best_f1_dev'] = best_dev_f1

            if is_loss_best: #if we found good loss then save
                model_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_{TRAIN_ID}_bestLoss.pth')
                metrics_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_{TRAIN_ID}_bestLoss_metrics.json')

                if os.path.exists(model_path): #remove the old best model if it exists
                    os.remove(model_path)
                    os.remove(metrics_path)
                torch.save(model, model_path)
                save_metrics(metrics, metrics_path)
                print(f'New best Loss: {best_dev_loss} at Epoch:{epoch} Model saved!')
            
            
            if is_f1_best: #if we found good f1 then save
                model_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_{TRAIN_ID}_bestF1.pth')
                metrics_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_{TRAIN_ID}_bestF1_metrics.json')
                
                if os.path.exists(model_path):#remove the old best model if it exists
                    os.remove(model_path)
                    os.remove(metrics_path)
                torch.save(model, model_path)
                save_metrics(metrics, metrics_path)
                print(f'New best F1: {best_dev_f1} at Epoch:{epoch} Model saved!')


        if (epoch+1) % 2 == 0:
            plot_combined_metrics(metrics['train_loss'], metrics['dev_loss'], 
                                  metrics['train_f1'], metrics['dev_f1'], metrics['best_f1_dev'], 
                                  metrics['best_loss_dev'])
            
    return metrics
 
def extract_metrics(directory,full, f1, loss):
    metrics_data = []

    if full:
        # Search for files matching the pattern *_metrics.json 
        file_pattern = os.path.join(directory, '*_metrics.json')
        files = glob.glob(file_pattern)
    if f1: #we are resuming from the best f1 score of the model
        file_pattern = os.path.join(directory, '*_bestF1_metrics.json')
        files = glob.glob(file_pattern)
    if loss: #we are resuming from the best loss score of the model
        file_pattern = os.path.join(directory, '*_bestLoss_metrics.json')
        files = glob.glob(file_pattern)


    if not files:
        print(f"No previous metrics files found in {directory}")
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


def run_training(
    training_id,
    model_path,
    sub_dir,
    train_percent=0.7,
    random_state=69,
    train_batch_size=32,
    test_batch_size=-1,
    epochs=15,
    resume=False,
    full=False,
    f1=False,
    loss=False,
):
    global MODEL_NAME
    global SAVE_DIR
    global TRAIN_ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    print(f"Train percent: {train_percent}")
    print(f"Random state: {random_state}")
    print("Train Batch size: ","FULL" if train_batch_size == -1 else train_batch_size)
    print("Test Batch size: ", "FULL" if test_batch_size == -1 else test_batch_size)
    print("Epochs: ", epochs)

    if isinstance(model_path, (str)): #this means that is it 
        model_path = os.path.abspath(model_path)
        if resume:
            print(f"We are resuming training of model found at: {model_path}\n\n")
            model = torch.load(model_path)
            dir_path, file_name = os.path.split(model_path)
            model_name = (file_name.split('.')[0]).upper() 
            metrics = extract_metrics(dir_path, full=full, f1=f1, loss=loss)
        else:
            print(f"We are training a NEW model defined at: {model_path}\n\n")
            dir_path, file_name = os.path.split(model_path)
            module_name = (file_name.split('.')[0])
            model_name = (file_name.split('.')[0]).upper()
            
            metrics = {
                'train_loss': [],
                'dev_loss': [],
                'train_f1': [],
                'dev_f1': [],
                'best_f1_dev': 0,
                'best_loss_dev': float('inf')
            }

            sys.path.append(dir_path)
            module = importlib.import_module(module_name)
            model_class = getattr(module, model_name)

            model = model_class()

        MODEL_NAME = model_name
        SAVE_DIR = os.path.join('.', f'{training_id}_{model_name}') #we save the model diffrently if we are doing comman line sinlge model
    
    else:
        metrics = {
                'train_loss': [],
                'dev_loss': [],
                'train_f1': [],
                'dev_f1': [],
                'best_f1_dev': 0,
                'best_loss_dev': float('inf')
            }

        model = model_path
        SAVE_DIR = sub_dir

        MODEL_NAME = 'no_name'


    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dataset, test_dataset = getDataSet(randomState=random_state, trainPercent=train_percent)

    if train_batch_size == -1:
        train_batch_size = len(train_dataset)
    if test_batch_size == -1:
        test_batch_size = len(test_dataset)

    TRAIN_ID = training_id

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    metrics = train(model, train_loader, test_loader, criterion, optimizer, device, epochs, metrics, metrics['best_f1_dev'], metrics['best_loss_dev'])

    model_path = os.path.join(SAVE_DIR, f'{model_name}_{training_id}.pth')
    torch.save(model, model_path)
    metrics_path = os.path.join(SAVE_DIR, f'{model_name}_{training_id}_metrics.json')
    save_metrics(metrics, metrics_path)
    print(f"Training completed. Model saved as '{model_path}' and metrics saved as '{metrics_path}'.")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Basic train script")
    parser.add_argument("training_id", type=str, help="Training ID")
    parser.add_argument('--resume', '-r', action='store_true', help='if we are resuming training then path to model is and already init model')
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("--train_percent", type=float, default=0.7, help="Percentage of data to use for training (default: 0.7)")
    parser.add_argument("--random_state", type=int, default=69, help="Random state for dataset split (default: 69)")
    parser.add_argument("--train_batch_size", "-trbs", type=int, required=True, help="Batch size for training (required), (-1) t do full batch")
    parser.add_argument("--test_batch_size", "-tebs",type=int, default=-1, help="Batch size for testing. If not present, uses full batch (-1)")    
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    
    # Group to handle the resume options
    resume_group = parser.add_mutually_exclusive_group(required=False)
    resume_group.add_argument('-full', action='store_true', help='resume from the very end of the last models training')
    resume_group.add_argument('-f1', action='store_true', help='Resume from best F1 option of the model')
    resume_group.add_argument('-loss', action='store_true', help='Resume from best Loss option of the model')

    args = parser.parse_args()

    if args.resume and not (args.full or args.f1 or args.loss):
        parser.error("--resume/-r requires at least one of -full, -f1, or -loss")

    run_training(
        training_id=args.training_id,
        model_path=args.model_path,
        train_percent=args.train_percent,
        random_state=args.random_state,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        resume=args.resume,
        full=args.full,
        f1=args.f1,
        loss=args.loss
    )

if __name__ == "__main__":
    main()