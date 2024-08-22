import torch
import argparse
import torch.nn  as nn
import torch.optim as optim
import sys
import json
import os
from torch.utils.data import Dataset

from torchvision import datasets
import torchvision.transforms as T

# import pickle
import dill as pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score, binary_f1_score
import importlib.util
import glob


'''
IMPORTANT NOTES about this training script
-we expect the forward pass to return logits
-we pass logits into the criterion
'''
def plot_combined_metrics(lossi, devlossi, f1i, devf1i, best_f1_dev, best_loss_dev):
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    # plt.subplot(1, 2, 1)
    # plt.plot(lossi, label='Train Loss')
    # plt.plot(devlossi, label='Validation Loss')
    # plt.title('Loss vs Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.text(0,-70,best_loss_dev)
    # plt.axhline(best_loss_dev) #horizontal line where best loss is 
    # print(best_loss_dev)
    # plt.legend()
    
    # # Plot F1 scores
    # plt.subplot(1, 2, 2)
    # plt.plot(f1i, label='Train F1')
    # plt.plot(devf1i, label='Validation F1')
    # plt.title('F1 Score vs Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('F1 Score')
    # plt.text(0,-70,best_f1_dev) #horizontal line where best f1 is 
    # plt.axhline(best_f1_dev)
    # plt.legend()
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


def load_data(data_path, batch_size):
    '''
    Load a dataset from a pickle file and return a DataLoader
    NO SUFFLE WE ASSUME THE pkl file data set was shuffeld at creation
    '''
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    if batch_size == -1:
        batch_size = len(dataset)
        print(f"The {(os.path.basename(data_path).split('.'))[0]} data set will be run FULL batch (size: {batch_size})")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
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

def main():
    parser = argparse.ArgumentParser(description="Basic train script")
    parser.add_argument("training_id", type=str, help="Training ID")
    parser.add_argument('--resume', '-r', action='store_true', help='if we are resuming training then path to model is and already init model')
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("data_path", type=str, help="Path to the dir with test and train data pkl files")
    parser.add_argument("--train_batch_size", "-trbs", type=int, required=True, help="Batch size for training (required), (-1) t do full batch")
    parser.add_argument("--test_batch_size", "-tebs",type=int, default=-1, help="Batch size for testing. If not present, uses full batch (-1)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    
    #group to handel the resume options
    resume_group = parser.add_mutually_exclusive_group(required=False)
    resume_group.add_argument('-full', action='store_true', help='resume from the very end of the last models training')
    resume_group.add_argument('-f1', action='store_true', help='Resume from best F1 option of the model')
    resume_group.add_argument('-loss', action='store_true', help='Resume from best Loss option of the model')

    args = parser.parse_args()

    if args.resume and not (args.full or args.f1 or args.loss):
        parser.error("--resume/-r requires at least one of -full, -f1, or -loss")

    



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model_path = os.path.abspath(args.model_path)
    data_path = os.path.abspath(args.data_path)
    
    print("Data Path: ",args.data_path)
    print("Train Batch size: ","FULL" if args.train_batch_size == -1 else args.train_batch_size)
    print("Test Batch size: ", "FULL" if args.test_batch_size == -1 else args.test_batch_size)
    print("Epochs: ",args.epochs)

    if args.resume:

        print("We are resuming training of model found at: ",model_path,'\n\n')
        model=torch.load(model_path) #TO BE UPDATED TO LOAD FROM STATE DIC AT A LATER DATE

        dir_path, file_name = os.path.split(model_path)
        model_name = (file_name.split('.')[0]).upper() 
        metrics = extract_metrics(dir_path, full=args.full, f1=args.f1, loss=args.loss)
    else:
        print("We are training a NEW model defined at: ",model_path,'\n\n')
        dir_path, file_name = os.path.split(model_path)
        module_name = (file_name.split('.')[0]) #all we doing here is taking someting like /usr/kuba/.../modeldef.py and returning modeldef
        model_name = (file_name.split('.')[0]).upper() #all we doing here is taking someting like /usr/kuba/.../modeldef.py and returning MODELDEF
        
        metrics = {
            'train_loss': [],
            'dev_loss': [],
            'train_f1': [],
            'dev_f1': [],
            'best_f1_dev': 0,
            'best_loss_dev': float('inf')
        }

        sys.path.append(dir_path)  #import the module
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)

        model = model_class()

    global TRAIN_ID
    TRAIN_ID = args.training_id
    global MODEL_NAME
    MODEL_NAME = model_name

    global SAVE_DIR
    SAVE_DIR = os.path.join('.', f'{TRAIN_ID}_{MODEL_NAME}')
    os.makedirs(SAVE_DIR, exist_ok=False)



#FIX THE BARCH SIZE ARGS
    train_loader = load_data(os.path.join(data_path,'train.pkl'), batch_size=args.train_batch_size)
    test_loader = load_data(os.path.join(data_path,'test.pkl'), batch_size=args.test_batch_size)


    ''' REMEBER we pass logits'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #Time to Train !!!!
    metrics = train(model, train_loader, test_loader, criterion, optimizer, device, args.epochs, metrics, metrics['best_f1_dev'], metrics['best_loss_dev'])

    #save model
    model_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_{TRAIN_ID}.pth')
    torch.save(model, model_path)
    metrics_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_{TRAIN_ID}_metrics.json')
    save_metrics(metrics, metrics_path)
    print(f"Training completed. Model saved as '{model_path}' and metrics saved as '{metrics_path}'.")

if __name__ == "__main__":
    main()