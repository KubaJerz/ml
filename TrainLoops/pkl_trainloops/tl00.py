import torch
import torch.nn  as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score 

'''
IMPORTANT NOTES about this training script
-we expect the forward pass to return logits
-we pass logits into the criterion
'''
def plot_combined_metrics(lossi, devlossi, f1i, devf1i):
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(lossi, label='Train Loss')
    plt.plot(devlossi, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(f1i, label='Train F1')
    plt.plot(devf1i, label='Validation F1')
    plt.title('F1 Score vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_{TRAIN_ID}_metrics.png")
    plt.close()


def load_data(data_path, batch_size):
    '''
    This code can/will be variable from application to application
    '''
    dataset = YourDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    lossi = [] #train_losses
    devlossi = [] #val_losses
    f1i = []  #train_f1_scores
    devf1i = [] #val_f1_scores

    model = model.to(device)
    
    model.train()
    for epoch in tqdm(range(epochs),desc=f'Progress: '):
        epoch_loss = 0
        epoch_f1 = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model.forward(X_batch)

            loss = criterion(logits, y_batch)
            f1 = multiclass_f1_score(logits, y_batch, num_classes=model.num_classes).item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_f1 += f1


        lossi.append(epoch_loss / len(train_loader))  # append avg loss per epoch
        f1i.append(epoch_f1 / len(train_loader))  # append avg f1 per epoch


        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_f1 = 0
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                devlogits = model.forward(X_batch)
                dev_loss = criterion(devlogits, y_batch).item()
                dev_f1 = multiclass_f1_score(devlogits, y_batch, num_classes=model.num_classes).item() 
                test_epoch_loss += dev_loss
                test_epoch_f1 += dev_f1

            devlossi.append(test_epoch_loss / len(test_loader))
            devf1i.append(test_epoch_f1 / len(test_loader))

        if (epoch+1) % 2 == 0:
            plot_combined_metrics(lossi, devlossi, f1i, devf1i)





def main():
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("WRONG Usage: python script.py <model_path> <data_path> <batch_size> <training_id> <epochs>(OPTIONAL DEF=50)")
        sys.exit(1)
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    TRAIN_ID = sys.argv[4]
    if len(sys.argv) == 6:
        epochs = int(sys.argv[5])
    else:
        epochs = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model_path = os.path.abspath(model_path)
    data_path = os.path.abspath(data_path)
    print(data_path)
    print(batch_size)
    print(epochs)
    
    model=torch.load(model_path)
    train_loader = load_data(data_path, batch_size)
    test_loader = load_data(data_path, batch_size)


    ''' REMEBER we pass logits'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #Time to Train !!!!
    train(model, train_loader, test_loader, criterion, optimizer, device, epochs)

    #save model
    MODEL_NAME = "mr_trained"
    torch.save(model.state_dict(), f'{MODEL_NAME}_{TRAIN_ID}.pth')
    print(f"Training completed. Model saved as '{MODEL_NAME}_{TRAIN_ID}.pth'")







if __name__ == "__main__":
    main()