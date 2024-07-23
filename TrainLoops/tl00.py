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
def plot_combined_metrics(lossi, devlossi, f1i, devf1i, epoch):
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
    plt.savefig(f"training_metrics_epoch_{epoch}.png")
    plt.close()


def load_data(data_path, batch_size):
    '''
    This code can/will be variable from application to application
    '''
    dataset = YourDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    lossi = [] #train_losses
    devlossi = [] #val_losses
    f1i = []  #train_f1_scores
    devf1i = [] #val_f1_scores

    model = model.to(device)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_f1 = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch: {epoch+1}/{epochs}'):
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
            devlogits = model.forward(X_test)
            devloss = criterion(devlogits, y_test).item()
            devf1 = multiclass_f1_score(devlogits, y_test, num_classes=model.num_classes).item()
            devlossi.append(devloss)
            devf1i.append(devf1)


    
    if (epoch+1) % 2 == 0:
        plot_combined_metrics(lossi, devlossi, f1i, devf1i, epoch+1)





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
    train_loader = load_data(data_path, batch_size)


    ''' REMEBER we pass logits'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #Time to Train !!!!
    train(model, train_loader, criterion, optimizer, device, epochs)

    #save model
    model_name = "mr_trained"
    torch.save(model.state_dict(), f'{model_name}.pth')
    print(f"Training completed. Model saved as '{trained_model}.pth'")






if __name__ == "__main__":
    main()