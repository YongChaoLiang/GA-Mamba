import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from train_model import GA_Mamba

def adjust_learning_rate(optimizer, initial_lr, epoch, max_epochs, power=0.9):
    lr = initial_lr * (1 - float(epoch) / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('train_data.csv')
#train_df = pd.read_csv('processed_GN.csv')
#train_df = pd.read_csv('processed_SMOTE.csv')
#train_df = pd.read_csv('processed_mixup3.csv')

train_X = train_df.drop(['Dmax', 'index'], axis=1).values
train_y = train_df['Dmax'].values

test_df = pd.read_csv('test_data.csv')
test_X = test_df.drop(['Dmax', 'index'], axis=1).values
test_y = test_df['Dmax'].values

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)
test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32).view(-1, 1).to(device)


path= "mamba_Mixup3_0.8722.pth"
#path='mamba_SMOTE_0.8621.pth'
#path='mamba_GN_0.8619.pth'
#path='mamba.pth'
input_dim = test_X_scaled.shape[1]
model = GA_Mamba(128,45,device).to(device)
model.load_state_dict(torch.load(path))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.eval()
with torch.no_grad():
    test_predictions = model(test_X_tensor)
    test_loss = criterion(test_predictions, test_y_tensor)
    test_predictions = test_predictions.cpu()
    test_r2 = r2_score(test_y, test_predictions.numpy())
    test_mse = mean_squared_error(test_y, test_predictions.numpy())
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_y, test_predictions.numpy())
    test_rmae=np.sqrt(test_mae)
test_predictions = test_predictions.cpu().numpy() if test_predictions.is_cuda else test_predictions.numpy()
test_y = test_y.cpu().numpy() if isinstance(test_y, torch.Tensor) else test_y

plt.figure(figsize=(10, 8))
plt.scatter(test_y, test_predictions, alpha=0.5, color='blue', label='Predictions vs Actual')

max_value = max(np.max(test_y), np.max(test_predictions))
min_value = min(np.min(test_y), np.min(test_predictions))
plt.plot([min_value, max_value], [min_value, max_value], 'r--', lw=2, label='Perfect Prediction')

plt.title('Actual vs Predicted Values', fontsize=16)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

print("\nTest Metrics:")
print(f"Loss: {test_loss.item():.4f}")
print(f"R² Score: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"RMAE: {test_rmae:.4f}")
