'''
GA-Mamba model construction and training
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
torch.autograd.set_detect_anomaly(True)
from mamba_ssm import Mamba as OfficialMamba  

class GA_Mamba(nn.Module):
    def __init__(self,d_model, state_size, device):
        super(Mamba, self).__init__()
        self.a=torch.tensor([3, 4, 5, 6, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 39, 40, 41, 42, 46, 47, 49, 50, 57, 58, 59, 60, 62, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 78, 79]).to(device)
        self.embeding=nn.Embedding(80,d_model).to(device)
        self.feature_expander = FeatureExpander(d_model).to(device)
        self.attention = AttentionModel(45).to(device)

        self.mamba = OfficialMamba(
            d_model=d_model,       
            d_state=state_size,    
            d_conv=4,             
            expand=3,             
            device=device
        )

        self.norm = nn.LayerNorm(d_model).to(device)
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        self.fc = DenseNet(45).to(device)
    def forward(self, x):
        y=self.embeding(self.a)
        y=y.unsqueeze(0)
        _, weight = self.attention(x)
        x = self.feature_expander(x * weight)  # [B, 45, d_model]
        x = x * y
        x = self.mamba(x)+x  # [B, seq_len, d_model]
        x = self.norm(x)
        x = self.net(x)
        x=x.squeeze(-1)
        return self.fc(x)

class FeatureExpander(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear=nn.Linear(1, d_model)
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
    
        """
        Input shape: [B, 45]  
        Output shape: [B, 45, d_model]
        """
        
        data_expanded = data.unsqueeze(-1)
        expanded_data = self.linear(data_expanded)
        
        return expanded_data
        
class DenseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.residual = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
    def forward(self, x):
    
        attn_weights = self.attention(x)
        weighted_features = x * attn_weights
        combined = torch.cat([x, weighted_features], dim=1)
        interaction_out = self.interaction_net(combined)
        residual_out = self.residual(x)
        fused = interaction_out + residual_out
        
        return self.regressor(fused)

class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim), 
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.2)
        self.predictor = nn.Linear(input_dim, 1)

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_features = x * attention_weights
        prediction = self.predictor(weighted_features)
        return prediction, attention_weights#torch.Size([32, 45]) torch.Size([32, 1])
        
def adjust_learning_rate(optimizer, initial_lr, epoch, max_epochs, power=0.9):
    lr = initial_lr * (1 - float(epoch) / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 128
    state_size = 45
    seq_len = 45
    batch_size = 32

    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    #train_df = pd.read_csv('processed_data/processed_GN.csv')
    #train_df = pd.read_csv('processed_data/processed_SMOTE.csv')
    #train_df = pd.read_csv('processed_data/processed_mixup3.csv')
    train_X = train_df.drop(['Dmax', 'index'], axis=1).values
    train_y = train_df['Dmax'].values
    test_X = test_df.drop(['Dmax', 'index'], axis=1).values
    test_y = test_df['Dmax'].values

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    train_X_tensor = torch.tensor(train_X_scaled, dtype=torch.float32).to(device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32).view(-1, 1).to(device)
    test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).view(-1, 1).to(device)

    dataset = TensorDataset(train_X_tensor, train_y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GA_Mamba(d_model, state_size, device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    initial_lr = 0.001
    best_test_r2 = -float('inf')  

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad()
                test_predictions = model(test_X_tensor)
                test_r2 = r2_score(test_y, test_predictions.cpu().numpy())

                test_mse = mean_squared_error(test_y, test_predictions.cpu().numpy())
                test_rmse = np.sqrt(test_mse)

                print(f"Validation @ Epoch {epoch + 1}:")
                print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
                print(f"  Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}")
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    torch.save(model.state_dict(), 'best_mamba_model.pth')
                    print(f"Saved new best model with Test R²: {test_r2:.4f}")
        adjust_learning_rate(optimizer, initial_lr, epoch, num_epochs, power=0.9)
        
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_X_tensor)
        train_predictions = model(train_X_tensor)

        final_test_r2 = r2_score(test_y, test_predictions.cpu().numpy())
        final_train_r2 = r2_score(train_y, train_predictions.cpu().numpy())
        test_mse = mean_squared_error(test_y, test_predictions.cpu().numpy())
        test_rmse = np.sqrt(test_mse)

        print("\nFinal Evaluation:")
        print(f"Train R²: {final_train_r2:.4f} | Test R²: {final_test_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}")
    
    test_predictions=test_predictions.cpu()
    test_y=test_y.cpu()
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
    
    # print("\nTest Metrics:")
    # #print(f"Loss: {test_loss.item():.4f}")
    # print(f"R² Score: {test_r2:.4f}")
    # print(f"MSE: {test_mse:.4f}")
    # print(f"RMSE: {test_rmse:.4f}")
