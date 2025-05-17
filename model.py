import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

if __name__ == "__main__":
    import os
    import glob
    
    # Find all CSV files in the data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data/track')
    CSV_PATHS = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not CSV_PATHS:
        print("No CSV files found in the data directory!")
        print(f"Looking in: {data_dir}")
        exit(1)
    
    print(f"Found {len(CSV_PATHS)} CSV files for training:")
    BATCH_SIZE  = 64
    LR          = 1e-3
    EPOCHS      = 25
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accelbrake_features = [
        'recv_speedX', 'recv_speedY', 'recv_speedZ',
        'recv_angle',
        'recv_wheelSpinVel_0', 'recv_wheelSpinVel_1',
        'recv_wheelSpinVel_2', 'recv_wheelSpinVel_3',
    ] + [f'recv_track_{i}' for i in range(19)]

    gear_features = [
        'recv_speedX', 'recv_rpm', 'recv_gear', 'recv_angle',
    ] + [f'recv_track_{i}' for i in range(19)]

    steer_features = [
        'recv_angle', 'recv_speedX', 'recv_speedY', 'recv_speedZ',   'recv_wheelSpinVel_0', 'recv_wheelSpinVel_1',
        'recv_wheelSpinVel_2', 'recv_wheelSpinVel_3',
    ] + [f'recv_track_{i}' for i in range(19)] 
    dfs = []
    for path in CSV_PATHS:
        print(f"  - Loading {os.path.basename(path)}")
        try:
            df = pd.read_csv(path)
            print(f"    Found {len(df)} training examples")
            dfs.append(df)
        except Exception as e:
            print(f"    Error loading {path}: {str(e)}")
    
    if not dfs:
        print("No valid CSV files could be loaded. Please check your data files.")
        exit(1)
        
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total training examples: {len(df_all)}")

    X_ab = df_all[accelbrake_features].values.astype(np.float32)
    X_g  = df_all[gear_features].values.astype(np.float32)
    X_s  = df_all[steer_features].values.astype(np.float32)

    def get_action_label(row):
        if row['send_brake'] > 0.1:
            return 2  # Braking
        elif row['send_accel'] > 0.2:  # Using 0.2 as threshold for binary acceleration
            return 1  # Accelerating (will be recorded as 1)
        else:
            return 0  

    y_action = df_all.apply(get_action_label, axis=1).astype(np.int64).values
    y_gear  = (df_all['send_gear'].astype(int) + 1).values
    y_steer = ((df_all['send_steer'].values.astype(np.float32) + 1.0) / 2.0)

    # --- 2) Scale inputs separately ---
    scaler_ab = MinMaxScaler()
    scaler_g = MinMaxScaler()
    scaler_s = MinMaxScaler()


    X_ab = scaler_ab.fit_transform(X_ab)
    X_g  = scaler_g.fit_transform(X_g)
    X_s  = scaler_s.fit_transform(X_s)

    joblib.dump({'accelbrake': scaler_ab, 'gear': scaler_g, 'steer': scaler_s}, 'torcs_input_scalers.pkl')

# --- 3) Dataset & DataLoader ---
class TorcsDataset(Dataset):
    def __init__(self, X_ab, X_g, X_s, action, gear, steer):
        self.X_ab = torch.from_numpy(X_ab).float()
        self.X_g  = torch.from_numpy(X_g).float()
        self.X_s  = torch.from_numpy(X_s).float()
        self.y_action = torch.from_numpy(action).long()
        self.y_gear   = torch.from_numpy(gear).long()
        self.y_str    = torch.from_numpy(steer).float()
    def __len__(self):
        return len(self.X_ab)
    def __getitem__(self, idx):
        return (self.X_ab[idx], self.X_g[idx], self.X_s[idx],
                self.y_action[idx], self.y_gear[idx], self.y_str[idx])

# Only create dataset and dataloader when running as main script
if __name__ == "__main__":
    print("Creating training dataset and dataloader...")
    train_ds = TorcsDataset(X_ab, X_g, X_s, y_action, y_gear, y_steer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset size: {len(train_ds)} examples")

# --- 4) Define separate models ---
class ActionClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 3)  # classes: no-op, accel, brake
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class GearNet(nn.Module):
    def __init__(self, input_size, n_gears=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_gears)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class SteeringNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)

if __name__ == "__main__":
    model_action = ActionClassifier(X_ab.shape[1]).to(DEVICE)
    model_gear   = GearNet(X_g.shape[1]).to(DEVICE)
    model_steer  = SteeringNet(X_s.shape[1]).to(DEVICE)

    loss_action = nn.CrossEntropyLoss()
    loss_gear   = nn.CrossEntropyLoss()
    loss_steer  = nn.MSELoss()

    opt_action = torch.optim.Adam(model_action.parameters(), lr=LR)
    opt_gear   = torch.optim.Adam(model_gear.parameters(), lr=LR)
    opt_steer  = torch.optim.Adam(model_steer.parameters(), lr=LR)    # --- 5) Training loop with loss tracking ---

    
    # Track losses for reporting
    epoch_losses_action = []
    epoch_losses_gear = []
    epoch_losses_steer = []
    
    for epoch in range(1, EPOCHS + 1):
        model_action.train()
        model_gear.train()
        model_steer.train()
        
        batch_losses_action = []
        batch_losses_gear = []
        batch_losses_steer = []
        
        for batch_idx, (Xb_ab, Xb_g, Xb_s, yb_action, yb_gear, yb_str) in enumerate(train_loader):
            Xb_ab, Xb_g, Xb_s = Xb_ab.to(DEVICE), Xb_g.to(DEVICE), Xb_s.to(DEVICE)
            yb_action, yb_gear, yb_str = yb_action.to(DEVICE), yb_gear.to(DEVICE), yb_str.to(DEVICE)

            # --- Action ---
            opt_action.zero_grad()
            out_action = model_action(Xb_ab)
            action_loss = loss_action(out_action, yb_action)
            action_loss.backward()
            opt_action.step()
            batch_losses_action.append(action_loss.item())

            # --- Gear ---
            opt_gear.zero_grad()
            out_gear = model_gear(Xb_g)
            gear_loss = loss_gear(out_gear, yb_gear)
            gear_loss.backward()
            opt_gear.step()
            batch_losses_gear.append(gear_loss.item())

            # --- Steering ---
            opt_steer.zero_grad()
            out_steer = model_steer(Xb_s)
            steer_loss = loss_steer(out_steer, yb_str)
            steer_loss.backward()
            opt_steer.step()
            batch_losses_steer.append(steer_loss.item())
            
    
        
        # Calculate average losses for the epoch
        avg_action_loss = sum(batch_losses_action) / len(batch_losses_action)
        avg_gear_loss = sum(batch_losses_gear) / len(batch_losses_gear)
        avg_steer_loss = sum(batch_losses_steer) / len(batch_losses_steer)
        
        epoch_losses_action.append(avg_action_loss)
        epoch_losses_gear.append(avg_gear_loss)
        epoch_losses_steer.append(avg_steer_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} completed | "
              f"Action Loss: {avg_action_loss:.4f} | "
              f"Gear Loss: {avg_gear_loss:.4f} | "
              f"Steer Loss: {avg_steer_loss:.4f}")    # --- 6) Save models ---
    print("Training complete. Saving models...")
    
    # Define the save paths
    save_dir = os.path.dirname(__file__)
    model_action_path = os.path.join(save_dir, 'track_model_action.pkl')
    model_gear_path = os.path.join(save_dir, 'track_model_gear.pkl')
    model_steer_path = os.path.join(save_dir, 'track_model_steer.pkl')
    
    # Save the models
    torch.save(model_action.state_dict(), model_action_path)
    torch.save(model_gear.state_dict(), model_gear_path)
    torch.save(model_steer.state_dict(), model_steer_path)
    
    print(f"Saved action model to {model_action_path}")
    print(f"Saved gear model to {model_gear_path}")
    print(f"Saved steering model to {model_steer_path}")
    print("Training complete! Models are ready to use.")
