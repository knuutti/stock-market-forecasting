import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, type, device=None):
    if device is None:
        device = get_device()
    model.to(device)
    best_val_loss = float('inf')
    all_train_losses = []
    all_val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/best_model_{type}.pth')

        all_train_losses.append(avg_train_loss)
        all_val_losses.append(avg_val_loss)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(all_train_losses, label='Train Loss')
    plt.plot(all_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Load the best model
    model.load_state_dict(torch.load(f'models/best_model_{type}.pth'))
    return model


# Evaluation function
def evaluate_model(model, test_loader, scaler_y, device=None):
    if device is None:
        device = get_device()
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # ensure X is on device for inference
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # move tensors to CPU before converting to numpy to avoid the cuda->numpy error
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    # Inverse transform
    real_predictions = scaler_y.inverse_transform(predictions)
    real_actuals = scaler_y.inverse_transform(actuals)

    # Print evaluation metrics
    rmse = root_mean_squared_error(real_actuals, real_predictions)
    mae = mean_absolute_error(real_actuals, real_predictions)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    return real_predictions, real_actuals
