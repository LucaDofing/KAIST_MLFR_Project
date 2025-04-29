# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def run_training(model, train_loader, test_loader, device, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, total_nodes = 0.0, 0
        for data in loader:
            data = data.to(device)
            pred = model(data)
            loss = loss_fn(pred, data.y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * data.num_nodes
            total_nodes += data.num_nodes
        return total_loss / total_nodes

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(train_loader, train=True)
        if epoch % 20 == 0 or epoch == 1:
            test_loss = run_epoch(test_loader, train=False)
            print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    print("Training done.")
