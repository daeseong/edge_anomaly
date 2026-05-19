from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module,
          dataloader: DataLoader,
          epochs: int,
          learning_rate: float,
          weight_decay: float,
          device: torch.device,
          checkpoint_dir: Path
          ) -> list[float]:
    """
    Trains the model using MSE loss and Adam optimizer.
    Saves the best model (lowest training loss) to checkpoint_dir.

    Args:
        model : ConvAutoencoder instance
        dataloader : training DataLoader (normal images only)
        epochs : number of training epochs
        learning_rate : Adam learning rate
        weight_decay : Adam weight decay (L2)
        device : torch.device to train on
        checkpoint_dir : path to save best_model.pth

    Returns:
        loss_history : per-epoch avg loss
    """

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay,
                                 )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)

            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)

        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "loss": best_loss,}, 
                        checkpoint_dir / "best_model.pth")

    print(f"\nComplete. Best loss: {best_loss:.6f}")
    return loss_history