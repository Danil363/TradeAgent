import torch
import numpy as np

def train(model, loader, val_loader, optimizer, loss_fn, device, epochs=30):
    # scheduler снижает LR, если val_loss перестал улучшаться
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    for epoch in range(epochs):
        model.train()
        losses = []

        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb).flatten()

            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb).flatten()
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = np.mean(losses)
        val_loss = np.mean(val_losses)

        # шаг шедулера
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )
