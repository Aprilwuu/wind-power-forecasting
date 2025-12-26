from src.models.representation.vae import SequenceVAE
from src.models.representation.rnn_vae import RNNSequenceVAE
from src.models.probabilistic.beta.beta_transformer import BetaTimeSeriesTransformer
from src.models.probabilistic.beta.vae_beta_transformer import VAEBetaForecast
from src.models.losses import beta_nll_loss

def train_vae_beta(model, train_loader, val_loader, device, num_epoachs=50, lr=1e-3):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = {"train_loss": [], "val_loss":[]}

    for epoch in range(num_epoachs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)        # y ∈ [0, 1]     
        
            alpha, beta = model(X_batch)
            loss = beta_nll_loss(alpha, beta, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                alpha, beta = model(X_batch)
                loss = beta_nll_loss(alpha, beta, y_batch)
                val_losses.append(loss.item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))

    return history
