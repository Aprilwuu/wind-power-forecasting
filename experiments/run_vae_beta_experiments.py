from src.models.vae import SequenceVAE
from src.models.rnn_vae import RNNSequenceVAE
from src.probabilistic.beta_transformer import BetaTimeSeriesTransformer
from src.probabilistic.vae_beta_transformer import VAEBetaForecast
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


vae = SequenceVAE(input_dim=D, hidden_dim=64, latent_dim=16)
vae.load_state_dict(torch.load("models/vae_mlp.pt"))

beta_tr = BetaTimeSeriesTransformer(input_dim=latent_dim)
model_mlp = VAEBetaForecast(
    vae=vae,
    beta_transformer=beta_tr,
    freeze_vae=True,
    use_sample_z=False,
).to(device)
        
