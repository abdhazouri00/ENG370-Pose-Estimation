import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import utils
import argparse
import model
import DataLoader


def main(args):
    # Device configuration
    dev = torch.device(args.dev)

    # Data loading
    args.dtype = "train"
    train = DataLoader.data_loader(args)
    args.dtype = 'valid'
    val = DataLoader.data_loader(args)
    # Model initialization
    net_g = model.LSTM_g(
        embedding_dim=args.embedding_dim,
        h_dim=args.hidden_dim,
        dropout=args.dropout,
        dev=dev
    ).double().to(dev)

    encoder = model.Encoder(
        h_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        dev=dev
    )

    decoder = model.Decoder(
        h_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        dev=dev
    )

    net_l = model.VAE(Encoder=encoder, Decoder=decoder).double().to(dev)

    # Load checkpoints if needed
    if args.load_checkpoint:
        net_g.load_state_dict(torch.load("checkpoint_g.pkl"))
        net_l.load_state_dict(torch.load("checkpoint_l.pkl"))

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        list(net_l.parameters()) + list(net_g.parameters()),
        lr=args.lr
    )

    # Fixed scheduler initialization (remove verbose if needed)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.7,
        patience=35,
        threshold=1e-8
    )

    print('=' * 100)
    print('Training ...')

    for epoch in range(args.n_epochs):
        start = time.time()

        # Training phase
        net_g.train()
        net_l.train()
        train_metrics = {'loss': 0, 'ade': 0, 'fde': 0, 'vim': 0}

        for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, start_end_idx) in enumerate(train):
            batch = obs_p.size(1)

            # Move data to device
            obs_p = obs_p.double().to(dev)
            obs_s = obs_s.double().to(dev)
            target_p = target_p.double().to(dev)
            target_s = target_s.double().to(dev)

            # Global and local motion separation
            obs_s_g = 0.5 * (obs_s.view(15, batch, 13, 3)[:, :, 0] + obs_s.view(15, batch, 13, 3)[:, :, 1])
            target_s_g = 0.5 * (target_s.view(14, batch, 13, 3)[:, :, 0] + target_s.view(14, batch, 13, 3)[:, :, 1])
            obs_s_l = (obs_s.view(15, batch, 13, 3) - obs_s_g.view(15, batch, 1, 3)).view(15, batch, 39)
            target_s_l = (target_s.view(14, batch, 13, 3) - target_s_g.view(14, batch, 1, 3)).view(14, batch, 39)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            speed_preds_g = net_g(global_s=obs_s_g)
            loss_g = loss_fn(speed_preds_g, target_s_g)

            output, mean, log_var = net_l(obs_s_l)
            loss_l = model.vae_loss_function(target_s_l, output, mean, log_var)

            # Combine predictions
            speed_preds = (speed_preds_g.view(14, batch, 1, 3) + output.view(14, batch, 13, 3)).view(14, batch, 39)
            preds_p = utils.speed2pos(speed_preds, obs_p, dev=dev)

            # Combined loss and backward pass
            loss = loss_g + 0.1 * loss_l
            loss.backward()
            optimizer.step()

            # Update metrics
            train_metrics['loss'] += loss.item() * batch
            train_metrics['ade'] += float(utils.ADE_c(preds_p, target_p))
            train_metrics['fde'] += float(utils.FDE_c(preds_p, target_p))
            train_metrics['vim'] += utils.myVIM(preds_p, target_p)

        # Validation phase
        net_g.eval()
        net_l.eval()
        val_metrics = {'loss': 0, 'ade': 0, 'fde': 0, 'vim': 0}

        with torch.no_grad():
            for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, start_end_idx) in enumerate(val):
                batch = obs_p.size(1)

                # Move data to device
                obs_p = obs_p.double().to(dev)
                obs_s = obs_s.double().to(dev)
                target_p = target_p.double().to(dev)
                target_s = target_s.double().to(dev)

                # Global and local motion separation
                obs_s_g = 0.5 * (obs_s.view(15, batch, 13, 3)[:, :, 0] + obs_s.view(15, batch, 13, 3)[:, :, 1])
                target_s_g = 0.5 * (target_s.view(14, batch, 13, 3)[:, :, 0] + target_s.view(14, batch, 13, 3)[:, :, 1])
                obs_s_l = (obs_s.view(15, batch, 13, 3) - obs_s_g.view(15, batch, 1, 3)).view(15, batch, 39)
                target_s_l = (target_s.view(14, batch, 13, 3) - target_s_g.view(14, batch, 1, 3)).view(14, batch, 39)

                # Forward pass
                speed_preds_g = net_g(global_s=obs_s_g)
                loss_g = loss_fn(speed_preds_g, target_s_g)

                output, mean, log_var = net_l(obs_s_l)
                loss_l = model.vae_loss_function(target_s_l, output, mean, log_var)

                # Combine predictions
                speed_preds = (speed_preds_g.view(14, batch, 1, 3) + output.view(14, batch, 13, 3)).view(14, batch, 39)
                preds_p = utils.speed2pos(speed_preds, obs_p, dev=dev)

                # Update metrics
                loss = loss_g + 0.1 * loss_l
                val_metrics['loss'] += loss.item() * batch
                val_metrics['ade'] += float(utils.ADE_c(preds_p, target_p))
                val_metrics['fde'] += float(utils.FDE_c(preds_p, target_p))
                val_metrics['vim'] += utils.myVIM(preds_p, target_p)

        # Calculate averages
        train_loss = train_metrics['loss'] / len(train.dataset)
        val_loss = val_metrics['loss'] / len(val.dataset)

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f"e: {epoch} "
              f"|loss_t: {train_loss:.6f} "
              f"|loss_v: {val_loss:.6f} "
              f"|fde_t: {train_metrics['fde'] / len(train.dataset):.6f} "
              f"|fde_v: {val_metrics['fde'] / len(val.dataset):.6f} "
              f"|ade_t: {train_metrics['ade'] / len(train.dataset):.6f} "
              f"|ade_v: {val_metrics['ade'] / len(val.dataset):.6f} "
              f"|vim_t: {train_metrics['vim'] / len(train.dataset):.6f} "
              f"|vim_v: {val_metrics['vim'] / len(val.dataset):.6f} "
              f"|time(s): {time.time() - start:.6f}")

    # Save models
    torch.save(net_g.state_dict(), 'checkpoint_g.pkl')
    torch.save(net_l.state_dict(), 'checkpoint_l.pkl')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--loader_shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--loader_workers', type=int, default=1)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--dev', type=str, default='cpu')

    args = parser.parse_args()
    main(args)