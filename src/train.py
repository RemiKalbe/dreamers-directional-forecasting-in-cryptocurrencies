# train.py
import argparse
import os
import logging
import json
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from src.sagemaker_feature_store_helper import load_feature_store_data
from src.model import TimeSeriesTransformer
from src.dataset import TimeSeriesDataset
from src.loss import MarginBCEWithLogitsLoss
from src.metric import compute_f1_score

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def save_model(model: TimeSeriesTransformer, model_dir: str):
    # Save model parameters
    model_info = {
        "state_dict": model.state_dict(),
        "params": model.get_params(),
    }
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model_info, f)


def model_fn(model_dir: str) -> TimeSeriesTransformer:
    """Load the model for inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model info
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model_info = torch.load(f, map_location=device)

    # Initialize model with saved params
    model = TimeSeriesTransformer(
        feature_size=model_info["params"]["feature_size"],
        seq_length=model_info["params"]["seq_length"],
        d_model=model_info["params"]["d_model"],
        num_heads=model_info["params"]["num_heads"],
        num_layers=model_info["params"]["num_layers"],
        dropout=model_info["params"]["dropout"],
    )

    # Load state dict
    model.load_state_dict(model_info["state_dict"])
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model architecture hyperparameters
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=os.environ.get("SM_TENSORBOARD_DIR", "./tb_logs"),
    )
    parser.add_argument("--feature-group-name", type=str)
    parser.add_argument("--feature-bucket-name", type=str)

    args, _ = parser.parse_known_args()

    # Load data from Feature Store
    feature_store_data = load_feature_store_data(
        bucket_name=args.feature_bucket_name, feature_group_name=args.feature_group_name
    )

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = TimeSeriesDataset(
        feature_store_data["train_features"],
        feature_store_data["train_targets"],
        args.seq_length,
    )
    test_dataset = TimeSeriesDataset(
        feature_store_data["test_features"],
        feature_store_data["test_targets"],
        args.seq_length,
    )

    # Create distributed samplers
    train_sampler = (
        DistributedSampler(train_dataset) if torch.cuda.device_count() > 1 else None
    )
    test_sampler = (
        DistributedSampler(test_dataset, shuffle=False)
        if torch.cuda.device_count() > 1
        else None
    )

    # Create dataloaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Get feature size from the data
    feature_size = feature_store_data["train_features"].shape[1]

    # Initialize model with parsed arguments
    model = TimeSeriesTransformer(
        feature_size=feature_size,  # Determined from data
        seq_length=args.seq_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Wrap model with DistributedDataParallel
    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Only log on main process
    is_main_process = args.local_rank in [-1, 0]
    if is_main_process:
        writer = tensorboard.SummaryWriter(args.tensorboard_dir)  # type: ignore
        logger.info(f"Training with {torch.cuda.device_count()} GPUs")

    # Setup training
    criterion = MarginBCEWithLogitsLoss(margin=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=5e-3,
        step_size_up=2 * len(train_loader),
        cycle_momentum=False,
        mode="triangular2",
    )

    # (batch_size, sequence_length, feature_size)
    dummy_input = torch.randn(1, args.seq_length, args.feature_size).to(device)
    writer.add_graph(model, dummy_input)

    # Training loop
    best_val_f1 = 0.0
    global_step = 0  # For tensorboard logging
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        total_train_samples = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            global_step += 1

            batch_size = X_batch.size(0)
            total_train_samples += batch_size

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            train_loss += loss.item() * batch_size

            # Convert outputs to binary predictions
            preds = (outputs >= 0.5).long()
            train_preds.append(preds)
            train_targets.append(y_batch.long())

            # Log batch metrics
            if is_main_process and batch_idx % 100 == 0:
                avg_loss = train_loss / total_train_samples
                logger.info(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                            "timestamp": time.time(),
                        }
                    )
                )

                writer.add_scalar("Training/BatchLoss", avg_loss, global_step)
                writer.add_scalar("Training/LearningRate", current_lr, global_step)

                # Log gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(
                            f"gradients/{name}", param.grad, global_step
                        )

        # Compute epoch metrics
        train_preds_tensor = torch.cat(train_preds)
        train_targets_tensor = torch.cat(train_targets)
        avg_train_loss = train_loss / len(train_loader.dataset)  # type: ignore
        train_f1 = compute_f1_score(train_targets_tensor, train_preds_tensor)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                preds = (outputs >= 0.5).long()
                val_preds.append(preds)
                val_targets.append(y_batch.long())

        # Compute validation metrics
        val_preds_tensor = torch.cat(val_preds)
        val_targets_tensor = torch.cat(val_targets)
        avg_val_loss = val_loss / len(test_loader.dataset)  # type: ignore
        val_f1 = compute_f1_score(val_targets_tensor, val_preds_tensor)

        if is_main_process:
            # Log epoch metrics to tensorboard
            writer.add_scalars(
                "Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch
            )

            writer.add_scalars("F1", {"train": train_f1, "val": val_f1}, epoch)

            # Log model weights
            for name, param in model.named_parameters():
                writer.add_histogram(f"parameters/{name}", param, epoch)

            # Log epoch metrics
            logger.info(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "train_f1": train_f1,
                        "val_loss": avg_val_loss,
                        "val_f1": val_f1,
                        "learning_rate": current_lr,
                        "timestamp": time.time(),
                    }
                )
            )

        # Save model only on main process
        if is_main_process and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if isinstance(model, DistributedDataParallel):
                model_to_save = model.module  # Get the underlying model
            else:
                model_to_save = model
            model_to_save.save(os.path.join(args.model_dir, "model.pth"))
