import numpy as np
import torch
from tqdm import tqdm

from kgat.utils.metriccs import calc_metrics_at_k


def evaluate(model, dataloader, Ks, device, use_validation=False):
    """
    Evaluate model performance on validation or test set.

    Args:
        model: The KGAT model to evaluate
        dataloader: DataLoader containing train/val/test data
        Ks: List of K values for top-K metrics
        device: Device to run evaluation on
        use_validation: If True, evaluate on validation set; if False, evaluate on test set

    Returns:
        cf_scores: Prediction scores for all users
        metrics_dict: Dictionary of metrics (precision, recall, ndcg) for each K

    Note:
        - During training: use_validation=True (for hyperparameter tuning and early stopping)
        - Final evaluation: use_validation=False (for reporting final performance)
    """
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict

    # Select validation or test set based on parameter
    eval_user_dict = (
        dataloader.val_user_dict if use_validation else dataloader.test_user_dict
    )
    eval_name = "Validation" if use_validation else "Test"

    model.eval()

    user_ids = list(eval_user_dict.keys())
    user_ids_batches = [
        user_ids[i : i + test_batch_size]
        for i in range(0, len(user_ids), test_batch_size)
    ]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = [
        "precision",
        "recall",
        "ndcg",
    ]
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc=f"Evaluating {eval_name}") as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode="predict")

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(
                batch_scores,
                train_user_dict,
                eval_user_dict,
                batch_user_ids.cpu().numpy(),
                item_ids.cpu().numpy(),
                Ks,
            )

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict
