#!/usr/bin/env python3
"""
Script to split total.csv into train, test, and validation sets.
80% train, 10% test, 10% validation with random shuffling.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def split_data(
    input_file,
    output_dir,
    train_ratio=0.8,
    test_ratio=0.1,
    val_ratio=0.1,
    random_seed=42,
):
    """
    Split the input CSV file into train, test, and validation sets.
    For each user, split their records into train/test/val sets.

    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        train_ratio: Ratio of training data (default: 0.8)
        test_ratio: Ratio of test data (default: 0.1)
        val_ratio: Ratio of validation data (default: 0.1)
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Read the total CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Total number of records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Initialize empty dataframes for each split
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    val_data = pd.DataFrame()

    # Group by user and split each user's records
    print(f"\nSplitting data per user...")
    user_groups = df.groupby("user")

    for user_id, user_df in user_groups:
        # Shuffle user's records
        user_df_shuffled = user_df.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )

        # Calculate split indices for this user
        user_total = len(user_df_shuffled)
        user_train_size = int(user_total * train_ratio)
        user_test_size = int(user_total * test_ratio)
        user_val_size = user_total - user_train_size - user_test_size

        # Split this user's data
        user_train = user_df_shuffled[:user_train_size]
        user_test = user_df_shuffled[user_train_size : user_train_size + user_test_size]
        user_val = user_df_shuffled[user_train_size + user_test_size :]

        # Add to overall splits
        train_data = pd.concat([train_data, user_train], ignore_index=True)
        test_data = pd.concat([test_data, user_test], ignore_index=True)
        val_data = pd.concat([val_data, user_val], ignore_index=True)

    # Calculate final split sizes
    total_samples = len(df)
    train_size = len(train_data)
    test_size = len(test_data)
    val_size = len(val_data)

    print(f"Final split sizes:")
    print(f"  Train: {train_size} samples ({train_size / total_samples * 100:.1f}%)")
    print(f"  Test: {test_size} samples ({test_size / total_samples * 100:.1f}%)")
    print(f"  Validation: {val_size} samples ({val_size / total_samples * 100:.1f}%)")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the splits
    train_file = output_path / "train.csv"
    test_file = output_path / "test.csv"
    val_file = output_path / "val.csv"

    print(f"\nSaving files:")
    train_data.to_csv(train_file, index=False)
    print(f"  Train data saved to: {train_file}")

    test_data.to_csv(test_file, index=False)
    print(f"  Test data saved to: {test_file}")

    val_data.to_csv(val_file, index=False)
    print(f"  Validation data saved to: {val_file}")

    # Print some statistics
    print(f"\nData statistics:")
    print(f"  Unique users in total: {df['user'].nunique()}")
    print(f"  Unique items in total: {df['item'].nunique()}")

    for name, data in [("Train", train_data), ("Test", test_data), ("Val", val_data)]:
        print(
            f"  {name}: {len(data)} records, {data['user'].nunique()} users, {data['item'].nunique()} items"
        )

    # Verify per-user split ratios
    print(f"\nVerifying per-user split ratios (sample of 10 users):")
    unique_users = df["user"].unique()
    sample_size = min(10, len(unique_users))
    sample_users = np.random.choice(unique_users, sample_size, replace=False)

    for user_id in sample_users:
        user_total = len(df[df["user"] == user_id])
        user_train = len(train_data[train_data["user"] == user_id])
        user_test = len(test_data[test_data["user"] == user_id])
        user_val = len(val_data[val_data["user"] == user_id])

        print(
            f"  User {user_id}: total={user_total}, train={user_train} ({user_train / user_total:.1%}), "
            f"test={user_test} ({user_test / user_total:.1%}), val={user_val} ({user_val / user_total:.1%})"
        )

    return train_data, test_data, val_data


if __name__ == "__main__":
    # File paths
    input_file = "total.csv"
    output_dir = "."

    # Split the data
    train_data, test_data, val_data = split_data(input_file, output_dir)

    print("\nData splitting completed successfully!")
