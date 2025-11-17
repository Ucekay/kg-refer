#!/usr/bin/env python3
"""
Script to split total.csv into train, test, and validation sets.

Following KGAT paper specification:
- 80% of each user's history goes to training set
- Remaining 20% goes to test set
- From the training set, randomly select 10% as validation set
- Final split: ~72% train, ~8% validation, ~20% test
"""

from pathlib import Path

import numpy as np
import pandas as pd


def split_data(
    input_file,
    output_dir,
    test_ratio=0.2,
    val_from_train_ratio=0.1,
    random_seed=42,
):
    """
    Split the input CSV file into train, test, and validation sets.
    For each user, split their records according to KGAT paper specification.

    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        test_ratio: Ratio of test data from total (default: 0.2 = 20%)
        val_from_train_ratio: Ratio of validation data from training set (default: 0.1 = 10%)
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
    print(f"\nSplitting data per user (KGAT paper specification)...")
    print(f"  Step 1: Split 80% for training (+ future validation), 20% for test")
    print(f"  Step 2: From training set, take 10% for validation")
    print(f"  Expected final ratio: ~72% train, ~8% validation, ~20% test\n")

    user_groups = df.groupby("user")
    users_with_no_val = 0
    users_with_no_test = 0

    for user_id, user_df in user_groups:
        # Shuffle user's records
        user_df_shuffled = user_df.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )

        user_total = len(user_df_shuffled)

        # Step 1: Split into (train+val) and test
        # 80% goes to train+val, 20% goes to test
        train_val_size = int(user_total * (1 - test_ratio))

        user_train_val = user_df_shuffled[:train_val_size]
        user_test = user_df_shuffled[train_val_size:]

        # Track users with no test data
        if len(user_test) == 0:
            users_with_no_test += 1

        # Step 2: From train+val, take 10% for validation
        # This gives us: train=72%, val=8%, test=20%
        # First secure train (90% of train+val), then take remaining 10% as validation
        # Use round instead of int for better distribution
        val_size = round(train_val_size * val_from_train_ratio)

        if val_size == 0 and train_val_size > 0:
            users_with_no_val += 1

        # Following paper: first secure train, then extract val from it
        train_size = train_val_size - val_size
        user_train = user_train_val[:train_size]
        user_val = user_train_val[train_size:]

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
    print(f"  Validation: {val_size} samples ({val_size / total_samples * 100:.1f}%)")
    print(f"  Test: {test_size} samples ({test_size / total_samples * 100:.1f}%)")

    # Warn about users with insufficient data
    if users_with_no_val > 0 or users_with_no_test > 0:
        print(f"\n⚠️  Warning: Some users have insufficient items for proper split:")
        if users_with_no_val > 0:
            print(
                f"  - {users_with_no_val} users have no validation data (train+val size * 10% < 0.5)"
            )
        if users_with_no_test > 0:
            print(f"  - {users_with_no_test} users have no test data (< 5 total items)")
        print(
            f"  Consider filtering out users with fewer interactions for better evaluation."
        )

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
            f"val={user_val} ({user_val / user_total:.1%}), test={user_test} ({user_test / user_total:.1%})"
        )

    return train_data, test_data, val_data


if __name__ == "__main__":
    # File paths
    input_file = "total.csv"
    output_dir = "."

    # Split the data according to KGAT paper specification
    # Final expected split: ~72% train, ~8% validation, ~20% test
    train_data, test_data, val_data = split_data(input_file, output_dir)

    print("\nData splitting completed successfully!")
    print("\nNote: This follows the KGAT paper specification:")
    print("  - 80% of each user's history → training set")
    print("  - 20% of each user's history → test set")
    print("  - 10% of training set → validation set")
    print("  - Final ratio: ~72% train, ~8% validation, ~20% test")
