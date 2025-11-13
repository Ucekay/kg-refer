#!/usr/bin/env python3
"""
Script to convert user-item pairs to user-item lists format.
Converts:
3 34
3 52
3 92
To:
3 34 52 92
"""

from pathlib import Path

import pandas as pd


def convert_to_list_format(input_file, output_file):
    """
    Convert user-item pairs CSV to user-item list format.

    Args:
        input_file: Path to input CSV file with user,item format
        output_file: Path to output text file with user item1 item2 ... format
    """
    print(f"Converting {input_file} to list format...")

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Group by user and collect all items for each user
    user_items = df.groupby("user")["item"].apply(list)

    # Sort users for consistency
    user_items = user_items.sort_index()

    # Sort items for each user for consistency
    user_items = user_items.apply(lambda x: sorted(x))

    # Write to output file in the requested format
    with open(output_file, "w") as f:
        for user, items in user_items.items():
            # Format: user item1 item2 item3 ...
            line = f"{user} " + " ".join(map(str, items)) + "\n"
            f.write(line)

    print(f"  Input records: {len(df)}")
    print(f"  Unique users: {len(user_items)}")
    print(f"  Output saved to: {output_file}")

    return user_items


def verify_conversion(input_file, converted_file, sample_lines=5):
    """
    Verify the conversion by showing some examples.
    """
    print(f"\nVerifying conversion for {input_file}:")

    # Read original data
    df = pd.read_csv(input_file)

    # Show sample of original format
    print(f"Original format (first {sample_lines} lines):")
    print(df.head(sample_lines).to_string(index=False))

    # Show sample of converted format
    print(f"\nConverted format (first {sample_lines} lines):")
    with open(converted_file, "r") as f:
        for i, line in enumerate(f):
            if i < sample_lines:
                print(line.strip())
            else:
                break


if __name__ == "__main__":
    # Files to convert
    files_to_convert = [
        ("train.csv", "train.txt"),
        ("test.csv", "test.txt"),
        ("val.csv", "val.txt"),
    ]

    print("Converting CSV files to user-item list format...")
    print("=" * 50)

    for input_csv, output_txt in files_to_convert:
        print(f"\nProcessing {input_csv} -> {output_txt}:")
        convert_to_list_format(input_csv, output_txt)
        verify_conversion(input_csv, output_txt)

    print("\n" + "=" * 50)
    print("All conversions completed successfully!")

    # Show final file sizes
    print("\nGenerated files:")
    for _, output_txt in files_to_convert:
        if Path(output_txt).exists():
            with open(output_txt, "r") as f:
                line_count = sum(1 for _ in f)
            print(f"  {output_txt}: {line_count} lines (unique users)")
