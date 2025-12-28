#!/usr/bin/env python3
"""Run financial flags detection for all banks and summarize results.

Outputs aggregated CSVs with bank column.

Usage:
    python -m scripts.data_prep.generate_labels --config config/local.yaml
    
Or with environment variable:
    export EMBEDDER_CONFIG=config/local.yaml
    python -m scripts.data_prep.generate_labels
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

from hierarchical.config import load_config, get_path, get_banks
from hierarchical.data.financial_flags import main as run_flags, load_config as load_flags_config


def run_all_banks(config_path: str | None = None) -> pd.DataFrame:
    """Run financial flags for all banks and collect results into aggregated CSVs.

    Args:
        config_path: Optional path to config file.

    Returns:
        DataFrame with summary results per bank.
    """
    # Load configuration
    config = load_config(config_path)
    
    data_dir = get_path("data_dir", required=True)
    output_dir = get_path("output_dir", required=False) or Path("./output")
    banks = get_banks()

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load flags config template
    flags_config_path = Path(__file__).parent / "config.yaml"
    flags_config = load_flags_config(str(flags_config_path))

    results = []
    all_emerging = []
    all_skipped = []

    for bank in banks:
        print(f"\n{'='*60}")
        print(f"Processing: {bank}")
        print("=" * 60)

        # Update config for this bank
        bank_config = flags_config.copy()
        bank_config["paths"] = {
            "transaction_file": str(data_dir / bank / "DTransaction_output.csv"),
            "account_file": str(data_dir / bank / "DAccount_output.csv"),
            "recurring_pattern_file": str(data_dir / bank / "DRecurringPattern_output.csv"),
            "output_dir": str(temp_dir),
        }

        # Write temporary config
        temp_config_path = temp_dir / f"config_{bank}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(bank_config, f)

        try:
            # Get user count from transaction file
            txn_file = data_dir / bank / "DTransaction_output.csv"
            if not txn_file.exists():
                print(f"  Skipping {bank}: transaction file not found")
                continue

            txn_df = pd.read_csv(txn_file, usecols=["user_name"], low_memory=False)
            user_count = txn_df["user_name"].nunique()

            # Run the flags detection
            run_flags(str(temp_config_path))

            # Read results and add bank column
            emerging_path = temp_dir / "emerging_flags.csv"
            skipped_path = temp_dir / "skipped_payment_flags.csv"

            emerging_count = 0
            skipped_count = 0

            if emerging_path.exists():
                emerging_df = pd.read_csv(emerging_path)
                emerging_df.insert(0, "bank", bank)
                all_emerging.append(emerging_df)
                emerging_count = len(emerging_df)

            if skipped_path.exists():
                skipped_df = pd.read_csv(skipped_path)
                skipped_df.insert(0, "bank", bank)
                all_skipped.append(skipped_df)
                skipped_count = len(skipped_df)

            results.append({
                "bank": bank,
                "users": user_count,
                "emerging_flags": emerging_count,
                "skipped_flags": skipped_count,
                "total_flags": emerging_count + skipped_count,
            })

        except Exception as e:
            print(f"Error processing {bank}: {e}")
            results.append({
                "bank": bank,
                "users": "ERROR",
                "emerging_flags": "ERROR",
                "skipped_flags": "ERROR",
                "total_flags": "ERROR",
            })

        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()

    # Clean up temp directory
    for f in temp_dir.iterdir():
        if f.is_file():
            f.unlink()
    temp_dir.rmdir()

    # Save aggregated CSVs
    print("\n" + "=" * 70)
    print("Saving aggregated outputs...")
    print("=" * 70)

    if all_emerging:
        combined_emerging = pd.concat(all_emerging, ignore_index=True)
        combined_emerging.to_csv(output_dir / "emerging_flags.csv", index=False)
        print(f"  {output_dir}/emerging_flags.csv ({len(combined_emerging)} rows)")

    if all_skipped:
        combined_skipped = pd.concat(all_skipped, ignore_index=True)
        combined_skipped.to_csv(output_dir / "skipped_payment_flags.csv", index=False)
        print(f"  {output_dir}/skipped_payment_flags.csv ({len(combined_skipped)} rows)")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Flag Counts by Bank")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Calculate totals
    print("\n" + "-" * 70)
    total_users = sum(r["users"] for r in results if isinstance(r["users"], int))
    total_emerging = sum(r["emerging_flags"] for r in results if isinstance(r["emerging_flags"], int))
    total_skipped = sum(r["skipped_flags"] for r in results if isinstance(r["skipped_flags"], int))
    print(
        f"TOTAL: {total_users:,} Users | "
        f"{total_emerging} Emerging + {total_skipped} Skipped = "
        f"{total_emerging + total_skipped} Total Flags"
    )

    # Save summary to CSV
    results_df.to_csv(output_dir / "summary_all_banks.csv", index=False)
    print(f"\nSummary saved to: {output_dir}/summary_all_banks.csv")

    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate financial flags for all banks")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (or set HIERARCHICAL_CONFIG env var)",
    )
    args = parser.parse_args()

    run_all_banks(args.config)


if __name__ == "__main__":
    main()
