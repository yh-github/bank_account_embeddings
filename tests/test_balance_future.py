import pandas as pd
import pytest
from hierarchical.data.balance import calculate_balance_per_transaction

def test_future_transactions_handled_correctly():
    """
    Verify that transactions AFTER the balance snapshot are calculated correctly,
    starting from the snapshot and moving forward.
    """
    # Setup: Snapshot at T=10, Balance=100
    df_accounts = pd.DataFrame([{
        "accountId": "ACC1", 
        "availableBalance": 100.0, 
        "balanceDateTime": pd.Timestamp("2020-01-10")
    }])

    # Transactions:
    # 1. T=5, Amt=+10 (Historical) -> Should be included in initial_calc
    # 2. T=15, Amt=-20 (Future) -> Should just add to snapshot
    df_transactions = pd.DataFrame([
        {"accountId": "ACC1", "id": "tx1", "date": "2020-01-05", "amount": 10.0, "direction": "Credit"},
        {"accountId": "ACC1", "id": "tx2", "date": "2020-01-15", "amount": -20.0, "direction": "Debit"},
    ])

    # Expected logic trace:
    # Snapshot (T=10) = 100.
    # Historical Txns (<= T=10): tx1 (+10).
    # Initial Balance (T=0) = Snapshot - Sum(Historical) = 100 - 10 = 90.
    
    # Run Calc
    # tx1 Balance: Initial + CumSum(tx1) = 90 + 10 = 100.
    # tx2 Balance: Initial + CumSum(tx2) = 90 + (10 - 20) = 90 - 10 = 80.
    # Check: Snapshot + Future_Changes = 100 - 20 = 80. Matches.

    df_res = calculate_balance_per_transaction(df_transactions, df_accounts, include_pending=False)

    # Assertions
    tx1 = df_res[df_res["id"] == "tx1"].iloc[0]
    tx2 = df_res[df_res["id"] == "tx2"].iloc[0]

    assert tx1["balance_after"] == 100.0, f"Historical txn balance wrong. Got {tx1['balance_after']}, expected 100.0"
    assert tx2["balance_after"] == 80.0, f"Future txn balance wrong. Got {tx2['balance_after']}, expected 80.0"
