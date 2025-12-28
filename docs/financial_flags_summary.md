# Emerging Financial Flags: Task Summary & Data Overview

## 1. Task Definition
The **"Emerging Financial Flags"** tasks are a suite of binary classification problems designed to predict the **future adoption** of specific financial products or behaviors.

*   **Goal**: Predict if a customer will interact with a specific category (e.g., "Crypto Exchanges", "Mortgage") for the **first time**, given that they have not done so during their initial history.
*   **Primacy Period**: A customer is only eligible for a positive label if the behavior emerges **at least 3 months** after their account start date.
    *   *Early Adopters (behavior within first 3 months)* are **excluded** from the task.
*   **Prediction Horizon**: The model predicts the event **T days** in advance (e.g., T=7, 14, 30).

## 2. Dataset Statistics (Primacy V2)

### Account Population (Negatives & Base Pool)
After applying strict quality filters to ensure fair evaluation:
*   **Total Raw Accounts**: **34,776**
*   **Valid Evaluation Accounts**: **26,796** (77.1% Pass Rate)
    *   *Filter Criteria*: 
        1.  **Calendar Span ≥ 90 days** (time between first and last transaction)
        2.  **Active Days ≥ 9 days** (days with at least one transaction)

These **26,796** accounts serve as the pool for **Negative** samples (users who never adopt the behavior).

### Positive Labels (Emerging Events)
Counts below reflect **valid positives** that meet the same filter criteria (Span ≥ 90, Active ≥ 9).

| Flag Name | Count |
| :--- | :--- |
| **Total Valid Positives** | **2,241** |
| `EMERGING_CREDIT_CARD_PAYMENTS` | 430 |
| `EMERGING_SAVINGS` | 428 |
| `EMERGING_LOANS` | 250 |
| `EMERGING_INVESTMENTS` | 236 |
| `EMERGING_BUY_NOW_PAY_LATER` | 181 |
| `EMERGING_CAR_FINANCE` | 175 |
| `EMERGING_MORTGAGE` | 97 |
| `EMERGING_LINE_OF_CREDIT` | 82 |
| `EMERGING_STUDENT_LOAN` | 78 |
| `EMERGING_PERSONAL_LOAN` | 73 |
| `EMERGING_PAYDAY_LOANS` | 50 |
| `EMERGING_CRYPTO_EXCHANGES` | 43 |
| `EMERGING_DEBT_COLLECTION_AGENCIES` | 35 |
| `EMERGING_EQUIPMENT_FINANCE` | 19 |
| `EMERGING_OTHER_SAVINGS_INVESTMENTS` | 16 |
| ... (Long Tail Categories) | < 15 each |

*(Only 7 raw positive flags were dropped due to the quality filters, a loss of 0.3%.)*

## 3. Labeling Methodology

The labeling process runs on raw transaction data using the following logic:

1.  **Baseline Establishment**: Determine the `Account Start Date` (date of the very first transaction).
2.  **Event Detection**: Find the date of the **first occurring transaction** in a target category (e.g., "Investments").
3.  **Primacy Validation**:
    *   `Time to Event = First Event Date - Account Start Date`
    *   **If Time to Event ≥ 3 Months (approx. 90 days)**:
        *   -> **Label: POSITIVE** (Emerging Behavior)
    *   **If Time to Event < 3 Months**:
        *   -> **Label: NONE** (Excluded). The user is considered an "early adopter" and removed from training/evaluation for this specific task to avoid "predicting the past" or trivial predictions.
    *   **If Event Never Occurs**:
        *   -> **Label: NEGATIVE** (Non-Adopter)

## 4. Evaluation Cut-off Strategy

To simulate a realistic production forecast with a horizon of **T days**:

*   **For Positives (Adopters)**:
    *   **Cut-off Point**: `Emerging Event Date - T days`
    *   **Input**: All transaction history up to the cut-off.
    *   **Target**: Predict outcome "1" (Event happens in T days).

*   **For Negatives (Non-Adopters)**:
    *   **Cut-off Point**: `Last Active Date - T days`
    *   **Input**: All transaction history up to the cut-off.
    *   **Target**: Predict outcome "0" (Event does not happen).
    *   *Note*: This simulates "predicting T days before today" for a current user.

*   **Filters Applied**:
    *   Min Span: **90 days**
    *   Min Active Days: **9 days**
