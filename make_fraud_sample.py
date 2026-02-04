import argparse
import pandas as pd


def build_fraud_sample(train_transaction_path: str,
                       train_identity_path: str,
                       output_path: str,
                       nrows: int = 500,
                       chunksize: int = 200000) -> None:
    # Load fraud rows from train_transaction
    tx_iter = pd.read_csv(train_transaction_path, chunksize=chunksize)
    fraud_parts = []
    total_needed = nrows
    for chunk in tx_iter:
        fraud_chunk = chunk[chunk['isFraud'] == 1]
        if not fraud_chunk.empty:
            fraud_parts.append(fraud_chunk)
            total_needed -= len(fraud_chunk)
            if total_needed <= 0:
                break
    if not fraud_parts:
        raise RuntimeError("No fraud rows found in train_transaction.csv")
    fraud_tx = pd.concat(fraud_parts, ignore_index=True).head(nrows)

    fraud_ids = set(fraud_tx['TransactionID'].tolist())

    # Find matching identity rows by TransactionID using chunks
    id_iter = pd.read_csv(train_identity_path, chunksize=chunksize)
    id_parts = []
    for chunk in id_iter:
        matched = chunk[chunk['TransactionID'].isin(fraud_ids)]
        if not matched.empty:
            id_parts.append(matched)
    id_df = pd.concat(id_parts, ignore_index=True) if id_parts else pd.DataFrame(columns=['TransactionID'])

    merged = fraud_tx.merge(id_df, on='TransactionID', how='left')
    merged.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_transaction', type=str, default='train_transaction.csv')
    parser.add_argument('--train_identity', type=str, default='train_identity.csv')
    parser.add_argument('--output', type=str, default='my_fraud_sample.csv')
    parser.add_argument('--nrows', type=int, default=500)
    args = parser.parse_args()

    build_fraud_sample(
        train_transaction_path=args.train_transaction,
        train_identity_path=args.train_identity,
        output_path=args.output,
        nrows=args.nrows,
    )
    print(f"Wrote fraud-only sample file: {args.output}")


