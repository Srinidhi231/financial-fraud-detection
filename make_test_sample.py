import argparse
import pandas as pd


def build_sample(test_transaction_path: str,
                 test_identity_path: str,
                 output_path: str,
                 nrows: int = 1000,
                 chunksize: int = 200000) -> None:
    tx = pd.read_csv(test_transaction_path, nrows=nrows)
    tx_ids = set(tx['TransactionID'].tolist())

    matched_identity_chunks = []
    for chunk in pd.read_csv(test_identity_path, chunksize=chunksize):
        matched = chunk[chunk['TransactionID'].isin(tx_ids)]
        if not matched.empty:
            matched_identity_chunks.append(matched)

    if matched_identity_chunks:
        id_df = pd.concat(matched_identity_chunks, ignore_index=True)
    else:
        id_df = pd.DataFrame(columns=['TransactionID'])

    merged = tx.merge(id_df, on='TransactionID', how='left')
    merged.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_transaction', type=str, default='test_transaction.csv')
    parser.add_argument('--test_identity', type=str, default='test_identity.csv')
    parser.add_argument('--output', type=str, default='my_test_input.csv')
    parser.add_argument('--nrows', type=int, default=1000)
    args = parser.parse_args()

    build_sample(
        test_transaction_path=args.test_transaction,
        test_identity_path=args.test_identity,
        output_path=args.output,
        nrows=args.nrows,
    )
    print(f"Wrote sample file: {args.output}")


