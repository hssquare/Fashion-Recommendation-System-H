from pathlib import Path
import csv
import pandas as pd


CSV_PATH = Path(
    r"C:\D Drive Storage\Project\FRS HHHHHHHH\archive\styles.csv"
)


def load_metadata(csv_path: Path) -> pd.DataFrame:
    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.reader(file)

        # Read header
        header = next(reader)

        for line_number, row in enumerate(reader, start=2):

            # We need at least the 10 expected columns
            if len(row) < 10:
                print(
                    f"Skipping invalid row {line_number}: "
                    f"only {len(row)} fields"
                )
                continue

            # First 9 fields are fixed.
            # Everything after that belongs to productDisplayName.
            fixed_fields = row[:9]
            product_name = ",".join(row[9:])

            rows.append(
                fixed_fields + [product_name]
            )

    return pd.DataFrame(
        rows,
        columns=header
    )


if __name__ == "__main__":

    df = load_metadata(CSV_PATH)

    print("\nMetadata shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nProduct 44065:")
    result = df[
        df["id"].astype(str).str.strip() == "44065"
    ][
        [
            "id",
            "masterCategory",
            "subCategory",
            "articleType"
        ]
    ]

    print(result.to_string(index=False))