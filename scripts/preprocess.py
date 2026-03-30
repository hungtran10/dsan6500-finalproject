# Imports
import os
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_csv_files(input_path):

    csv_files = [
        input_path / "batch1_1.csv",
        input_path / "batch1_2.csv",
        input_path / "batch1_3.csv"
    ]

    dfs = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        df["batch_csv"] = csv.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Parse Json Data column
    df["parsed_json"] = [json.loads(x) if isinstance(x, str) else {} for x in df["Json Data"]]

    # Extract fields from JSON data
    def extract_fields(js):
        invoice = js.get("invoice", {})
        subtotal = js.get("subtotal", {})

        return pd.Series({
            "client_name": invoice.get("client_name"),
            "seller_name": invoice.get("seller_name"),
            "invoice_number": invoice.get("invoice_number"),
            "invoice_date": invoice.get("invoice_date"),
            "due_date": invoice.get("due_date"),
            "tax": subtotal.get("tax"),
            "total_amount": subtotal.get("total")
        })

    fields = df["parsed_json"].apply(extract_fields)

    # Concatenate results
    df = pd.concat([df, fields], axis=1)

    # Remove records with duplicate file names
    df.drop_duplicates(subset=['File Name'], inplace=True)

    # Function to handle data type and formatting
    def enforce_invoice_dtypes(df):

        text_cols = [
            "client_name",
            "seller_name",
            "invoice_number"
        ]

        date_cols = [
            "invoice_date",
            "due_date"
        ]

        df = df.copy()

        # ---------- TEXT FIELDS ----------
        for col in text_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
            )

        # ---------- DATE FIELDS ----------
        for col in date_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
            )

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

        # ---------- CLEAN NUMERIC STRINGS ----------
        for col in ["tax", "total_amount"]:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
                .replace("", np.nan)
            )

        # ---------- HANDLE TAX PERCENTAGES ----------
        def compute_tax_value(tax_val, total_val):

            if pd.isna(tax_val):
                return np.nan

            # Percentage case
            if isinstance(tax_val, str) and tax_val.endswith("%"):

                try:
                    rate = float(tax_val.replace("%","")) / 100

                    total_val = float(total_val)

                    subtotal = total_val / (1 + rate)

                    tax_amount = total_val - subtotal

                    return tax_amount

                except:
                    return np.nan

            # Normal numeric case
            try:
                return float(tax_val)
            except:
                return np.nan

        # Convert total first
        df["total_amount"] = pd.to_numeric(
            df["total_amount"],
            errors="coerce"
        )

        # Compute tax
        df["tax"] = df.apply(
            lambda row: compute_tax_value(row["tax"], row["total_amount"]),
            axis=1
        )

        return df

    df = enforce_invoice_dtypes(df)

    return df

    
