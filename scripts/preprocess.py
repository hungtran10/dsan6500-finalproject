# Imports
import os
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def preprocess_csv_files(input_path, output_path, csv_files=None):
    """
    Function that loads CSVs (single, list, or default batch),
    combines and extracts data-type enforced data.
    """

    input_path = Path(input_path)

    # Handle variable input types for csv_files
    if csv_files is None:
        csv_paths = [
            input_path / "batch1_1.csv",
            input_path / "batch1_2.csv",
            input_path / "batch1_3.csv"
        ]
    elif isinstance(csv_files, (str, Path)):
        csv_paths = [input_path / Path(csv_files).name]
    else:
        csv_paths = [input_path / Path(p).name for p in csv_files]

    dfs = []
    for p in csv_paths:
        if p.exists():
            temp_df = pd.read_csv(p)
            temp_df["batch_csv"] = p.name
            dfs.append(temp_df)
        else:
            print(f"Skipping missing CSV: {p}")

    if not dfs:
        raise FileNotFoundError("No CSVs were found or loaded. Check your paths!")

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
            "total_amount": subtotal.get("total"),
            "net_worth": np.nan,   # filled in later after numeric conversion
        })

    fields = df["parsed_json"].apply(extract_fields)
    df = pd.concat([df, fields], axis=1)

    # Remove records with duplicate file names
    df.drop_duplicates(subset=["File Name"], inplace=True)

    def enforce_invoice_dtypes(df):
        text_cols = ["client_name", "seller_name", "invoice_number"]
        date_cols = ["invoice_date", "due_date"]

        df = df.copy()

        # TEXT FIELDS
        for col in text_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
            )

        # DATE FIELDS
        for col in date_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
            )
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # CLEAN NUMERIC STRINGS
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

        def compute_tax_value(tax_val, total_val):
            if pd.isna(tax_val):
                return np.nan

            # Percentage case
            if isinstance(tax_val, str) and tax_val.endswith("%"):
                try:
                    rate = float(tax_val.replace("%", "")) / 100
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

        # Convert total_amount to numeric
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")

        # Compute tax as amount
        df["tax"] = df.apply(
            lambda row: compute_tax_value(row["tax"], row["total_amount"]),
            axis=1
        )

        # Compute net_worth = total_amount - tax
        df["net_worth"] = np.round(df["total_amount"] - df["tax"], 2)

        return df

    df = enforce_invoice_dtypes(df)

    # Drop unnecessary columns
    df = df.drop(columns=["parsed_json", "Json Data"])

    # Save DataFrame as CSV
    df.to_csv(output_path, index=False)

    return df

class InvoiceImagePreprocessor:
    """
    Invoice processing pipeline for image files
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

        # Create output directory if doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
    def image_preprocessing(self, image_path, save_output=True):
        """
        Complete invoice preprocessing pipeline with all enhancement steps
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                return None
                
            # Step 1: Grayscale 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Bilateral Filter 
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Step 3: Upscale Image (2x for better OCR accuracy)
            height, width = denoised.shape
            upscaled = cv2.resize(denoised, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            
            # Step 4: Adaptive Threshold 
            binary = cv2.adaptiveThreshold(upscaled, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 15, 8)
            
            # Step 5: Morphological Operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            morphed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            
            # Step 6: Deskew the Image
            deskewed = self.deskew_image(morphed)
            
            # Step 7: Remove Borders
            border_removed = self.remove_borders(deskewed)
            
            # Step 8: Final Clean Binarized Image 
            kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            final = cv2.morphologyEx(border_removed, cv2.MORPH_CLOSE, kernel_final)
            
            # Save processed image if requested
            if save_output:
                output_path = self.output_dir / f"processed_{Path(image_path).name}"
                cv2.imwrite(str(output_path), final)
                return final, output_path
            
            return final, None
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, None

    def deskew_image(self, image):
        """
        Detect and correct skew in the image using Hough line detection
        """
        try:
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for line in lines[:10]: 
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5:
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        deskewed = cv2.warpAffine(image, rotation_matrix, (w, h),
                                                flags=cv2.INTER_CUBIC,
                                                borderMode=cv2.BORDER_REPLICATE)
                        return deskewed
        except:
            pass
        
        return image

    def remove_borders(self, image):
        """
        Remove document borders and edge artifacts
        """
        try:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add small margin and crop
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2*margin)
                h = min(image.shape[0] - y, h + 2*margin)
                
                cropped = image[y:y+h, x:x+w]
                return cropped
        except:
            pass
        
        return image

    def process_images(self, csv_path, image_folder_path, batch_size=50):
        """
        Process entire image folder with progress tracking
        """

        # Import and clean CSV of all files
        df = pd.read_csv(csv_path)
        print(f"Processing {len(df)} images from {csv_path}")
        print(f"Images location: {image_folder_path}")
        
        results = []
        failed_images = []
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            for idx, row in batch_df.iterrows():
                filename = row['File Name']
                image_path = Path(image_folder_path) / filename
                
                if not image_path.exists():
                    failed_images.append(f"File not found: {filename}")
                    continue
                
                processed_img, output_path = self.image_preprocessing(
                    image_path, save_output=True
                )
                
                # Image processing successful
                if processed_img is not None:
                    results.append({
                        'original_file': filename,
                        'processed_file': output_path.name if output_path else None,
                        'original_path': str(image_path),
                        'processed_path': str(output_path) if output_path else None,
                        'status': 'success'
                    })

                # Image processing unsuccessful
                else:
                    failed_images.append(filename)
                    results.append({
                        'original_file': filename,
                        'processed_file': None,
                        'original_path': str(image_path),
                        'processed_path': None,
                        'status': 'failed'
                    })

        # Print results
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
        print(f"Failed: {len(failed_images)}")
        print(f"Processed images saved to: {self.output_dir}")

        # Show some failed images
        if failed_images:
            print(f"\nFailed images:")
            for failed in failed_images[:10]:  # Show first 10
                print(f"  - {failed}")
            if len(failed_images) > 10:
                print(f"  ... and {len(failed_images) - 10} more")

        # Save processing results
        results_df = pd.DataFrame(results)
        return results_df
    
    def visualize_sample_results(self, results_df, n_samples=3):
        """
        Visualize sample preprocessing results
        """
        successful_results = results_df[results_df['status'] == 'success'].head(n_samples)
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 5*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, (_, row) in enumerate(successful_results.iterrows()):
            # Load original image
            original = cv2.imread(row['original_path'])
            if original is not None:
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                axes[i, 0].imshow(original_rgb)
                axes[i, 0].set_title(f'Original: {row["original_file"]}')
                axes[i, 0].axis('off')
            
            # Load processed image
            if row['processed_path'] and Path(row['processed_path']).exists():
                processed = cv2.imread(row['processed_path'], cv2.IMREAD_GRAYSCALE)
                axes[i, 1].imshow(processed, cmap='gray')
                axes[i, 1].set_title(f'Processed: {row["processed_file"]}')
                axes[i, 1].axis('off')
        
        plt.tight_layout(w_pad=0.5)
        plt.show()