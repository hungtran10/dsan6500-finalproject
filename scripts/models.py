import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import cv2
from PIL import Image
import re
import json
from pathlib import Path
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

class InvoiceTextDetector:
    """
    Text detection and OCR extraction for preprocessed invoice images
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$€£¥₹()[] '

    def detect_text_regions(self, image_path, visualize=False):
        """
        Detect likely text regions in an invoice image using contour-based heuristics.

        Parameters
        ----------
        image_path : str or Path
            Path to the invoice image.
        visualize : bool, optional
            If True, display the detected text regions on top of the image.

        Returns
        -------
        image : numpy.ndarray or None
            Grayscale image array if the image was loaded successfully; otherwise None.
        text_regions : list[tuple] or None
            List of bounding boxes in (x, y, w, h) format for detected text regions,
            or None if the image could not be loaded.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # If image doesn't exist
        if image is None:
            return None, None

        text_regions = self._contour_based_detection(image)

        if visualize:
            self._visualize_text_regions(image, text_regions)

        return image, text_regions

    def _contour_based_detection(self, image):
        """
        Detect text-like regions in a grayscale image using morphological dilation
        followed by contour extraction. This is a heuristic method designed to 
        group nearby characters into text blocks before OCR is applied.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale invoice image.

        Returns
        -------
        list[tuple]
            Bounding boxes for detected text regions in (x, y, w, h) format.
        """

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        dilated = cv2.dilate(image, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 15 and w < image.shape[1] * 0.9:
                text_regions.append((x, y, w, h))

        return text_regions

    def _visualize_text_regions(self, image, text_regions):
        """
        Visualize detected text regions on a grayscale image.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale invoice image.
        text_regions : list[tuple]
            Bounding boxes in (x, y, w, h) format to draw on the image.
        """
        image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for x, y, w, h in text_regions:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_copy)
        plt.title('Detected Text Regions')
        plt.axis('off')
        plt.show()

    def extract_text_from_image(self, image_path, confidence_threshold=30):
        """
        Extract word-level OCR output from an invoice image using Tesseract.
        The OCR output is filtered to keep only non-empty words with confidence
        above the given threshold.

        Parameters
        ----------
        image_path : str or Path
            Path to the invoice image.
        confidence_threshold : int, optional
            Minimum OCR confidence required to keep a detected word.

        Returns
        -------
        list[dict] or None
            A list of OCR word records. Each record contains:
            - text
            - confidence
            - bbox
            - block_num
            - par_num
            - line_num
            - word_num

            Returns None if the image cannot be loaded or OCR fails.
        """
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            # Apply OCR function to extract text into a dict
            ocr_data = pytesseract.image_to_data(image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            extracted_text = []

            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()

                # if confidence > threshold and model text
                if confidence > confidence_threshold and text:
                    extracted_text.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]),
                        'block_num': ocr_data['block_num'][i],
                        'par_num': ocr_data['par_num'][i],
                        'line_num': ocr_data['line_num'][i],
                        'word_num': ocr_data['word_num'][i]
                    })

            return extracted_text

        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return None

    def extract_invoice_fields(self, extracted_text):
        """
        Extract structured invoice fields from OCR word output using regex rules.

        Parameters
        ----------
        extracted_text : list[dict]
            OCR output from extract_text_from_image, where each element contains
            a text token and its bounding box/confidence metadata.

        Returns
        -------
        dict
            Dictionary of extracted invoice fields such as:
            - invoice_number
            - date
            - total_amount
            - vendor_name
            - client_name
            - vat_rate
            - currency

        Notes
        -----
        This method first reconstructs approximate text lines from OCR bounding boxes,
        then searches both line-level text and full text using regular expressions.
        """

        if not extracted_text:
            return {}

        full_text = ' '.join([item['text'] for item in extracted_text])
        lines = []
        current_line = []
        current_y = None

        for item in sorted(extracted_text, key=lambda x: (x['bbox'][1], x['bbox'][0])):
            y_pos = item['bbox'][1]
            if current_y is None or abs(y_pos - current_y) < 10:
                current_line.append(item['text'])
                current_y = y_pos
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [item['text']]
                current_y = y_pos

        if current_line:
            lines.append(' '.join(current_line))

        line_text = '\n'.join(lines)

        invoice_fields = {}

        patterns = {
            'invoice_number': [
                r'invoice\s*(?:no\.?|#|number)?\s*:?\s*([A-Z0-9\-]{4,})',
                r'inv(?:\.|oice)?\s*(?:no\.?|#)?\s*:?\s*([A-Z0-9\-]{4,})',
                r'(?:^|\n)(?:invoice|inv).*?([0-9]{6,})',
                r'(?:^|\n)([0-9]{6,})\s*(?:\n|$)',
                r'no\.?\s*:?\s*([0-9]{6,})',
            ],
            'date': [
                r'date.*?([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'issued?.*?([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4})',
                r'([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2})',
            ],
            'total_amount': [
                r'(?:total|grand\s*total|final).*?\$?\s*([0-9,]+\.[0-9]{2})',
                r'gross\s*worth.*?\$?\s*([0-9,]+\.[0-9]{2})',
                r'amount\s*due.*?\$?\s*([0-9,]+\.[0-9]{2})',
                r'\$\s*([0-9,]+\.[0-9]{2})(?=\s*$|\n)',
                r'total.*?([0-9,]+\.[0-9]{2})',
            ],
            'vendor_name': [
                r'^([A-Z][A-Za-z\-\s&\.]{2,30})(?=\n|\r)',
                r'seller:\s*\n?\s*([A-Za-z\-\s&\.]{3,40})',
                r'from:\s*([A-Za-z\-\s&\.]{3,40})',
            ],
            'client_name': [
                r'client:\s*\n?\s*([A-Za-z\-\s&\.]{3,40})',
                r'bill\s*to:\s*([A-Za-z\-\s&\.]{3,40})',
                r'to:\s*([A-Za-z\-\s&\.]{3,40})',
            ],
            'vat_rate': [
                r'vat.*?([0-9]{1,2})%',
                r'tax.*?([0-9]{1,2})%',
                r'([0-9]{1,2})%',
            ],
            'currency': [
                r'(\$|€|£|¥|₹)',
                r'(USD|EUR|GBP|INR)',
            ]
        }

        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, line_text, re.IGNORECASE | re.MULTILINE)
                if not match:
                    match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)

                if match:
                    value = match.group(1).strip()

                    if field in ['vendor_name', 'client_name']:
                        value = re.sub(r'^(seller|client|from|to):\s*', '', value, flags=re.IGNORECASE)
                        value = re.sub(r'\s*(ltd|inc|corp|llc)\.?\s*', '', value, flags=re.IGNORECASE)
                    elif field == 'total_amount':
                        value = re.sub(r'[,$]', '', value)
                        try:
                            float(value)
                        except ValueError:
                            continue

                    invoice_fields[field] = value
                    break

        return invoice_fields
    
    def process_single_image(self, image_path, visualize_regions=False):
        """
        Run the full OCR and field extraction pipeline on a single invoice image.

        Parameters
        ----------
        image_path : str or Path
            Path to the invoice image.
        visualize_regions : bool, optional
            If True, display detected text regions during processing.

        Returns
        -------
        dict
            A result dictionary containing:
            - image_path
            - filename
            - success
            - extracted_text
            - invoice_fields
            - text_regions
            - total_words
            - avg_confidence
        """
        result = {
            'image_path': str(image_path),
            'filename': Path(image_path).name,
            'success': False,
            'extracted_text': [],
            'invoice_fields': {},
            'text_regions': [],
            'total_words': 0,
            'avg_confidence': 0
        }
        
        try:
            # Detect text regions
            image, text_regions = self.detect_text_regions(image_path, visualize_regions)
            if image is None:
                return result
            
            result['text_regions'] = text_regions
            
            # Extract text using OCR
            extracted_text = self.extract_text_from_image(image_path)
            if extracted_text:
                result['extracted_text'] = extracted_text
                result['total_words'] = len(extracted_text)
                result['avg_confidence'] = np.mean([item['confidence'] for item in extracted_text])
                
                # Extract invoice fields
                result['invoice_fields'] = self.extract_invoice_fields(extracted_text)
                result['success'] = True
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
        
        return result
    
    def process_dataset(self, processed_images_df, batch_size=10):
        """
        Process a dataset of preprocessed invoice images in batches.
        Stores full per-image results in self.full_results for later analysis.

        Parameters
        ----------
        processed_images_df : pandas.DataFrame
            DataFrame containing preprocessed image paths and status information.
            Must include at least:
            - status
            - processed_path
        batch_size : int, optional
            Number of images to process per batch.

        Returns
        -------
        pandas.DataFrame
            Summary DataFrame containing filename, image path, success flag,
            word count, average OCR confidence, and whether invoice fields were found.

        Saves:
        - detailed_text_extraction.csv
        - extracted_invoice_fields.csv
        - text_detection_summary.csv
        """
        successful_images = processed_images_df[processed_images_df['status'] == 'success']
        print(f"Processing text detection for {len(successful_images)} images...")
        
        results = []
        
        for i in tqdm(range(0, len(successful_images), batch_size), desc="Processing OCR batches"):
            batch_df = successful_images.iloc[i:i+batch_size]
            
            for _, row in batch_df.iterrows():
                if row['processed_path'] and Path(row['processed_path']).exists():
                    result = self.process_single_image(row['processed_path'])
                    results.append(result)
        
        # Save detailed text extraction results
        detailed_results = []
        for result in results:
            if result['success']:
                for text_item in result['extracted_text']:
                    detailed_results.append({
                        'filename': result['filename'],
                        'text': text_item['text'],
                        'confidence': text_item['confidence'],
                        'bbox_x': text_item['bbox'][0],
                        'bbox_y': text_item['bbox'][1],
                        'bbox_width': text_item['bbox'][2],
                        'bbox_height': text_item['bbox'][3],
                        'block_num': text_item['block_num'],
                        'line_num': text_item['line_num']
                    })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(self.output_dir / 'detailed_text_extraction.csv', index=False)
        
        # Extract invoice fields summary
        invoice_fields_summary = []
        for result in results:
            if result['success'] and result['invoice_fields']:
                summary = {'filename': result['filename']}
                summary.update(result['invoice_fields'])
                invoice_fields_summary.append(summary)
        
        # Save invoice fields summary to CSV
        if invoice_fields_summary:
            fields_df = pd.DataFrame(invoice_fields_summary)
            fields_df.to_csv(self.output_dir / 'extracted_invoice_fields.csv', index=False)
        
        # Save simplified results for easier loading
        simplified_results = []
        for result in results:
            simplified_results.append({
                'filename': result['filename'],
                'image_path': result['image_path'],
                'success': result['success'],
                'total_words': result['total_words'],
                'avg_confidence': result['avg_confidence'],
                'has_invoice_fields': len(result['invoice_fields']) > 0
            })
        
        simplified_df = pd.DataFrame(simplified_results)
        simplified_df.to_csv(self.output_dir / 'text_detection_summary.csv', index=False)
        
        self._print_summary(results)
        
        # Store full results for visualization
        self.full_results = results
        
        return simplified_df
    
    def _print_summary(self, results):
        """
        Print a summary of OCR and invoice field extraction results.

        Parameters
        ----------
        results : list[dict]
            List of per-image result dictionaries produced by process_single_image function.
        """
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{'='*60}")
        print("TEXT DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful extractions: {len(successful)}")
        print(f"Failed extractions: {len(failed)}")
        
        if successful:
            avg_words = np.mean([r['total_words'] for r in successful])
            avg_confidence = np.mean([r['avg_confidence'] for r in successful])
            print(f"Average words per image: {avg_words:.1f}")
            print(f"Average confidence: {avg_confidence:.1f}%")
            
            # Count extracted fields
            field_counts = {}
            for result in successful:
                for field in result['invoice_fields'].keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            if field_counts:
                print(f"\nExtracted invoice fields:")
                for field, count in field_counts.items():
                    print(f"  {field}: {count} images ({count/len(successful)*100:.1f}%)")
    
    def visualize_text_extraction(self, image_path, result):
        """
        Visualize OCR word boxes and extracted invoice fields for a single image.

        Parameters
        ----------
        image_path : str or Path
            Path to the invoice image.
        result : dict
            Result dictionary returned by process_single_image.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes around detected text
        for text_item in result['extracted_text']:
            x, y, w, h = text_item['bbox']
            confidence = text_item['confidence']
            
            # Color based on confidence
            if confidence > 80:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 50:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
            
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_rgb, f"{text_item['text'][:10]}...", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(image_rgb)
        plt.title(f'Text Detection Results: {Path(image_path).name}')
        plt.axis('off')
        plt.show()
        
        # Print extracted fields
        if result['invoice_fields']:
            print("\nExtracted Invoice Fields:")
            for field, value in result['invoice_fields'].items():
                print(f"  {field}: {value}")

    def create_analysis_dashboard(self):
        """
        Create a comprehensive analysis dashboard for OCR and invoice extraction.
        Requires process_dataset to have been run first so that self.full_results exists.

        Returns
        -------
        dict or None
            Summary statistics including:
            - total_processed
            - successful
            - field_extraction_rates
            - avg_confidence
            - avg_words

            Returns None if no results are available.
        """
        if not hasattr(self, 'full_results'):
            print("No results to analyze. Run process_dataset first.")
            return
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE INVOICE ANALYSIS DASHBOARD")
        print(f"{'='*80}")
        
        # Overall Processing Statistics
        successful_results = [r for r in self.full_results if r['success']]
        failed_results = [r for r in self.full_results if not r['success']]
        
        print(f"\nPROCESSING OVERVIEW")
        print(f"{'='*50}")
        print(f"Total images processed: {len(self.full_results):,}")
        print(f"Successful extractions: {len(successful_results):,} ({len(successful_results)/len(self.full_results)*100:.1f}%)")
        print(f"Failed extractions: {len(failed_results):,} ({len(failed_results)/len(self.full_results)*100:.1f}%)")
        
        if not successful_results:
            print("No successful results to analyze.")
            return
        
        # OCR Quality Analysis
        print(f"\nOCR QUALITY METRICS")
        print(f"{'='*50}")
        
        word_counts = [r['total_words'] for r in successful_results]
        confidences = [r['avg_confidence'] for r in successful_results]
        
        print(f"Average words per invoice: {np.mean(word_counts):.1f}")
        print(f"Median words per invoice: {np.median(word_counts):.0f}")
        print(f"Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"Average OCR confidence: {np.mean(confidences):.1f}%")
        print(f"Median OCR confidence: {np.median(confidences):.1f}%")
        print(f"Confidence range: {min(confidences):.1f}% - {max(confidences):.1f}%")
        
        # Field Extraction Analysis
        print(f"\nFIELD EXTRACTION ANALYSIS")
        print(f"{'='*50}")
        
        field_counts = {}
        all_fields = {}
        
        for result in successful_results:
            for field, value in result['invoice_fields'].items():
                if field not in field_counts:
                    field_counts[field] = 0
                    all_fields[field] = []
                field_counts[field] += 1
                all_fields[field].append(value)
        
        total_invoices = len(successful_results)
        print(f"Field extraction success rates:")

        # Calculate success % per field
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_invoices * 100
            print(f"  {field:15}: {count:3d} invoices ({percentage:5.1f}%)")
        
        # Data Quality Insights
        print(f"\nDATA QUALITY INSIGHTS")
        print(f"{'='*50}")
        
        if 'total_amount' in all_fields:
            amounts = []
            for amount_str in all_fields['total_amount']:
                try:
                    # Clean and convert to float
                    clean_amount = re.sub(r'[,$]', '', amount_str)
                    amount = float(clean_amount)
                    amounts.append(amount)
                except:
                    pass
            
            if amounts:
                print(f"Total amount statistics:")
                print(f"  Count: {len(amounts)} valid amounts")
                print(f"  Average: ${np.mean(amounts):.2f}")
                print(f"  Median: ${np.median(amounts):.2f}")
                print(f"  Range: ${min(amounts):.2f} - ${max(amounts):.2f}")
                print(f"  Total value: ${sum(amounts):,.2f}")
        
        if 'invoice_number' in all_fields:
            inv_numbers = all_fields['invoice_number']
            print(f"\nInvoice number patterns:")
            print(f"  Total extracted: {len(inv_numbers)}")
            
            # Analyze patterns
            numeric_only = [n for n in inv_numbers if n.isdigit()]
            alphanumeric = [n for n in inv_numbers if not n.isdigit()]
            
            print(f"  Numeric only: {len(numeric_only)} ({len(numeric_only)/len(inv_numbers)*100:.1f}%)")
            print(f"  Alphanumeric: {len(alphanumeric)} ({len(alphanumeric)/len(inv_numbers)*100:.1f}%)")
            
            if numeric_only:
                lengths = [len(n) for n in numeric_only]
                print(f"  Numeric length range: {min(lengths)} - {max(lengths)} digits")
        
        # Create visualizations
        self._create_analysis_plots(successful_results, all_fields)
        
        # Data Export Summary
        print(f"\nEXPORTED DATA FILES")
        print(f"{'='*50}")
        csv_files = list(self.output_dir.glob('*.csv'))
        for csv_file in csv_files:
            file_size = csv_file.stat().st_size / 1024  # KB
            print(f"  {csv_file.name}: {file_size:.1f} KB")
        
        return {
            'total_processed': len(self.full_results),
            'successful': len(successful_results),
            'field_extraction_rates': {k: v/total_invoices for k, v in field_counts.items()},
            'avg_confidence': np.mean(confidences),
            'avg_words': np.mean(word_counts)
        }
    
    def _create_analysis_plots(self, successful_results, all_fields):
        """
        Create diagnostic plots for OCR quality and invoice field extraction.

        Parameters
        ----------
        successful_results : list[dict]
            List of successful per-image OCR/extraction results.
        all_fields : dict
            Dictionary mapping field names to lists of extracted field values.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Invoice Processing Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: OCR Confidence Distribution
        confidences = [r['avg_confidence'] for r in successful_results]
        axes[0, 0].hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('OCR Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidences):.1f}%')
        axes[0, 0].legend()
        
        # Plot 2: Words per Invoice Distribution
        word_counts = [r['total_words'] for r in successful_results]
        axes[0, 1].hist(word_counts, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Words per Invoice Distribution')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(word_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(word_counts):.1f}')
        axes[0, 1].legend()
        
        # Plot 3: Field Extraction Success Rates
        field_counts = {}
        for result in successful_results:
            for field in result['invoice_fields'].keys():
                field_counts[field] = field_counts.get(field, 0) + 1
        
        if field_counts:
            fields = list(field_counts.keys())
            rates = [field_counts[f]/len(successful_results)*100 for f in fields]
            
            bars = axes[1, 0].bar(range(len(fields)), rates, color='orange', alpha=0.7)
            axes[1, 0].set_title('Field Extraction Success Rates')
            axes[1, 0].set_xlabel('Fields')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_xticks(range(len(fields)))
            axes[1, 0].set_xticklabels(fields, rotation=45, ha='right')
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{rate:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Amount Distribution (if available)
        if 'total_amount' in all_fields:
            amounts = []
            for amount_str in all_fields['total_amount']:
                try:
                    clean_amount = re.sub(r'[,$]', '', amount_str)
                    amount = float(clean_amount)
                    if amount > 0 and amount < 10000:  # Filter reasonable amounts
                        amounts.append(amount)
                except:
                    pass
            
            if amounts:
                axes[1, 1].hist(amounts, bins=15, color='purple', alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Invoice Amount Distribution')
                axes[1, 1].set_xlabel('Amount ($)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(np.mean(amounts), color='red', linestyle='--',
                                  label=f'Mean: ${np.mean(amounts):.2f}')
                axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No amount data\navailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Invoice Amount Distribution')
        
        plt.tight_layout()
        plt.show()
    
    

    def visualize_sample_results(self, n_samples=3):
        """
        Visualize a small set of sample OCR and extraction results.

        Parameters
        ----------
        n_samples : int, optional
            Number of successful invoice examples to display.
        """
        if not hasattr(self, 'full_results'):
            print("No results to visualize. Run process_dataset first.")
            return
        
        successful_results = [r for r in self.full_results if r['success']][:n_samples]
        
        for i, result in enumerate(successful_results):
            print(f"\n{'='*60}")
            print(f"Sample {i+1}: {result['filename']}")
            print(f"{'='*60}")
            
            # Show basic stats
            print(f"Total words detected: {result['total_words']}")
            print(f"Average confidence: {result['avg_confidence']:.1f}%")
            
            # Show extracted invoice fields
            if result['invoice_fields']:
                print("\nExtracted Invoice Fields:")
                for field, value in result['invoice_fields'].items():
                    print(f"  {field}: {value}")
            
            # Show sample text (first 10 words)
            if result['extracted_text']:
                print(f"\nSample extracted text (first 10 words):")
                sample_text = ' '.join([item['text'] for item in result['extracted_text'][:10]])
                print(f"  {sample_text}...")
            
            # Visualize on image
            if Path(result['image_path']).exists():
                self.visualize_text_extraction(result['image_path'], result)

# Usage Example
if __name__ == "__main__":
    # Initialize text detector
    text_detector = InvoiceTextDetector()
    
    # Load preprocessing results
    preprocessing_results = pd.read_csv('/kaggle/working/all_processing_results.csv')
    
    # Process text detection with improved field extraction
    print("Starting text detection and OCR with improved field extraction...")
    ocr_summary = text_detector.process_dataset(preprocessing_results)
    
    # Create comprehensive analysis dashboard
    print("\nGenerating comprehensive analysis dashboard...")
    analysis_results = text_detector.create_analysis_dashboard()
    
    # Visualize sample results
    print("\nVisualizing sample results...")
    text_detector.visualize_sample_results(n_samples=3)
    
    print(f"\nText detection results saved to: {text_detector.output_dir}")
    print("Files created:")
    print("  - text_detection_summary.csv: Processing summary")
    print("  - detailed_text_extraction.csv: Word-level extraction details")
    print("  - extracted_invoice_fields.csv: Structured invoice field data")
    print("\n🎉 Analysis complete! Check the dashboard above for insights.")
