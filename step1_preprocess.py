# -*- coding: utf-8 -*-
"""
Step 1: Paper Preprocessing

This script takes a scientific paper in PDF format as input and performs two main tasks:
1.  Uses the Nougat OCR model to convert the PDF into a structured Markdown (.md) file,
    preserving formulas and tables.
2.  Extracts all images from the PDF, processes them to a target DPI, and saves them
    in a separate 'images' directory.

This script is designed to be called as part of an automated pipeline.

Inputs (via command-line arguments):
  --pdf_path: The file path to the input PDF.
  --output_dir: The directory where the .md file and 'images' folder will be saved.

Configuration (read from config.ini):
  [PATHS]
    NOUGAT_MODEL_PATH: The local file path to the pre-downloaded Nougat model.
  [SETTINGS]
    TARGET_DPI: The target resolution for extracted images.
"""

# 1. IMPORTS
import fitz  # PyMuPDF
import subprocess
from pathlib import Path
from PIL import Image
import io
import argparse
import configparser
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 2. CORE FUNCTIONS

def extract_and_process_images(pdf_path: Path, images_output_dir: Path, target_dpi: int):
    """
    Extracts all images from a PDF file, resamples them to a target DPI, and saves them as PNGs.

    Args:
        pdf_path (Path): Path object for the source PDF file.
        images_output_dir (Path): Directory to save the processed images.
        target_dpi (int): The target resolution for the images. Images will not be upscaled.
    """
    print(f"--- [Step 1] Starting image extraction from {pdf_path.name} ---")
    images_output_dir.mkdir(exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ❌ ERROR: Failed to open PDF file '{pdf_path}'. Error: {e}")
        return

    image_count = 0
    
    # Iterate through each page of the PDF
    for page_num in range(len(doc)):
        # get_page_images(full=True) provides detailed information for each image
        for img_index, img in enumerate(doc.get_page_images(page_num, full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Use Pillow to open and process the image
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Resample the image if a target DPI is set
                if target_dpi > 0:
                    # Get original DPI if available, default to 72
                    original_dpi = pil_image.info.get('dpi', (72, 72)) 
                    scale = target_dpi / original_dpi[0]
                    # Only downscale images to avoid making them blurry
                    if scale < 1: 
                        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                # Define the output path, saving as PNG to preserve quality
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                output_path = images_output_dir / image_filename
                
                pil_image.save(output_path, "PNG")
                image_count += 1
                
            except Exception as e:
                print(f"  ⚠️ WARNING: Could not process image on page {page_num + 1}. Error: {e}")

    print(f"--- [Step 1] Image extraction complete. Found and saved {image_count} images. ---")
    doc.close()


def process_scientific_pdf(pdf_path_str: str, output_dir_str: str, nougat_model_path: str, target_dpi: int):
    """
    The main preprocessing function that orchestrates Nougat OCR and image extraction.

    Args:
        pdf_path_str (str): Path to the input PDF file.
        output_dir_str (str): Path to the directory for saving outputs.
        nougat_model_path (str): Path to the local Nougat model checkpoint.
        target_dpi (int): Target DPI for image processing.
    """
    pdf_file = Path(pdf_path_str)
    output_path = Path(output_dir_str)
    
    if not pdf_file.exists():
        print(f"❌ ERROR: Input PDF file not found at '{pdf_path_str}'")
        sys.exit(1)

    # 1. Create the main output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"--- [Step 1] All outputs will be saved in: {output_path.resolve()} ---")

    # 2. Convert PDF to Markdown using a local Nougat model
    local_checkpoint_path = Path(nougat_model_path)
    if not local_checkpoint_path.exists() or not local_checkpoint_path.is_dir():
        print(f"❌ ERROR: Nougat model folder not found at '{local_checkpoint_path}'")
        print("   Please check the NOUGAT_MODEL_PATH in your config.ini file.")
        sys.exit(1)

    print("--- [Step 1] Calling Nougat for PDF-to-Markdown conversion (this may take a moment)... ---")
    print(f"--- [Step 1] Using local model: {local_checkpoint_path} ---")
    try:
        # Use subprocess to run the nougat command-line tool
        command = [
            "nougat",
            str(pdf_file.resolve()),
            "--out", str(output_path.resolve()),
            "--checkpoint", str(local_checkpoint_path.resolve())
        ]
        
        # Using capture_output=True and check=True is cleaner for error handling
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # This will raise a CalledProcessError if nougat fails
        )
        
        print("--- [Step 1] Nougat processing finished successfully. ---")
        # Rename the output .mmd file to .md for consistency
        mmd_file = output_path / f"{pdf_file.stem}.mmd"
        md_file = output_path / f"{pdf_file.stem}.md"
        if mmd_file.exists():
            mmd_file.rename(md_file)
            print(f"   -> Markdown file saved to: {md_file}")
        else:
            print(f"  ⚠️ WARNING: Expected Nougat output file not found: {mmd_file}")
            
    except FileNotFoundError:
        print("❌ ERROR: 'nougat' command not found.")
        print("   Please ensure you have installed nougat-ocr and the command is in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("❌ ERROR: Nougat failed during execution.")
        print("--- Nougat Error Output ---")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred during Nougat processing: {e}")
        sys.exit(1)
        
    # 3. Extract and process images from the PDF
    images_dir = output_path / "images"
    extract_and_process_images(pdf_file, images_dir, target_dpi)
    
    print("\n✅ Step 1 (Preprocessing) completed successfully!")


# 3. SCRIPT ENTRYPOINT
if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Step 1: Process a PDF paper into Markdown and extract images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output .md and images folder.")
    parser.add_argument("--config", type=str, default='config.ini', help="Path to the configuration file.")
    args = parser.parse_args()

    # Read configuration from the central config file
    config = configparser.ConfigParser()
    if not Path(args.config).exists():
        print(f"❌ ERROR: Configuration file '{args.config}' not found.")
        sys.exit(1)
    config.read(args.config)

    try:
        nougat_path = config.get('PATHS', 'NOUGAT_MODEL_PATH')
        dpi = config.getint('SETTINGS', 'TARGET_DPI')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"❌ ERROR: Missing configuration in '{config_path}': {e}")
        sys.exit(1)

    # Run the main processing function with the provided arguments and config
    process_scientific_pdf(
        pdf_path_str=args.pdf_path,
        output_dir_str=args.output_dir,
        nougat_model_path=nougat_path,
        target_dpi=dpi
    )