import os,sys
import logging
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(message)s')

path = "data/2/"

def convert_single_file(heic_path, jpg_path, output_quality) -> tuple:
    """
    Convert a single HEIC file to JPG format.
    
    #### Args:
        - heic_path (str): Path to the HEIC file.
        - jpg_path (str): Path to save the converted JPG file.
        - output_quality (int): Quality of the output JPG image.

    #### Returns:
        - tuple: Path to the HEIC file and conversion status.
    """
    try:
        with Image.open(heic_path) as image:
            image.save(jpg_path, "JPEG", quality=output_quality)
        # Preserve the original access and modification timestamps
        heic_stat = os.stat(heic_path)
        os.utime(jpg_path, (heic_stat.st_atime, heic_stat.st_mtime))
        return heic_path, True  # Successful conversion
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logging.error("Error converting '%s': %s", heic_path, e)
        return heic_path, False  # Failed conversion

#register_heif_opener()
if not os.path.isdir(path):
   logging.error("Directory '%s' does not exist.", path)

heic_files = [file for file in os.listdir(path) if file.lower().endswith("heic")]
total_files = len(heic_files)

jpg_dir = os.path.join(path, "ConvertedFiles")
os.makedirs(jpg_dir, exist_ok=True)



tasks = []
for file_name in heic_files:
    heic_path = os.path.join(path, file_name)
    jpg_path = os.path.join(jpg_dir, os.path.splitext(file_name)[0] + ".jpg")
    
    #skip conversion if the JPG already exists
    if os.path.exists(jpg_path):
        logging.info("Skipping '%s' as the JPG already exists.", file_name)
        continue

    tasks.append((heic_path, jpg_path))

print(tasks)

    #Convert HEIC files to JPG in parallel using ThreadPoolExecutor
num_converted = 0

with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_file = {
        executor.submit(convert_single_file, heic_path, jpg_path, output_quality=50): heic_path
        for heic_path, jpg_path in tasks
    }

    for future in as_completed(future_to_file):
        heic_file = future_to_file[future]
        try:
            _, success = future.result()
            if success:
                num_converted += 1

            # Display progress
            progress = int((num_converted / total_files) * 100)
            print(f"Conversion progress: {progress}%", end="\r", flush=True)
        except Exception as e:
            logging.error("Error occured during conversion of '%s' %s", heic_file, e)

print(f"\nConversion completed successfully. {num_converted} files converted.")


    
