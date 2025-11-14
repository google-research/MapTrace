# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script preprocesses image and text data from parquet files."""
import argparse
import os
import pandas as pd
import io
import re
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry import LineString
import traceback

parser = argparse.ArgumentParser(description="Preprocess MapTrace data.")
parser.add_argument(
    "--train_path_dir",
    default="/home/jupyter/shared/MapTrace_data",
    help="Path to the directory containing the training data.",
)
parser.add_argument(
    "--output_dir",
    default="/home/jupyter/shared/MapTrace_data_PROCESSED",
    help="Path to the directory where the processed data will be saved.",
)
parser.add_argument(
    "--draw_start_end",
    default=True,
    help="Whether to draw start/end dots on the processed image.",
)
parser.add_argument(
    "--num_workers",
    default=90,
    help="Number of parallel workers to use.",
)
args = parser.parse_args()

TRAIN_PATH_DIR = args.train_path_dir
OUTPUT_DIR = args.output_dir
DRAW_START_END = args.draw_start_end
NUM_WORKERS = args.num_workers

def _round_coord_match(match):
    """Rounds coordinates in a matched string."""
    try:
        x = float(match.group(1))
        y = float(match.group(2))
        return f"({x:.4f}, {y:.4f})"
    except Exception:
        return match.group(0)

def round_coords_in_text(text):
    """Finds and rounds all coordinate tuples in a block of text."""
    return re.sub(r"\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)", _round_coord_match, text)

def draw_dots_on_image(image, coord1, coord2, radius=20):
    """Draws green (start) and red (end) dots on an image."""
    img_with_dots = image.convert("RGB")
    draw = ImageDraw.Draw(img_with_dots)
    width, height = img_with_dots.size

    # Draw start dot (green)
    px1 = coord1[0] * width
    py1 = coord1[1] * height
    bbox1 = (px1 - radius, py1 - radius, px1 + radius, py1 + radius)
    draw.ellipse(bbox1, fill="green")

    # Draw end dot (red)
    px2 = coord2[0] * width
    py2 = coord2[1] * height
    bbox2 = (px2 - radius, py2 - radius, px2 + radius, py2 + radius)
    draw.ellipse(bbox2, fill="red")
    return img_with_dots

def smooth_path(path: list[tuple[int, int]], epsilon: float = 0.005) -> list[tuple[int, int]]:
  """Simplifies a path using the Ramer-Douglas-Peucker algorithm."""
  if not path:
    return []

  def perpendicular_distance(point, line_start, line_end):
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    if np.all(a == b):
      return np.linalg.norm(p - a)
    line_vec = b - a
    line_len_sq = np.sum(line_vec**2)
    t = np.dot(p - a, line_vec) / line_len_sq
    if t < 0:
      return np.linalg.norm(p - a)
    if t > 1:
      return np.linalg.norm(p - b)
    projection = a + t * line_vec
    return np.linalg.norm(p - projection)

  dmax = 0
  index = 0
  end = len(path) - 1
  for i in range(1, end):
    d = perpendicular_distance(path[i], path[0], path[end])
    if d > dmax:
      index = i
      dmax = d
  if dmax > epsilon:
    results1 = smooth_path(path[:index + 1], epsilon)
    results2 = smooth_path(path[index:], epsilon)
    return results1[:-1] + results2
  else:
    return [path[0], path[end]]


def create_processed_batch(samples):
    """
    Processes a batch of data (from one file) and returns it in a
    Parquet-friendly format.

    'samples' is a dictionary of lists, e.g., samples['image_bytes'] is a [list of <image_bytes>]
    """
    batch_processed_images = []
    batch_messages = []

    for img_bytes, label_str, text_input in zip(samples['image_bytes'], samples['label_text'], samples['input_text']):
        label = ast.literal_eval(label_str)
        img = Image.open(io.BytesIO(img_bytes))
        if DRAW_START_END and label:
            img = draw_dots_on_image(img, label[0], label[-1])

        smooth_path_list = []
        if label:
            epsilon = 0.005
            line = LineString(label)
            while True:
                simplified_line = line.simplify(epsilon, preserve_topology=False)
                smooth_path_list = [(round(x, 4), round(y, 4)) for x, y in simplified_line.coords]
                if len(smooth_path_list) <= 35:
                    break
                epsilon *= 1.5

        with io.BytesIO() as img_byte_arr:
            img.save(img_byte_arr, format='PNG')
            processed_img_bytes = img_byte_arr.getvalue()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": round_coords_in_text(text_input)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(smooth_path_list)}],
            },
        ]

        batch_processed_images.append(processed_img_bytes)
        batch_messages.append(messages)

    return {
        "image": batch_processed_images,
        "messages": batch_messages
    }


def process_file(input_filepath):
    """
    Worker function to process a single parquet file.
    Reads the file, processes its contents, and saves a new file.
    """
    try:
        df = pd.read_parquet(input_filepath)

        batch_dict = {
            'image_bytes': df['image_bytes'].tolist(),
            'label_text': df['label_text'].tolist(),
            'input_text': df['input_text'].tolist()
        }

        processed_data = create_processed_batch(batch_dict)

        processed_df = pd.DataFrame(processed_data)

        output_filename = os.path.basename(input_filepath)
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        processed_df.to_parquet(output_filepath, index=False)

        return output_filepath
    except Exception as e:
        print(f"Error processing file {input_filepath}: {e}")
        traceback.print_exc()
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        all_files = [
            os.path.join(TRAIN_PATH_DIR, f)
            for f in os.listdir(TRAIN_PATH_DIR)
            if f.endswith('.parquet')
        ]
        if not all_files:
            print(f"No .parquet files found in {TRAIN_PATH_DIR}. Exiting.")
            return
        print(f"Found {len(all_files)} parquet files to process.")
    except FileNotFoundError:
        print(f"Input directory not found: {TRAIN_PATH_DIR}. Exiting.")
        return

    print(f"Starting processing with {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file, filepath): filepath for filepath in all_files}

        print("Processing files...")
        for future in tqdm(as_completed(futures), total=len(all_files), desc="Processing files"):
            filepath = futures[future]
            try:
                result = future.result()
                if result is None:
                    print(f"Failed to process {filepath}")
            except Exception as e:
                print(f"An exception occurred while processing {filepath}: {e}")
                traceback.print_exc()

    print(f"--- All processing complete ---")
    print(f"Processed files are saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()