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
"""Inference code for the pathfinding model with batch generation."""
import argparse
import io
import os
import pickle
from io import BytesIO

import networkx as nx
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.spatial import distance
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

parser = argparse.ArgumentParser(
    description="Inference code for the pathfinding model.")
parser.add_argument(
    "--model_id",
    type=str,
    required=True,
    help="Model endpoint",
)

parser.add_argument(
    "--lora_adapter_path",
    type=str,
    default=None,
    help="Path to lora adapter",
)

parser.add_argument(
    "--delta",
    type=int,
    default=0,
    help="Whether output in the form of delta points.",
)
parser.add_argument(
    "--absolute",
    type=int,
    default=0,
    help="Whether output in the form of absolute coordinates.",
)

parser.add_argument(
    "--precision",
    type=int,
    default=-1,
    help="Whether output in the form of normalized coordinates.",
)

parser.add_argument(
    "--no_marks",
    type=int,
    default=0,
    help="Whether to output the image without the start and end points.",
)

parser.add_argument(
    "--output_path",
    type=str,
    default="./output",
    help="The output path for the inference results.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="The batch size to use for inference.",
)

parser.add_argument(
    "--parquet_file",
    type=str,
    default="/gcs/xcloud-shared/artemispanag/wayfinder_data/mapbench/test-00000-of-00001.parquet",
    help="The MapBench parquet file to use for inference.",
)

parser.add_argument(
    "--annotations_path",
    type=str,
    default="/gcs/xcloud-shared/artemispanag/wayfinder_data/mapbench/annotations",
    help="The annotations path MapBench pickle graphs.",
)

args = parser.parse_args()
BATCH_SIZE = args.batch_size
parquet_file = args.parquet_file
annotations_path = args.annotations_path

if "gemma-3" in args.model_id:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True)
    if args.lora_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        print(f"Loaded LoRa adapter from {args.lora_adapter_path}")
    processor = AutoProcessor.from_pretrained(args.model_id,
                                              local_files_only=True)
    processor.tokenizer.padding_side = "left"

if "qwen2.5" in args.model_id.lower():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto")
    if args.lora_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        print(f"Loaded LoRa adapter from {args.lora_adapter_path}")
    # default processer
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

euclidean = distance.euclidean

PROMPT = (
    "You are provided an image of a path with a start location denoted in green and an end location denoted in red.\n",
    "The normalized xy-coordinates of the start location are {} and of the end location {}.\n",
    "Output a list of normalized coordinates in the form of a list [(x1,y1), (x2,y2)...] of the path between the start and end location.\n",
    "Ensure that the path follows the traversable locations of the map.",
)
if args.delta:
    PROMPT = [
        "You are provided an image of a path with a start location denoted in green and an end location denoted in red.\n",
        "The normalized xy-coordinates of the start location are {} and of the end location {}.\n",
        "Output a list of (dx, dy) moves in the form of a list [(dx1,dy1), (dx2,dy2)...] from the start to the end marker.\n",
        "Ensure that the generated path follows the traversable locations of the map.",
    ]
if args.absolute:
    PROMPT = [
        "You are provided an image of a path with a start location denoted in green and an end location denoted in red.\n",
        "The xy-coordinates of the start location are {} and of the end location {}.\n",
        "Output a list of coordinates in the form of a list [(x1,y1), (x2,y2)...] of the path between the start and end location.\n",
        "Ensure that the generated path follows the traversable locations of the map.",
    ]


def resize_image_to_max_bytes(
    image: Image.Image,
    max_bytes: int = 25245000,
    image_format: str = "PNG",
    quality: int = 95,
) -> Image.Image:
    """Resizes a PIL Image to be at most max_bytes.

    This function attempts to find the largest possible image dimensions that
    result
    in a file size under the specified byte limit, while preserving the aspect
    ratio.
    It uses a binary search on the image's scale factor for efficiency.

    Args:
        image (Image.Image): The input PIL Image object.
        max_bytes (int): The maximum desired file size in bytes.
        image_format (str): The target image format (e.g., "JPEG", "PNG"). JPEG is
          recommended for photos to get better compression.
        quality (int): The quality for JPEG compression (1-95). Higher is better.
          This is ignored for formats like PNG.

    Returns:
        Image.Image: A new PIL Image object that is resized, or the original
                     image if it's already within the byte limit.
    """
    # Save the image to an in-memory buffer
    buffer = io.BytesIO()

    # When saving JPEG, 'quality' is a key parameter for size.
    # For other formats like PNG, options might be different (e.g., "optimize").
    save_kwargs = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = quality

    image.save(buffer, **save_kwargs)
    initial_size = buffer.tell()

    # If the image is already within the limit, return it
    if initial_size <= max_bytes:
        return image

    # --- Start binary search for the best scale factor ---
    low = 0.0
    high = 1.0
    best_image = image  # Fallback to the original if something goes wrong

    # Iterate a few times to find the optimal scale
    for _ in range(10):  # 10 iterations are usually enough to converge
        scale = (low + high) / 2

        # Calculate new dimensions
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        if new_width == 0 or new_height == 0:
            # Dimensions are too small, stop searching in this direction
            high = scale
            continue

        # Resize the image
        # Image.Resampling.LANCZOS is a high-quality downsampling filter
        resized_image = image.resize((new_width, new_height),
                                     Image.Resampling.LANCZOS)

        # Check the size of the resized image
        buffer = io.BytesIO()
        resized_image.save(buffer, **save_kwargs)
        current_size = buffer.tell()

        if current_size <= max_bytes:
            # This is a potential candidate, try for a slightly larger size
            best_image = resized_image
            low = scale
        else:
            # Too big, need to reduce the scale
            high = scale

    return best_image


def normalize_coordinates(coords, dims):
    """Normalizes a list of coordinates to a [0, 1] frame."""
    width, height = dims
    return [[p[0] / width, p[1] / height] for p in coords]


def calculate_dtw(s1, s2, dist_fn=euclidean):
    """
    Calculates the Dynamic Time Warping (DTW) distance and warping path
    between two sequences.

    Args:
      s1: The first sequence (e.g., a NumPy array or list).
      s2: The second sequence (e.g., a NumPy array or list).
      dist_fn: A function to calculate the distance between two points.
               Defaults to the Euclidean distance.

    Returns:
      A tuple containing:
        - distance (float): The DTW distance between the two sequences.
        - path (list): The optimal warping path, a list of (s1_index, s2_index) tuples.
    """
    n = len(s1)
    m = len(s2)

    # Create a cost matrix (dtw_matrix) and initialize it with infinity
    dtw_matrix = np.full((n + 1, m + 1), float("inf"))
    dtw_matrix[0, 0] = 0

    # Fill the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_fn(s1[i - 1], s2[j - 1])
            # The cost is the distance plus the minimum of the costs of the adjacent cells
            last_min = min(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1],
            )  # Match
            dtw_matrix[i, j] = cost + last_min

    # The DTW distance is the value in the bottom-right cell
    distance = dtw_matrix[n, m]

    # --- Backtrack to find the optimal path ---
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        # Find the direction of the minimum cost path
        min_cost_idx = np.argmin([
            dtw_matrix[i - 1, j],  # Insertion
            dtw_matrix[i, j - 1],  # Deletion
            dtw_matrix[i - 1, j - 1],
        ])  # Match

        if min_cost_idx == 0:
            i -= 1
        elif min_cost_idx == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    # The path is constructed backwards, so we reverse it
    path.reverse()

    return distance, path

def analyze_dtw(sequence1, sequence2):
    """Performs DTW analysis on two coordinate sequences.

    The input sequences are first normalized to be scale-invariant.

    Args:
        sequence1 (numpy.ndarray): The first sequence of coordinates (N, D).
        sequence2 (numpy.ndarray): The second sequence of coordinates (M, D).

    Returns:
        dict: A dictionary containing the calculated DTW metrics.
    """
    s1 = np.array(sequence1, dtype=np.double)
    s2 = np.array(sequence2, dtype=np.double)
    if len(s2.shape) == 3:
        s2 = s2[0]

    if s1.ndim == 1:
        s1 = s1.reshape(-1, 1)
    if s2.ndim == 1:
        s2 = s2.reshape(-1, 1)

    # Return empty metrics if a sequence is too short to normalize or empty
    if s1.shape[0] < 2 or s2.shape[0] < 2:
        return {
            "dtw_distance": float("inf"),
            "path_length": 0,
            "path_start_indices": None,
            "path_end_indices": None,
        }

    distance, path = calculate_dtw(s1, s2, dist_fn=euclidean)

    path_length = len(path)

    metrics = {
        "dtw_distance":
        distance,
        "path_length":
        path_length,
        "path_start_indices":
        path[0],
        "path_end_indices":
        path[-1]
    }
    return metrics


def bytes_to_pil(bytes_str):
    image = Image.open(BytesIO(bytes_str)).convert("RGB")
    return image


def pil_to_bytes(pil_image):
    with BytesIO() as output:
        pil_image.save(output, format="PNG")
        return output.getvalue()


def generate_response(
    prompts: list[str],
    image_data: list[bytes],
    model_id: str = None,
) -> list[str]:
    """Generates a batch of responses from the Gemma model."""
    images = [[bytes_to_pil(im)] for im in image_data]
    if "gemma" in args.model_id:
        messages = [[{
            "role":
            "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": p
                },
            ],
        }] for p in prompts]

        # Preparation for batch inference
        texts = [
            processor.apply_chat_template(msg,
                                          tokenize=False,
                                          add_generation_prompt=True)
            for msg in messages
        ]

        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            pad_to_multiple_of=8,
            padding="max_length",
            max_length=2048,
            truncation=True,
        ).to(model.device)
        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=2048)

        # Batch decode returns a list of responses
        responses = processor.batch_decode(generate_ids,
                                           skip_special_tokens=True)

        # Clean the prompt from the response string
        cleaned_responses = [res.split("model\n")[-1] for res in responses]

    if "qwen2.5" in args.model_id.lower():
        # Sample messages for batch inference
        messages = [[{
            "role":
            "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": p
                },
            ],
        }] for p in prompts]

        # Preparation for batch inference
        texts = [
            processor.apply_chat_template(msg,
                                          tokenize=False,
                                          add_generation_prompt=True)
            for msg in messages
        ]
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Batch Inference
        generated_ids = model.generate(**inputs, max_new_tokens=6048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        cleaned_responses = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

    return cleaned_responses


def draw_start_end_points(img, start, end):
    """Draws start and end points on a copy of the given image.

    The start point is marked in green and the end point in red.

    Args:
      img: The PIL Image object to draw on.
      start: A tuple (x, y) representing the start coordinates.
      end: A tuple (x, y) representing the end coordinates.

    Returns:
      A new PIL Image object with the start and end points drawn.
    """
    curr_img = img.copy()
    draw = ImageDraw.Draw(curr_img)
    radius = 10
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius,
         start[1] + radius),
        fill="green",
    )
    draw.ellipse(
        (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
        fill="red")
    return curr_img


def plot_path(img, path):
    """Plots a path on a copy of the given image.

    Args:
      img: The PIL Image object to draw on.
      path: A list of tuples (x, y) representing the path coordinates.

    Returns:
      A new PIL Image object with the path plotted.
    """
    curr_img = img.copy()
    draw = ImageDraw.Draw(curr_img)
    for i in range(len(path) - 1):
        draw.line([path[i], path[i + 1]], fill="blue", width=4)
    return curr_img


def parse_response(response):
    """Parses a response from the Gemini model."""
    if "[" in response:
        response = response.split("[")[-1]
        if "]" in response:
            response = response.split("]")[0]
    response = eval(f"[{response.strip()}]")
    if isinstance(response[0], list) or isinstance(response[0], tuple):
        path = response
    elif isinstance(response[0], dict):
        keys = list(response[0].keys())
        path = [(r[keys[0]], r[keys[1]]) for r in response]
    else:
        raise ValueError(f"Invalid response: {response}")
    return path


def save_print_results(all_results, total_metrics, count, output_filename):
    """Saves the results to a file and prints the metrics."""
    print(f"\n--- Results Summary (Processed {count} examples) ---")
    if count == 0:
        print("No valid results to summarize.")
        return

    print(
        f"Average DTW distance: {total_metrics['dtw_distance']/count:.2f}")

    with open(output_filename, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {output_filename}")

all_results = []
total_metrics = {
    "dtw_distance": 0,
}


def process_batch(batch_tasks):
    """Processes a batch of query items."""
    prepared_items = []
    prompts = []
    images = []

    for item in batch_tasks:
        i, row, query_idx, query = item
        try:
            image = bytes_to_pil(row["image"]["bytes"])
        except Exception as e:
            print(
                f"Warning: Could not load image for task {query_idx}. Skipping. Error: {e}"
            )
            continue

        map_class = row["map_class"]
        width, height = image.size
        image_id = row["image_id"]

        try:
            graph = pickle.load(
                open(
                    os.path.join(annotations_path,
                                 f"{map_class}/{image_id}.pkl"), "rb"))
            label2coords = {
                node[1]["label"]: node[0]
                for node in graph.nodes(data=True)
                if "label" in node[1]
            }

            start_loc = label2coords[query["start"]]
            start_loc_normalized = (start_loc[0] / width,
                                    start_loc[1] / height)
            end_loc = label2coords[query["destination"]]
            end_loc_normalized = (end_loc[0] / width, end_loc[1] / height)

            ground_truth_path = nx.shortest_path(graph, start_loc, end_loc)
        except (KeyError, nx.NetworkXNoPath, FileNotFoundError) as e:
            print(
                f"Warning: Could not find path for {image_id}, query {query_idx}. Skipping. Error: {e}"
            )
            continue
        print("Found path! ")

        curr_img = image if args.no_marks else draw_start_end_points(
            image, start_loc, end_loc)

        normalized_ground_truth_path = [
            (p[0] / width, p[1] / height) for p in ground_truth_path
        ]
        resized_image = resize_image_to_max_bytes(curr_img, max_bytes=2524500)

        if args.precision > 0:
            start_loc = (round(start_loc[0], args.precision),
                         round(start_loc[1], args.precision))
            end_loc = (round(end_loc[0], args.precision),
                       round(end_loc[1], args.precision))
            start_loc_normalized = (
                round(start_loc_normalized[0], args.precision),
                round(start_loc_normalized[1], args.precision),
            )
            end_loc_normalized = (
                round(end_loc_normalized[0], args.precision),
                round(end_loc_normalized[1], args.precision),
            )
        if args.absolute:
            curr_prompt = "".join(PROMPT).format(start_loc, end_loc)
        else:
            curr_prompt = "".join(PROMPT).format(start_loc_normalized,
                                                end_loc_normalized)

        # Append prepared data for batch processing
        prepared_items.append({
            "image_id": image_id,
            "map_class": map_class,
            "query": query,
            "query_idx": query_idx,
            "normalized_ground_truth_path": normalized_ground_truth_path,
            "resized_image_size": resized_image.size,
        })
        prompts.append(curr_prompt)
        images.append(pil_to_bytes(resized_image))

    if not prepared_items:
        return []

    # Generate responses for the entire valid batch
    try:
        responses = generate_response(prompts=prompts,
                                      image_data=images,
                                      model_id=args.model_id)
    except Exception as e:
        print(f"Error during batch generation: {e}")
        return []

    batch_results = []
    for i, response in enumerate(responses):
        item_data = prepared_items[i]
        try:
            parsed_response = parse_response(response)

            if args.delta:
                start_loc = item_data["normalized_ground_truth_path"][0]
                end_loc = item_data["normalized_ground_truth_path"][-1]
                path = [start_loc]
                for move in parsed_response:
                    new_loc = (path[-1][0] + move[0], path[-1][1] + move[1])
                    path.append(new_loc)
                path.append(end_loc)
                parsed_response = path

            if args.absolute:
                img_width, img_height = item_data["resized_image_size"]
                parsed_response = [
                    (p[0] / img_width, p[1] / img_height)
                    for p in parsed_response
                ]

            dtw_metrics = analyze_dtw(
                item_data["normalized_ground_truth_path"], parsed_response)

            result = {
                "image_id":
                item_data["image_id"],
                "map_class":
                item_data["map_class"],
                "query":
                item_data["query"],
                "query_idx":
                item_data["query_idx"],
                "response":
                parsed_response,
                "ground_truth_path":
                item_data["normalized_ground_truth_path"],
                "dtw_metrics":
                dtw_metrics,
            }
            batch_results.append(result)
        except Exception as e:
            print(
                f"Error post-processing response for {item_data['image_id']}, query {item_data['query_idx']}. Error: {e}"
            )
            continue

    return batch_results


if __name__ == "__main__":
    dataset = pd.read_parquet(parquet_file)

    results = []
    output_path = args.output_path
    if os.path.exists(output_path):
        try:
            with open(output_path, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded {len(results)} existing results from {output_path}")
        except (pickle.UnpicklingError, EOFError):
            print(
                f"Warning: Could not read existing results file at {output_path}. Starting fresh."
            )
            results = []

    # Filter out bad results and create a set of already processed tasks
    exists = set()
    valid_results = []
    for res in results:
        if res and res.get("dtw_metrics", {}).get("dtw_distance",
                                                  float("inf")) <= 10000:
            valid_results.append(res)
            exists.add(
                tuple((res["image_id"], res["query"]["start"],
                       res["query"]["destination"])))
    results = valid_results
    print(f"{len(exists)} tasks already completed and will be skipped.")

    tasks = []
    for i, row in dataset.iterrows():
        for idx, query in enumerate(row["queries"]):
            if tuple(
                (row["image_id"], query["start"], query["destination"])) in exists:
                continue
            tasks.append((i, row, idx, query))

    print(f"Total new tasks to process: {len(tasks)}")

    results_new = []
    # Use tqdm for a progress bar
    with tqdm(total=len(tasks), desc="Processing Batches") as pbar:
        for i in range(0, len(tasks), BATCH_SIZE):
            batch_tasks = tasks[i:i + BATCH_SIZE]
            batch_results = process_batch(batch_tasks)

            if batch_results:
                results_new.extend(batch_results)

            pbar.update(len(batch_tasks))

            # Save intermediate results periodically
            if i // BATCH_SIZE % 10 == 1 and i > 0:
                current_results = results + results_new
                count = len(current_results)
                if count > 0:
                    # Recalculate total_metrics
                    temp_total_metrics = {k: 0 for k in total_metrics}
                    for res in current_results:
                        for key in temp_total_metrics:
                            if (key in res["dtw_metrics"] and
                                    res["dtw_metrics"][key] < float("inf")):
                                temp_total_metrics[key] += res["dtw_metrics"][key]
                    save_print_results(current_results, temp_total_metrics,
                                       count, output_path)

    # Final processing and saving
    all_results = results + results_new
    count = len(all_results)
    print(f"\nFinished processing. Total valid results: {count}")

    if count > 0:
        # Recalculate final metrics
        final_total_metrics = {k: 0 for k in total_metrics}
        for res in all_results:
            dtw_metrics = res["dtw_metrics"]
            for key in final_total_metrics:
                if key in dtw_metrics and dtw_metrics[key] < float("inf"):
                    final_total_metrics[key] += dtw_metrics[key]
                else:
                    print(
                        f"Warning: Metric '{key}' is inf for image {res['image_id']}. Skipping."
                    )

        save_print_results(all_results, final_total_metrics, count,
                           output_path)
    else:
        print("No new results were generated.")