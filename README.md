# MapTrace: A 2M-Sample Synthetic Dataset for Wayfinding Path Tracing

This repository contains the dataset and code for **MapTrace**, a large-scale synthetic dataset for training multimodal models to trace paths on wayfinding maps.

**Paper:** "MapTrace: Scalable Data Generation for Route Tracing on Maps"

**Paper Link:** [here](TBD)

-----

## Motivation

<div>
<p align="center">
<img src="assets/maptrace_overview_v2.png" width="600px">
</p>
</div>

1.  **The Problem:** State-of-the-art multimodal large language models (MLLMs) struggle to parse the **semantic constraints** of wayfinding maps, such as tracing a valid route from a start to an end point.
2.  **The Cause:** This gap is likely due to the lack of large-scale, high-quality supervision for this specific task.
3.  **The Challenge:** Collecting this data manually is expensive and slow. It requires access to a large, diverse set of maps and laborious, pixel-level human annotations.
4.  **Our Solution:** We developed a synthetic data generation pipeline that:
      * Uses pixel colors to **propose candidate paths**.
      * Employs MLLMs as **critics** to validate the semantic viability of these paths.

This automated pipeline allowed us to generate **1 million high-quality training samples**.

-----

## Dataset Format

<div>
<p align="center">
<img src="assets/qualitative_examples_map_trace.png" width="600px">
</p>
</div>

The MapTrace dataset contains 2M annotated paths designed to train models on route-tracing tasks. The data consists of a map image, a text-based query, and a corresponding list of normalized coordinates defining the correct path.

The dataset is provided in two distinct parts:

  * **`maptrace_parquet`**: Contains paths on more complex, stylized maps, such as those found in brochures, park directories, or shopping malls.
  * **`floormap_parquet`**: Contains paths on simpler, structured floor maps, typical of office buildings, apartment complexes, or campus maps.
  * **`maptrace_20k`**: Contains the path of the initial dataset used in the paper. The data is already postprocessed and do not require further processing for training.

The full dataset is hosted on Huggingface in Parquet format. Access it [here](https://huggingface.co/datasets/google/MapTrace/). The total size of the dataset including images and text is 210GB.


### Schema

Each row in the Parquet files contains the following fields:
  * `image_bytes`: The raw bytes of the generated map image (without post-processing).
  * `label_text`: A string representation of a list of coordinates defining the target path (e.g., `[(0.1, 0.2), (0.1, 0.3), ...]`). All coordinates are normalized between 0.0 and 1.0.
  * `input_text`: A natural language question (prompt) asking the model to find the path specified in `label_text` (e.g., "Show me how to get from the lobby to Conference Room B.").
  * `map_description`: A natural language description of the map image. Used by the text-to-image generation to create the synthetic image.

-----

## Code

### Setup

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install pytorch:**
Ensure that you use the CUDA version compatible with your environment. Experiments were conducted with CUDA 12.4.

    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r src/requirements.txt
    ```

### Code Structure

- `src/`
  - `postprocess_data.py` — Utility to preprocess raw dataset folders into finetuning-ready Parquet files (postprocessing, cleaning, and formatting).
  - `finetune_gemma27b.py` — Example finetuning script for Gemma 2.7B. Demonstrates dataset loading, training loop configuration, and integration with Accelerate/FSDP.
  - `hf_inference.py` — Inference script using Hugging Face Transformers to evaluate finetuned models on MapBench.
  - `vllm_inference.py` — Inference script using vLLM to evaluate finetuned models on MapBench.
  - `requirements.txt` — Python dependencies used by the scripts.

Example: finetuning command (using accelerate)
```bash
accelerate launch --config_file fsdp_config.yaml src/finetune_gemma27b.py \
  --output_dir /path/to/output \
  --train_path_dir /path/to/processed/training/data
```

Example: vLLM inference
```bash
python src/vllm_inference.py \
  --model_id /hf/model/id \
  --lora_adapter_path /path/to/ft/model \
  --precision 4 \
  --output_path /path/to/output/directory \
  --parquet_file /path/to/mapbench/data \
  --annotations_path /path/to/mapbench/graph/folder
```

### Usage Instructions

* Download the data from [here](https://huggingface.co/datasets/google/MapTrace/)
* Post process each folder (maptrace, and floormaps) separately using `postprocess_data.py`.
* In `finetune_gemma27b.py` update the training path directory. It should include all **postprocessed** parquet files you want to train on.
* Update the `fsdp_config.yaml` with your sharding configuration. See details [here](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp)
* Run finetuning as shown above. 

## Citation

If you use the MapTrace dataset or code in your research, please cite our paper:

```bibtex
@inproceedings{TBD}
```

```
This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
```
