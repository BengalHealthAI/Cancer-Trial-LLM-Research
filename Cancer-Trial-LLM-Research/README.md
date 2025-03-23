# Cancer-Trial-LLM-Research

This repository contains code for fine-tuning the LLaMA-2 model for classifying clinical trial eligibility criteria using QLoRA. The project is organized into modules for data processing, training, inference, and evaluation.

## Project Structure
- `src/data_processing.py`: Data loading and transformation.
- `src/train.py`: Model training using QLoRA.
- `src/predict.py`: Model inference/prediction.
- `src/metrics.py`: Functions to compute evaluation metrics.
- `notebooks/`: Jupyter/Colab notebooks for exploratory analysis.
- `requirements.txt`: Python dependencies.

## Installation
1. Clone the repository: 
    - git clone https://github.com/BengalHealthAI/Cancer-Trial-LLM-Research.git
    - Change directory to Cancer-Trial-LLM-Research
2. Create a virtual environment and install dependencies:
    - pip install -r requirements.txt

## Usage
- Run training: python src/train.py
- Run inference: python src/predict.py --input "Your sample input text here"
- Evaluate predictions: python src/metrics.py --predictions path/to/predictions.pkl
