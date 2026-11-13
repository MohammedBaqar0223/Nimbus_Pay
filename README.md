# NimbusPay – AI-Powered Financial Support Ticket Triage

## Overview

NimbusPay is an automated financial customer-support ticket classification and prioritization system built using Large Language Models (LLMs).

The project fine-tunes **Qwen2.5-3B-Instruct** to understand customer support messages and generate structured JSON outputs containing ticket categories, priority levels, transaction details, language information, and human-escalation requirements.

The primary objective is to transform unstructured customer complaints into consistent, machine-readable ticket information that can support automated customer-service workflows.

## Key Features

- **Automated Ticket Classification:** Classifies customer queries into predefined financial support categories.
- **Priority Assignment:** Assigns P1, P2, or P3 priorities based on financial impact, fraud reports, and escalation rules.
- **Transaction Information Extraction:** Extracts transaction amounts, dates, IDs, and payment channels.
- **Language Detection:** Identifies English and Hinglish customer messages.
- **Human Escalation Detection:** Determines whether a ticket requires human intervention.
- **Structured JSON Generation:** Produces standardized JSON responses with exactly eight required fields.
- **Dataset Cleaning:** Identifies and corrects inconsistent, malformed, or incomplete training labels.
- **Parameter-Efficient Fine-Tuning:** Uses LoRA and 4-bit QLoRA to fine-tune the base language model.

## Classification Schema

The model generates a JSON object containing the following eight fields:

| Field | Description |
|---|---|
| `category` | Category of the customer complaint |
| `priority` | Ticket priority: P1, P2, or P3 |
| `amount_paise` | Transaction or disputed amount in paise |
| `txn_date` | Transaction date in YYYY-MM-DD format |
| `txn_id` | Normalized transaction ID |
| `channel` | Payment channel |
| `language` | English (`en`) or Hinglish (`hi-en`) |
| `needs_human` | Boolean indicating human intervention requirements |

### Supported Categories

| Category | Description |
|---|---|
| `payment_failed` | Failed, declined, stuck, or incomplete payments |
| `refund` | Refund requests, duplicate charges, and missing refunds |
| `fraud` | Unauthorized or unrecognized transactions |
| `account_access` | Login, password, PIN, and OTP problems |
| `kyc` | Identity verification and documentation issues |
| `offers` | Cashback, coupons, promotional offers, and rewards |
| `other` | Other financial support queries |

### Priority Rules

- **P1 – Critical:** Fraud complaints, disputed amounts of ₹50,000 or more, or legal threats.
- **P2 – High:** Payment failures or refund complaints involving ₹5,000 or more, and account-access issues.
- **P3 – Normal:** All remaining cases.

Human intervention is required for P1 tickets and customers who indicate repeated attempts to contact support.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Model loading and inference |
| Qwen2.5-3B-Instruct | Base language model |
| Hugging Face Datasets | Dataset loading and processing |
| TRL | Supervised fine-tuning |
| PEFT | Parameter-efficient fine-tuning |
| LoRA / QLoRA | Model adaptation |
| BitsAndBytes | 4-bit model quantization |
| Pandas | Dataset manipulation and cleaning |
| Weights & Biases | Training experiment tracking |

## Project Workflow

1. **Dataset Loading:** Load the training dataset in JSONL format.
2. **Dataset Auditing:** Validate existing labels against the required classification schema.
3. **Data Cleaning:** Correct inconsistent priority values, normalize transaction IDs, standardize dates, and remove unparseable JSON outputs.
4. **Baseline Evaluation:** Evaluate the original Qwen2.5-3B-Instruct model on the development dataset.
5. **Rule-Based Prompting:** Introduce detailed classification rules and evaluate the model's zero-shot performance.
6. **Model Fine-Tuning:** Fine-tune Qwen using LoRA and 4-bit QLoRA.
7. **Experimentation:** Compare different training configurations and dataset versions.
8. **Evaluation:** Generate predictions on the development dataset and evaluate them using the provided scoring script.
9. **Final Prediction:** Generate predictions for the test dataset and validate their JSON structure.
10. **Error Analysis:** Compare predicted labels with actual development-set labels to identify classification errors.

## Model Training Configuration

The primary fine-tuning configuration used in the notebook is:

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Fine-Tuning Method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| Learning Rate | 2e-4 |
| Training Epochs | 3 |
| Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Maximum Sequence Length | 512 |
| Optimizer | Paged AdamW 8-bit |
| Learning Rate Scheduler | Cosine |
| Precision | FP16 |

The LoRA adapters are applied to attention and MLP projection layers:

- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

## Experiments

The notebook explores multiple training approaches:

| Experiment | Description |
|---|---|
| Experiment 1 | QLoRA fine-tuning using the cleaned dataset |
| Experiment 2 | QLoRA fine-tuning using the original, uncleaned dataset |
| Experiment 3 | Fine-tuning experiment using a 16-bit model configuration |
| Experiment 4 | QLoRA fine-tuning with two epochs and Weights & Biases tracking |

These experiments are intended to investigate the effects of dataset quality, training precision, and training duration on model performance.

## Dataset Format

The project uses JSONL datasets with conversational message structures.

An example input record:

```json
{
  "id": "sample_001",
  "messages": [
    {
      "role": "system",
      "content": "Ticket triage instructions"
    },
    {
      "role": "user",
      "content": "My UPI payment failed but money was deducted."
    },
    {
      "role": "assistant",
      "content": "{\"category\":\"payment_failed\",\"priority\":\"P2\",\"amount_paise\":null,\"txn_date\":null,\"txn_id\":null,\"channel\":\"upi\",\"language\":\"en\",\"needs_human\":false}"
    }
  ]
}
```

### Expected Dataset Files

- `train.jsonl` – Original training dataset.
- `train_cleaned.jsonl` – Cleaned training dataset.
- `dev.jsonl` – Development dataset used for evaluation.
- `test_inputs.jsonl` – Test tickets requiring predictions.
- `score.py` – Evaluation script supplied with the project.

**Note:** The dataset and scoring script are external project dependencies and are not included in the uploaded notebook.

## Installation

The notebook is designed to run in Google Colab with GPU acceleration.

Install the required libraries:

```bash
pip install -U transformers
pip install -U "bitsandbytes>=0.46.1"
pip install datasets
pip install -q --upgrade trl==0.25.1
pip install peft accelerate
pip install wandb
```

## Running the Project

### 1. Set Up Google Colab

Open the notebook in Google Colab and enable a GPU runtime.

### 2. Upload Project Files

Place the required datasets and scoring script in your Google Drive project directory:

```text
NimbusPay_FT_Project/
│
├── train.jsonl
├── dev.jsonl
├── test_inputs.jsonl
└── score.py
```

Mount Google Drive and copy the files into the Colab environment.

### 3. Execute the Notebook

Run the notebook cells sequentially to:

- Load and audit the training dataset.
- Clean inconsistent training labels.
- Evaluate the baseline model.
- Fine-tune Qwen using QLoRA.
- Run additional experiments.
- Generate and evaluate predictions.

### 4. Generate Test Predictions

The final prediction stage processes `test_inputs.jsonl` and generates:

```text
predictions_test.jsonl
```

Each output record contains a ticket ID and the model-generated JSON classification.

### 5. Validate Predictions

The notebook includes validation checks to verify:

- JSON parsing succeeds.
- All eight required fields are present.
- The generated predictions follow the expected output structure.

## Output Files

| File | Description |
|---|---|
| `train_cleaned.jsonl` | Cleaned training dataset |
| `cleaning_log.csv` | Records dataset corrections and dropped entries |
| `my_dev_predictions.jsonl` | Baseline development predictions |
| `my_secon_dev_predictions.jsonl` | Development predictions after rule-based prompting |
| `exp1_cleaned_qlora.jsonl` | Experiment 1 predictions |
| `exp2_raw_qlora.jsonl` | Experiment 2 predictions |
| `exp3_16bit_lora.jsonl` | Experiment 3 predictions |
| `exp4_lora.jsonl` | Experiment 4 predictions |
| `predictions_test.jsonl` | Final test predictions |
| `adapter/` | Saved LoRA adapter and tokenizer |

## Experiment Tracking

Weights & Biases is integrated into the fourth experiment to monitor training progress and visualize training curves.

The project uses a W&B project named `NIMBUS`.

An API key must be configured securely through Google Colab Secrets using the key name:

```text
WANDB_API_KEY
```

## Limitations and Future Improvements

Potential areas for further development include:

- Improving classification accuracy through systematic error analysis.
- Evaluating performance across individual ticket categories.
- Improving handling of ambiguous and multilingual customer queries.
- Adding more robust JSON validation and structured output enforcement.
- Comparing different quantization and fine-tuning configurations.
- Integrating the trained model into a real-time customer-support application.
- Developing an API for integration with financial customer-support systems.

## Acknowledgements

- [Qwen](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) – Base language model.
- [Weights & Biases](https://wandb.ai/) – Experiment tracking.

---

**Project:** NimbusPay\
**Model:** Qwen2.5-3B-Instruct  
**Approach:** Supervised Fine-Tuning using LoRA / QLoRA
