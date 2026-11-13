# NimbusPay Support Ticket Triage — Fine-Tuning Report

**Model:** `Qwen/Qwen2.5-3B-Instruct`
**Hardware:** Google Colab Free Tier (NVIDIA T4 15 GB)  
**Best Submitted Result (Exp 1):** **92.0% Exact Match** | **98.7% Mean Field Accuracy** | **100.0% Valid JSON** on `dev.jsonl`

---

## 1. Part A: Data Audit & Cleaning (`train.jsonl`)

Auditing the 1,445 rows in `train.jsonl` against `SCHEMA.md` using a custom schema-validation script revealed 5 distinct failure modes across 145 corrupted rows:

| Issue Found | Count | Action | Resolution |
|---|---:|---|---|
| **unparseable JSON** (e.g., `tr-00060`, `tr-00095`) | 25 | `drop` | Dropped because partial labels cannot be reconstructed without guessing. |
| **Wrong key & non-spec values** (`"prio": "low"/"medium"` instead of `"priority"`) | 70 | `fix` | Renamed `"prio"` to `"priority"` and mapped `low` -> `P3`, `medium` -> `P2`, `high` -> `P1`. |
| **Non-ISO date format** (`DD/MM/YYYY`, e.g., `02/09/2026`) | 30 | `fix` | Converted to `YYYY-MM-DD` per spec. |
| **Unnormalized `txn_id`** (hyphens or lowercase, e.g., `NP-9718716109`) | 20 | `fix` | Stripped spaces/hyphens and converted to uppercase (`NP9718716109`). |
| **Business rule violation** (`priority == "P1"` with `needs_human == false`) | 30 | `fix` | Updated `needs_human` to `true` per spec (`true` if `priority` is `P1`). |

---

## 2. Part B & C: Baselines & Training Pipeline

* **Stack:** Hugging Face `transformers`, `peft`, `trl` (`SFTTrainer`), and `bitsandbytes`
* **LoRA Configuration:** `r=16`, `lora_alpha=32`, `lora_dropout=0.1`, targeting attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), keeping trainable parameters well below the 50M ceiling.
* **Sequence Length (`max_length=512`):** Length analysis of `train.jsonl` showed 95% of tickets are under 85 words, while ~2–3% contain long quoted email threads. `max_length=512` covers ~98% of tickets without truncation while fitting T4 VRAM.
* **Inference:** Greedy decoding (`do_sample=False`, `max_new_tokens=200`), stripping whitespace only.

---

## 3. Part D: Controlled Experiments

| Experiment | What Changed | Valid JSON | Dev Exact Match | Dev Mean Field Acc | Peak VRAM | Training Time |
|---|---|---:|---:|---:|---:|---:|
| **Baseline 1** | Base model, fixed prompt (no rules) | 65.5% | 0.0% | 0.1% | — | — |
| **Baseline 2** | Base model + `SCHEMA.md` rules in prompt | 100.0% | 7.5% | 70.6% | — | — |
| **Exp 1 (Submitted)** | **Cleaned data (`train_cleaned.jsonl`)**, 4-bit QLoRA, 3 epochs | 100.0% | **92.0%** | **98.7%** | ~8.88 GB | 35.0 min |
| **Exp 2** | **Raw `train.jsonl`** (vs. cleaned data), 4-bit QLoRA, 3 epochs | 100.0% | 93.0% | 98.8% | 8.88 GB | 41.8 min |
| **Exp 3** | Repeat 4-bit QLoRA run on cleaned data (`quant_config` retained) | 100.0% | 94.0% | 98.9% | ~8.88 GB | 35.5 min |
| **Exp 4** | **2 epochs** (vs. 3 epochs), 4-bit QLoRA + W&B logging | 100.0% | 84.0% | 97.3% | ~8.88 GB | 31.0 min |

**Key Takeaways:**
1. **Prompting vs. Fine-Tuning:** Putting rules in the prompt (Baseline 2) fixed JSON formatting (100% valid) and lifted field accuracy to 70.6%, but failed on multi-rule interactions (`priority` 40.0%, `fraud` 0.0% EM). Fine-tuning jumped Exact Match from **7.5% to 92.0–94.0%**.
2. **Cleaned vs. Raw Data (Exp 1 vs. Exp 2):** Raw data scored 93.0% vs. 92.0% for cleaned data—a 2-ticket (1.0%) difference within normal run-to-run variance. Because ~90% of `train.jsonl` was already clean, 3 epochs allowed the model to ignore minority noise, though cleaned data still outperformed raw on `txn_date` (99.0% vs. 98.5%).
3. **Training Duration (Exp 1 vs. Exp 4):** Reducing epochs from 3 to 2 dropped Exact Match from **92.0% to 84.0%**, and `priority` dropped from 94.5% to 86.0% and `needs_human` dropping from 97.0% to 95.0%.

---

## 4. Part E: Error Analysis (16 / 200 Failures on `dev.jsonl`)

On `dev.jsonl`, Exp 1 achieved **100.0%** on `amount_paise`, `channel`, and `language`, and **99.5%** on `category` and `txn_id`. The weakest fields were **`priority` (94.5%)** and **`needs_human` (97.0%)**.

| Ticket ID | Ticket Snippet / Issue | Predicted vs. Actual | Explanation |
|---|---|---|---|
| **`dv-00006`** | Received `2026-07-02`, *"day before yesterday"* | `txn_date`: `'2026-07-01'` vs `'2026-06-30'` | Off-by-one error across the June/July month boundary. |
| **`dv-00011`** | Refund for `₹84,137` | `priority`: `'P2'` vs `'P1'`; `needs_human`: `False` vs `True` | Missed `>= ₹50,000` rule for `P1` (which also flips `needs_human` to `True`). |
| **`dv-00034`** | Payment declined, `Rs. 36737` | `priority`: `'P1'` vs `'P2'`; `needs_human`: `True` vs `False` | Over-escalated a large `P2` amount (`< ₹50,000`) to `P1`. |
| **`dv-00035`** | Double charged, `₹6,686` | `priority`: `'P3'` vs `'P2'` | Missed `>= ₹5,000` rule for `refund` `P2` priority. |
| **`dv-00055`** | Charged twice, `INR 7,576` | `priority`: `'P3'` vs `'P2'` | Missed `>= ₹5,000` rule for `refund` `P2` priority. |
| **`dv-00056`** | Double charged, `Rs 44,206.75` | `priority`: `'P1'` vs `'P2'`; `needs_human`: `True` vs `False` | Over-escalated amount near the `₹50,000` `P1` boundary. |
| **`dv-00075`** | Txn ID `np 41328 08828` | `txn_id`: `'NP413280828'` vs `'NP4132808828'` | Dropped one repeated `8` during space-removal normalization. |
| **`dv-00098`** | Duplicate payment, `41.5k` | `priority`: `'P1'` vs `'P2'`; `needs_human`: `True` vs `False` | Over-escalated `41.5k` (`< ₹50,000`) to `P1`. |
| **`dv-00120`** | Received `2026-06-27`, *"parso hi hua yeh"* | `txn_date`: `'2026-06-27'` vs `'2026-06-25'` | Failed to resolve Hinglish `"parso"` (2 days ago) against received date. |
| **`dv-00135`** | Refund pending, `Rs. 9843` | `priority`: `'P3'` vs `'P2'` | Missed `>= ₹5,000` threshold for `refund` `P2`. |
| **`dv-00141`** | Top-up failed, `INR 61,597` | `priority`: `'P2'` vs `'P1'`; `needs_human`: `False` vs `True` | Missed `>= ₹50,000` rule for `P1` escalation. |
| **`dv-00146`** | Cannot try PIN (*"third time"*) | `priority`: `'P3'` vs `'P2'` | Missed rule that `account_access` tickets are always at least `P2`. |
| **`dv-00148`** | Duplicate payment `INR 42,062` + 16 quoted `>` replies | `priority`: `'P3'` vs `'P2'` | Long quoted email thread (~1,000+ tokens) diluted attention on `₹42,062`. |
| **`dv-00164`** | PIN unblocked (*"again and again"*) + 11 quoted `>` replies | `needs_human`: `False` vs `True` | Long quoted email thread exceeded `max_length=512`, hiding `"again and again"`. |
| **`dv-00177`** | *"There has been a red flag on my top up"* | `category`: `'fraud'` vs `'payment_failed'` | `"red flag"` triggered `fraud`, even though the user initiated the top-up. |
| **`dv-00180`** | Balance `Rs. 241670`, txn `₹3,130` | `priority`: `'P2'` vs `'P3'` | Extracted `amount_paise` correctly (`313000`), but large balance biased `priority` to `P2`. |

### What I Would Do Next With One More Week
1. **Increase `max_length` to 1024** (or strip `>` quoted lines where appropriate) so long email threads like `dv-00148` and `dv-00164` do not dilute attention.
2. **True 16-bit LoRA & Rank Sweep:** Run an unquantized 16-bit LoRA comparison and test `r=32` at `learning_rate=1e-4`.

---

## 5. AI Tools Used
* **Gemini:** Used as an debugging assistant for saving th `auditted jsonl`, `trainned models `, apart from this I seeked it's help with debugging the auditting code logic and performing a format check on `predictions_test.jsonl`. Also needed help with version control of `bitsandbytes`,`transformers` and `trl`. Also to fix the `date` formatting and `txn_id` I took help from gemini.