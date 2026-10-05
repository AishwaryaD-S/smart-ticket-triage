# smart-ticket-triage
# 🎫 Smart Ticket Triage — an NLP Co-Pilot for Customer Support

Reads a raw customer support message and instantly predicts **where it should go, how urgent it is, and how long it will take** — then shows the result in an interactive triage console.


<!-- Add a screenshot of the console here: ![Triage Console](images/console.png) -->

## Problem

Support teams spend hours reading, tagging, prioritising and routing tickets. A *"production is down"* message ends up in the same queue as *"please add dark mode"*. Slow triage means angry customers, missed SLAs and burned-out agents.

## What it does

| Output | Task | Why it matters |
|---|---|---|
| **Category** (Billing, Technical Issue, Account Access, Shipping, Refund & Return, Feature Request) | Multi-class text classification | Routes the ticket to the right team |
| **Priority** (Low → Critical) | Ordinal classification | Decides who gets answered first |
| **Resolution time** (hours) | Regression | Sets customer expectations and SLAs |

On top of the models it adds a 0–100 **urgency score**, first-response **SLA**, confidence bars, the **words that drove the decision**, and a **live priority queue**.

## Dataset

`triage_tickets_dataset.csv` — 6,000 tickets, 9 columns. The data is **synthetic** (generated from templates with typos, urgency phrases and about 4% label noise to mimic messy real helpdesk exports), so it is safe to share.

| Column | Description |
|---|---|
| `ticket_id` | Unique ticket ID |
| `created_at` | Timestamp the ticket was created |
| `channel` | Email, Chat, Web Form, Phone, Twitter |
| `customer_tier` | Free, Standard, Premium, Enterprise |
| `subject` | Ticket subject line |
| `message` | Raw customer message (main model input) |
| `category` | **Target 1** — one of 6 classes |
| `priority` | **Target 2** — Low, Medium, High, Critical |
| `resolution_hours` | **Target 3** — hours to resolve |

**Dataset snapshot** (2025-01-01 to 2026-09-30, no missing values)

| | Distribution |
|---|---|
| Category | Technical Issue 1,422 · Billing 1,175 · Account Access 976 · Shipping 965 · Refund & Return 854 · Feature Request 608 |
| Priority | Medium 2,769 · Low 1,883 · High 1,009 · Critical 339 |
| Tier | Standard 2,093 · Free 2,080 · Premium 1,258 · Enterprise 569 |
| Channel | Twitter 1,246 · Web Form 1,195 · Phone 1,193 · Email 1,184 · Chat 1,182 |
| Resolution hours | median 16.8 · mean 30.0 · max 412.6 (right-skewed) |

The pipeline also works on a real export (Zendesk, Freshdesk, Kaggle customer-support datasets) as long as you keep these column names.

## Approach

1. **Features** — TF-IDF on two views at once: word 1–2-grams (phrases like "password reset") and character 2–5-grams (survives typos). Customer tier and channel are one-hot encoded for the priority and time models.
2. **Category** — Logistic Regression, Linear SVM and Complement NB compared with 5-fold cross-validation (macro-F1); Logistic Regression is used for its probabilities.
3. **Priority** — class-weighted Logistic Regression, evaluated with exact accuracy, within-one-level accuracy and macro-F1, since priority is ordinal.
4. **Resolution time** — Ridge regression on `log1p(hours)`, compared with a median baseline (MAE and R²).
5. **Explainability** — top words per category from the model weights.
6. **Triage engine + UI** — one `triage()` function, an `ipywidgets` console, and a Gradio app with a public link.

## Results

Held-out test set (1,200 tickets, 20% stratified split):

| Model | Metric | Score |
|---|---|---|
| Category | Macro-F1 (5-fold CV, Logistic Regression) | 0.962 ± 0.009 |
| Category | Accuracy / macro-F1 (test) | 0.968 / 0.968 |
| Priority | Exact accuracy | 0.874 |
| Priority | Within one level | 0.935 |
| Priority | Macro-F1 | 0.828 |
| Resolution time | MAE (hours) | 9.8 h vs 22.3 h median baseline |
| Resolution time | R² (log space) | 0.821 |

**Notes**
- Logistic Regression, Linear SVM and Complement NB score almost the same on category (0.962–0.963 CV macro-F1). Logistic Regression is used because it gives usable probabilities for the UI.
- The Critical class is the hard one: recall is 0.91 but precision is only 0.53, so the model over-flags Critical. It is a deliberate trade-off (class weighting) so urgent tickets are rarely missed.
- The dataset is template-generated, so these scores are higher than you should expect on a real helpdesk export.

## Quick start

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
cd <YOUR-REPO>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook smart_ticket_triage.ipynb
```

Run all cells top to bottom. The last cell launches a Gradio web app and prints a public `gradio.live` link (valid while the session is running, about 72 hours at most).

Or click the **Open in Colab** badge above — no installation needed.

## Project structure

```
.
├── smart_ticket_triage.ipynb     # full pipeline: EDA, models, triage engine, UI
├── triage_tickets_dataset.csv    # synthetic dataset (6,000 tickets)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

`ticket_triage_models.joblib` is created when you run the notebook and is not committed.

## Roadmap

- Sentence embeddings or a fine-tuned DistilBERT, compared against TF-IDF
- Sentiment and language detection to flag angry or non-English customers
- Ordinal regression or gradient boosting for priority
- Active learning: agents correct predictions in the UI and the models retrain weekly
- Permanent deployment (Hugging Face Spaces or FastAPI)

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · ipywidgets · Gradio

## License

MIT — see [LICENSE](LICENSE).

