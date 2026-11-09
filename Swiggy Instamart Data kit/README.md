# Customer Intelligence & Support Copilot

An educational Python capstone project inspired by an Instamart-style
customer support and analytics workflow. It combines data engineering,
customer segmentation, churn prediction, ticket triage, policy
retrieval, and a database-backed refund assistant.

> **Important:** This is a teaching implementation. All generated
> customer and order records are synthetic. No real customer data,
> hosted LLM, API key, payment integration, or network call is required.

## Learning outcomes

By completing this project, students will learn to:

1.  Load and audit CSV exports using Python and pandas.
2.  Clean customer and order data and store trusted tables in SQLite.
3.  Analyse customer behaviour using SQL and pandas.
4.  Segment customers using RFM analysis and K-means clustering.
5.  Explore customer groups using PCA.
6.  Build a leakage-aware **customer churn prediction** model.
7.  Redact personally identifiable information (PII) and classify
    support tickets.
8.  Retrieve relevant policy passages using TF-IDF similarity.
9.  Execute bounded, database-backed refund tools.

## Project highlights

### 1. Customer segmentation

Customer segmentation groups customers with similar purchasing
behaviour. This helps a business understand different customer types and
plan suitable retention or engagement strategies.

**Methods used in this project:**

-   **RFM analysis**
    -   **Recency:** How recently did the customer place an order?
    -   **Frequency:** How many completed orders did the customer place?
    -   **Monetary:** How much did the customer spend?
-   **K-means clustering:** Groups customers based on their RFM
    behaviour.
-   **PCA (Principal Component Analysis):** Reduces the number of
    dimensions so customer patterns can be explored more easily.

**Typical workflow:**

1.  Select completed orders.
2.  Calculate Recency, Frequency, and Monetary values for each customer.
3.  Prepare and scale the customer features as required by the model.
4.  Compare K-means cluster counts from K=2 to K=8.
5.  Use K=4 as the notebook's initial teaching configuration.
6.  Review the customer groups and interpret their behaviour.

**Example of possible segment interpretations:**

  -----------------------------------------------------------------------
  Example segment         Possible behaviour      Possible business
                                                  action
  ----------------------- ----------------------- -----------------------
  High-value active       Recent and frequent     Retain and reward
                          purchases with high     
                          spend                   

  Loyal regular           Frequent purchases      Encourage continued
                                                  engagement

  New or occasional       Few purchases or short  Improve onboarding
                          history                 

  At-risk or inactive     Purchases are less      Consider a win-back
                          recent                  campaign
  -----------------------------------------------------------------------

These are illustrative interpretations, not guaranteed cluster names.
Actual cluster meanings must be determined from the generated results.

### 2. Customer churn prediction

**Customer churn** means a customer stops purchasing or becomes
inactive. Churn prediction uses historical customer behaviour to
estimate which customers may stop ordering.

This project includes a lapse/churn prediction workflow using a
supervised machine-learning model.

**Prediction workflow:**

1.  Define a historical observation period.
2.  Calculate customer features using orders placed strictly before
    `2025-09-01`.
3.  Use September outcomes to create the prediction label.
4.  Split the data into training and test sets.
5.  Fit preprocessing only on training data.
6.  Train and tune the model using `GridSearchCV`.
7.  Evaluate predictions on held-out test data.
8.  Save the trained model using `joblib`.

**Important: avoiding data leakage**

The model must not use future information when predicting churn. In this
project, features use orders before `2025-09-01`, while the label uses
September outcomes. This separation helps prevent future order
information from leaking into the input features.

The Premium retention list is illustrative only: its Premium label uses
the full RFM window and is post-hoc. It must not be presented as a
leakage-free model evaluation.

**Example prediction output (illustrative only):**

  Customer   Predicted status   Suggested action
  ---------- ------------------ ----------------------------
  C001       At risk            Review recent activity
  C002       Active             Continue normal engagement
  C003       At risk            Consider a retention offer

The actual output depends on the generated dataset and trained model.
These example rows are not fixed project results.

## Setup and run

Python 3.11+ is recommended.

### 1. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

``` bash
source .venv/bin/activate
```

### 2. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate synthetic data

``` bash
python generate_data.py
```

### 4. Run the project

``` bash
python run_project.py
```

### 5. Run the notebook

Open:

``` text
notebooks/Customer_Intelligence_Support_Copilot.ipynb
```

Run the notebook cells from top to bottom. The notebook uses the same
modules as the scripts.

## Repository map

``` text
project/
├── generate_data.py
├── mock_llm.py
├── run_project.py
├── requirements.txt
├── data/
│   └── raw/
├── src/
│   ├── data_pipeline.py
│   ├── customer_models.py
│   ├── triage.py
│   ├── policy_rag.py
│   └── refund_agent.py
├── sql/
│   └── queries.sql
├── notebooks/
│   └── Customer_Intelligence_Support_Copilot.ipynb
├── outputs/
└── state/
```

### Main files

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `generate_data.py`                  Generates deterministic synthetic
                                      CSV and policy data

  `mock_llm.py`                       Offline stand-in for
                                      classification, policy answers, and
                                      agent tool calls

  `src/data_pipeline.py`              Data audit, cleaning, SQLite
                                      loading, and validation

  `src/customer_models.py`            RFM analysis, K-means segmentation,
                                      PCA, and churn/lapse modelling

  `src/triage.py`                     PII redaction, ticket
                                      classification, retries,
                                      validation, and routing

  `src/policy_rag.py`                 Policy chunking, TF-IDF retrieval,
                                      and grounded policy evaluation

  `src/refund_agent.py`               Order lookup, refund calculation,
                                      ledger, and bounded tool loop

  `sql/queries.sql`                   Seven SQL learning patterns

  `outputs/`                          Generated project artifacts

  `state/`                            Agent checkpoint examples
  -----------------------------------------------------------------------

## Data workflow

1.  Load local CSV exports with pandas, for example:

    ``` python
    import pandas as pd

    orders = pd.read_csv(
        "data/raw/orders.csv",
        keep_default_na=False
    )
    ```

2.  Inspect the dataset's shape, data types, null/blank counts,
    duplicate rows, keys, and impossible values.

3.  Apply documented cleaning rules one at a time.

4.  Compare row counts and important values before and after cleaning.

5.  Save clean CSV files and load trusted data into SQLite.

6.  Run validation assertions before analytics or modelling. A failed
    assertion should stop the pipeline rather than silently pass invalid
    data downstream.

In a production system, data may come from an approved API or data
warehouse. This teaching project intentionally uses local synthetic
data.

## Part A --- Data foundation

The pipeline:

-   Standardizes city and gender values.
-   Treats invalid ages as missing.
-   Resolves duplicate orders by keeping the latest timestamp.
-   Imputes missing amounts from catalogue prices.
-   Sets negative delivery times to missing.
-   Imputes delivery durations only for completed orders, by product
    category.

Review each cleaning rule before teaching or changing it. Use the
capstone handout's numeric checkpoints as acceptance tests.

## Part B --- Customer segmentation and churn prediction

### Customer segmentation

-   RFM uses completed orders and a fixed snapshot date.
-   K-means diagnostics compare K=2 through K=8.
-   The notebook starts with K=4 as an interpretable teaching
    configuration.
-   PCA helps explore the customer feature space.

### Churn/lapse prediction

-   Features use orders strictly before `2025-09-01`.
-   The label uses September outcomes.
-   Preprocessing is fitted within the training pipeline.
-   `GridSearchCV` is used for model tuning.
-   The saved model is exported with `joblib`.

Keep the feature period and label period separate. Do not use future
order information as a prediction feature.

## Part C --- Ticket triage and privacy

-   Empty and overlong tickets are skipped before model calls.
-   Email, phone, and order ID patterns are redacted before prompting.
-   Retries handle simulated transient errors.
-   JSON category values are allow-listed.
-   Malformed outputs are accepted only when exactly one allowed
    category can be detected using `re.findall`.
-   Human review receives guardrail, invalid-output, and budget cases.
-   Fraud tickets go to security.
-   Premium customers go to the premium desk.
-   Other tickets go to the support queue.

## Part D --- Policy retrieval and refund tools

-   Policy documents are split into overlapping chunks.
-   `TfidfVectorizer` represents the text.
-   Cosine similarity ranks relevant policy passages.
-   The offline mock returns context-derived sentences.
-   Refund rules are implemented in Python tools backed by SQLite.

**Teaching refund rules:**

-   Completed orders delivered in more than 45 minutes qualify for a
    full refund.
-   Orders over INR 1000 are escalated instead of being automatically
    refunded.

This is a teaching simulation, not a real payment integration.

## Suggested learning sequence

1.  Python fundamentals and project setup.
2.  CSV loading with pandas.
3.  Data inspection and cleaning.
4.  SQLite and SQL queries.
5.  RFM customer analytics.
6.  K-means customer segmentation and PCA.
7.  Churn prediction and leakage prevention.
8.  Ticket classification and PII redaction.
9.  TF-IDF policy retrieval.
10. Refund tools and final project execution.

## Useful learning resources

Use these resources to learn the underlying concepts. Adapt patterns to
this project and cite sources in your notes.

-   pandas: https://github.com/pandas-dev/pandas
-   scikit-learn: https://github.com/scikit-learn/scikit-learn
-   Python tutorial:
    https://github.com/python/cpython/tree/main/Doc/tutorial
-   Jupyter Notebook: https://github.com/jupyter/notebook
-   SQLite: https://github.com/sqlite/sqlite
-   scikit-learn text feature extraction:
    https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

## Limitations and next steps

-   This is a teaching implementation. It may need adjustment to match
    every numeric checkpoint in a strict grading rubric. Compare actual
    outputs with the supplied brief instead of hardcoding expected
    answers.
-   The offline mock model is a deterministic, rule-based simulation,
    not a real language model.
-   The refund ledger is a local JSONL audit trail, not a real
    transaction system.
-   The example segment descriptions and prediction rows in this README
    are illustrative, not guaranteed outputs.
-   Production work would require access controls, secure secret
    management, monitoring, schema contracts, data lineage, model drift
    checks, and human approval workflows.
#   S w i g g y - I n s t a m a r t - D a t a - k i t  
 