# Implementation plan

## Product and implementation

Build the student-performance mini-project as a Python-only Streamlit app, not a React/JavaScript frontend. Use the real, anonymized UCI Student Performance records for Math and Portuguese courses. The default experience summarizes final grades, course averages, pass/fail outcomes, and absences, with searchable/filterable course-record views, an individual record detail view, and charts. UCI supplies absence counts rather than attendance percentages; therefore the app will show source absences and an explicitly labeled attendance estimate based on a user-adjustable number of school days, never present that estimate as source data. Record identifiers will be anonymized course/row labels because the source contains no names.

Use pandas and scikit-learn to train a Random Forest classifier that predicts whether the final grade G3 is at least 10 on UCI's 0–20 grade scale. Predictors include earlier grades G1/G2 and pre-final factors such as absences, study time, past failures, age, and course. Split the combined course records into stratified training and holdout test sets (80/20); report holdout accuracy, precision, recall, F1, and a confusion matrix. Generate out-of-fold predictions for displayed source records so each row's illustrative model prediction was made without training on that row; keep the actual G3-derived pass/fail outcome separate. Explain that these are historical public records and educational decision-support, not guarantees or a substitute for teacher judgment. Allow an optional CSV upload using the documented feature schema.

## Project structure

- `app.py`: Streamlit navigation, dataset controls, metrics, charts, course-record table/detail, model evaluation, and CSV upload.
- `student_data.py`: load the bundled UCI course files, assign display-only record IDs, validate optional labeled uploads, and derive recorded pass/fail outcomes.
- `model.py`: preprocessing, stratified 80/20 holdout fit/evaluation, out-of-fold predictions, and feature-importance extraction.
- `train_model.py`: repeatable command-line training/test summary.
- `serve.py`: start Streamlit privately and proxy its HTTP/websocket requests on `PORT`, while serving `manus-routes.json` as JSON at the root origin.
- `data/`: authentic UCI Math and Portuguese CSV files plus a blank upload schema template; dataset attribution is in `README.md`.
- `assets/classroom-pulse-mark.png`: small branded mark used in the Streamlit sidebar and project metadata.
- `.streamlit/config.toml`: light theme settings; `requirements.txt`: Python dependencies.
- `Dockerfile`: Python container entry point honoring `PORT`; `README.md`: local run, data source, model, attendance caveat, upload schema, and hosting instructions.
- `TODO.md`: outcome-level scope tracker; `plan.md`: implementation and design decisions.

Run locally with `streamlit run app.py`; use `python serve.py` for the managed Preview/container proxy on port 3000. There is no managed database or external API; uploaded CSVs stay in the current session and are not persisted.

## Design direction

Use a focused academic analytics console: bright neutral surfaces, deep ink text, and an indigo accent keep dense information readable. Streamlit's responsive sidebar and columns provide navigation, filters, and a clear metric/chart hierarchy without a separate frontend. Use compact status labels, clearly named scales, and plain disclosures for source grades, derived attendance, and model estimates. Keep interactions direct—search, select, inspect, upload, and download—with no decorative motion. Brand essence: a practical class-performance lens for educators; clear, supportive, and evidence-led. The wordmark pairs a stacked-mark motif with “Classroom Pulse”; the signature color is indigo.
