# Classroom Pulse

A Python-only Streamlit mini-project for exploring student course performance and demonstrating pass/fail prediction with a real, anonymized education dataset.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with two included UCI course files: Mathematics and Portuguese. Its overview renders without training the ML model; model fitting is deferred until the user opens Student records or Model lab. It shows anonymized course-record IDs, grade progress (G1, G2, G3), course averages, pass/fail outcomes, absences, filters, charts, and an individual course-record detail view. Each row represents a student in one course; the source has no names or universal student identifier, so the IDs are generated for display and do not link a person across the two files.

## Dataset, outcome, and attendance estimate

The bundled `data/student-mat.csv` and `data/student-por.csv` are from Cortez's **Student Performance** dataset in the UCI Machine Learning Repository. UCI describes records from two Portuguese secondary schools, with Math and Portuguese course files. Grades G1 and G2 are earlier-period marks; G3 is the final mark on a 0–20 scale. The app uses G3 ≥ 10/20 as Pass and G3 < 10/20 as Fail. UCI notes that G1 and G2 are strongly correlated with G3, so the project models a later final outcome from earlier grades and other recorded factors; results are historical and illustrative rather than causal or guaranteed.

The source records a count of absences, not attendance percentages and not the number of possible school days. The app therefore labels attendance as an **estimate** and lets the viewer change the assumed number of school days (default 180). Its formula is `max(0, (assumed school days − absences) / assumed school days × 100)`. This derived estimate is not a source measurement. The dashboard also shows the original absence count.

UCI's repository lists 649 student instances and licenses the files under Creative Commons Attribution 4.0. The bundled course files contain 395 Math rows and 649 Portuguese rows, or 1,044 student-course records in total; that combined row count is not a deduplicated count of people.

**Attribution:** Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CSVs are included unmodified; the app adds display IDs and derived labels/estimates at runtime.

## Machine-learning workflow

The Model lab trains a scikit-learn Random Forest classifier on a stratified 80/20 training/holdout split. The target is whether G3 reaches 10/20. Inputs are course, age, study time, previous failures, absences, G1, and G2. The app reports holdout accuracy, pass precision, pass recall, F1, and a confusion matrix. It shows out-of-fold predictions and confidence scores for course records, so each displayed record is left out when its particular prediction is generated. Model scores are not calibrated probabilities and should not be used as a high-stakes decision by themselves.

The app also accepts a labeled CSV upload in the sidebar. Required columns are `course`, `age`, `studytime`, `failures`, `absences`, `G1`, `G2`, and `G3`. An optional `record_id` column is used for display; `school` is optional. Grades must use a 0–20 scale, and the upload must contain at least three Pass and three Fail examples. Uploaded records are held in the active session only; the app does not save them.

## Streamlit hosting

For Streamlit Community Cloud, place this project in a Git repository, select `app.py` as the app entry point, and keep `requirements.txt` at the project root. `Dockerfile` is also included for container-based Streamlit hosting; it listens on the `PORT` environment variable and defaults to port 3000.

## References

- UCI dataset record: https://archive.ics.uci.edu/dataset/320/student+performance
- Original study: Cortez, P. & Silva, A. M. G. (2008), *Using data mining to predict secondary school student performance*: https://repositorium.uminho.pt/server/api/core/bitstreams/991a0e2b-249d-466d-afef-937d975ff7fc/content
- Dataset license: https://creativecommons.org/licenses/by/4.0/


For the managed website Preview or the included container, the Python `serve.py` launcher starts Streamlit on a private internal port and proxies its HTTP/websocket traffic on `PORT` (default 3000). It also serves `manus-routes.json` as JSON at `/manus-routes.json`. When running a plain local Streamlit session, `streamlit run app.py` remains the simplest command.
