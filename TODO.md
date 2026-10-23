# TODO

## Python/Streamlit performance dashboard
- [x] Build the mini-project as a Python-only Streamlit website.
- [x] Show an overview with key metrics such as total student records, average marks, attendance estimate, and overall pass rate.
- [x] Show anonymized student/course records in a clear, searchable table and provide simple filters for student record, subject/course, and performance status.
- [x] Provide an individual student/course-record view with profile details, an overall academic summary, and marks broken down by period and course.
- [x] Show whether each student record passes or fails based on the published course result, separately identifying any model prediction.
- [x] Show attendance percentages with clear indicators for strong or low attendance; identify percentages as estimates derived from source absence counts and an adjustable school-day assumption.
- [x] Compare subject/course-wise performance across the class and use visual summaries for marks distribution, attendance estimates, and course averages.
- [x] Keep the layout usable on desktop, tablet, and mobile through Streamlit's responsive layout.

## AI/ML training, testing, and prediction
- [x] Use a Python AI/ML model to train and test student-performance data.
- [x] Split labeled data into training and holdout test sets; report model evaluation metrics and the confusion matrix.
- [x] Predict pass/fail from grades and attendance-related inputs, show the prediction and confidence beside the known result, and clearly distinguish model estimates from outcomes.
- [x] Use a real, anonymized, attributable dataset by default; accept an optional labeled CSV upload for training/evaluation and document its required columns.

## Streamlit project readiness
- [x] Include Python dependencies and run instructions so the application can be started with Streamlit.
- [x] Keep the application implementation Python-only; do not build the dashboard in React or JavaScript.
