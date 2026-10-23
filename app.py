from __future__ import annotations

import pandas as pd
import streamlit as st

from model import train_and_evaluate
from student_data import load_uploaded_records, load_uci_records

st.set_page_config(page_title="Classroom Pulse | Student Performance", layout="wide")


@st.cache_data(show_spinner="Training and evaluating the student-performance model…")
def analyze_records(records: pd.DataFrame) -> dict:
    return train_and_evaluate(records)


@st.cache_data

def get_uci_records() -> pd.DataFrame:
    return load_uci_records()


def format_grade(value: float) -> str:
    return f"{value:.1f} / 20"


def prepare_display_frame(scored: pd.DataFrame, school_days: int) -> pd.DataFrame:
    frame = scored.copy()
    frame["attendance_estimate"] = ((school_days - frame["absences"]) / school_days * 100).clip(0, 100)
    frame["attendance_status"] = frame["attendance_estimate"].map(
        lambda value: "Strong" if value >= 90 else ("Watch" if value >= 75 else "Low")
    )
    frame["mark_percent"] = frame["G3"] * 5
    if "model_confidence" in frame.columns:
        frame["confidence_percent"] = frame["model_confidence"] * 100
    return frame


def apply_filters(
    records: pd.DataFrame,
    query: str,
    selected_course: str,
    selected_outcome: str,
) -> pd.DataFrame:
    filtered = records.copy()
    if query.strip():
        needle = query.strip().casefold()
        filtered = filtered[
            filtered["record_id"].astype(str).str.casefold().str.contains(needle, na=False)
            | filtered["course"].astype(str).str.casefold().str.contains(needle, na=False)
        ]
    if selected_course != "All courses":
        filtered = filtered[filtered["course"] == selected_course]
    if selected_outcome != "All outcomes":
        filtered = filtered[filtered["actual_outcome"] == selected_outcome]
    return filtered


def model_analysis_or_stop(records: pd.DataFrame) -> dict:
    try:
        return analyze_records(records)
    except ValueError as exc:
        st.error(f"The model could not train on this dataset: {exc}")
        st.info("Upload a labeled CSV containing at least three passing and three failing records, or remove the upload to use the included UCI data.")
        st.stop()
        return {}


def show_record_table(records: pd.DataFrame, include_model: bool = True) -> None:
    columns = {
        "record_id": "Student record",
        "course": "Course / subject",
        "school": "School code",
        "age": "Age",
        "G1": "Period 1 /20",
        "G2": "Period 2 /20",
        "G3": "Final /20",
        "mark_percent": "Final mark %",
        "absences": "Absences",
        "attendance_estimate": "Est. attendance %",
        "actual_outcome": "Recorded result",
    }
    if include_model:
        columns["model_prediction"] = "ML prediction"
        columns["confidence_percent"] = "Model confidence %"
    view = records[list(columns)].rename(columns=columns)
    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "Period 1 /20": st.column_config.NumberColumn(format="%.1f"),
            "Period 2 /20": st.column_config.NumberColumn(format="%.1f"),
            "Final /20": st.column_config.NumberColumn(format="%.1f"),
            "Final mark %": st.column_config.NumberColumn(format="%.1f%%"),
            "Est. attendance %": st.column_config.NumberColumn(format="%.1f%%"),
            "Model confidence %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


with st.sidebar:
    st.image("assets/classroom-pulse-mark.png", width=54)
    st.title("Classroom Pulse")
    st.caption("Student performance, clearly in view")
    st.divider()
    uploaded_file = st.file_uploader(
        "Use your own labeled CSV",
        type=["csv"],
        help="Optional. The expected columns are listed in the Model lab and README.",
    )
    school_days = st.number_input(
        "School days for attendance estimate",
        min_value=30,
        max_value=250,
        value=180,
        step=1,
        help="The UCI source provides absence counts, not attendance percentages. This adjustable value is an assumption used only to calculate an estimate.",
    )
    page = st.radio("Workspace", ["Overview", "Student records", "Model lab"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        source_records = load_uploaded_records(uploaded_file)
        source_label = f"Uploaded CSV · {len(source_records):,} course records"
        st.sidebar.success(source_label)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.sidebar.error(f"CSV not loaded: {exc}")
        source_records = get_uci_records()
        source_label = "UCI Student Performance dataset (fallback)"
else:
    source_records = get_uci_records()
    source_label = "UCI Student Performance dataset"

records = prepare_display_frame(source_records, int(school_days))

with st.sidebar:
    st.caption(f"Active data: {source_label}")
    st.divider()
    search_query = st.text_input("Find a student record", placeholder="e.g. MATH-0001")
    course_choices = ["All courses", *sorted(records["course"].dropna().astype(str).unique().tolist())]
    selected_course = st.selectbox("Subject / course", course_choices)
    selected_outcome = st.selectbox("Performance status", ["All outcomes", "Pass", "Fail"])
    st.caption("Pass = final grade of 10/20 or higher.")

filtered = apply_filters(records, search_query, selected_course, selected_outcome)

st.title("Student performance")
st.caption(
    "An anonymized view of academic course records. Student IDs are generated for display; "
    "the source does not contain names or a universal student ID."
)

if page == "Overview":
    if filtered.empty:
        st.info("No records match the current filters. Adjust the filters in the sidebar.")
    else:
        average_mark = filtered["mark_percent"].mean()
        attendance_average = filtered["attendance_estimate"].mean()
        pass_rate = filtered["actual_outcome"].eq("Pass").mean() * 100
        metric_columns = st.columns(4)
        metric_columns[0].metric("Student-course records", f"{len(filtered):,}")
        metric_columns[1].metric(
            "Average final mark",
            f"{average_mark:.1f}%",
            help=f"Mean final grade on the source scale: {filtered['G3'].mean():.1f} / 20.",
        )
        metric_columns[2].metric("Estimated attendance", f"{attendance_average:.1f}%")
        metric_columns[3].metric("Recorded pass rate", f"{pass_rate:.1f}%")

        st.caption(
            f"Attendance is an estimate using {int(school_days)} assumed school days: "
            "(school days − recorded absences) ÷ school days. Change the assumption in the sidebar."
        )
        st.caption("Model fitting is deferred until you open Student records or Model lab so this overview can load first.")

        chart_left, chart_right = st.columns([1.35, 1])
        with chart_left:
            st.subheader("Average final mark by course")
            course_summary = (
                filtered.groupby("course", as_index=False)
                .agg(average_mark_percent=("mark_percent", "mean"), records=("G3", "size"))
                .set_index("course")
            )
            st.bar_chart(course_summary[["average_mark_percent"]])
            st.caption("Course marks are converted from the source's 0–20 scale to a percentage for this chart.")
        with chart_right:
            st.subheader("Attendance estimate status")
            attendance_counts = (
                filtered["attendance_status"]
                .value_counts()
                .reindex(["Strong", "Watch", "Low"], fill_value=0)
                .rename_axis("Estimated status")
                .to_frame("Course records")
            )
            st.bar_chart(attendance_counts)
            st.caption("Strong ≥90% · Watch 75–89% · Low <75% (all derived estimates).")

        st.subheader("Final-mark distribution")
        grade_band = pd.cut(
            filtered["G3"],
            bins=[-0.1, 5, 10, 15, 20.1],
            labels=["0–4", "5–9", "10–14", "15–20"],
            right=False,
        )
        distribution = pd.crosstab(grade_band, filtered["course"]).reindex(
            ["0–4", "5–9", "10–14", "15–20"], fill_value=0
        )
        st.bar_chart(distribution)
        st.caption("Grades are from the UCI dataset's original 0–20 scale; 10/20 is the pass threshold used here.")

        st.subheader("Records matching your filters")
        show_record_table(filtered.head(20), include_model=False)
        st.caption(f"Showing up to 20 of {len(filtered):,} matching records.")

elif page == "Student records":
    st.subheader("Student course records")
    st.write("Search by the generated record ID or course, then select a record to inspect its marks and model estimate.")
    if filtered.empty:
        st.info("No records match the current filters. Adjust the filters in the sidebar.")
    else:
        analysis = model_analysis_or_stop(source_records)
        records = prepare_display_frame(analysis["records"], int(school_days))
        filtered = apply_filters(records, search_query, selected_course, selected_outcome)
        show_record_table(filtered)
        st.divider()
        record_ids = filtered["record_id"].astype(str).tolist()
        chosen_id = st.selectbox("Student record detail", record_ids)
        selected = filtered.loc[filtered["record_id"].astype(str) == chosen_id].iloc[0]

        st.subheader(f"Record {selected['record_id']} · {selected['course']}")
        detail_metrics = st.columns(4)
        detail_metrics[0].metric("Recorded result", selected["actual_outcome"])
        detail_metrics[1].metric("Final grade", format_grade(float(selected["G3"])))
        detail_metrics[2].metric("Model prediction", selected["model_prediction"])
        detail_metrics[3].metric("Model confidence", f"{selected['confidence_percent']:.1f}%")

        grade_columns = st.columns(3)
        grade_columns[0].metric("Period 1", format_grade(float(selected["G1"])))
        grade_columns[1].metric("Period 2", format_grade(float(selected["G2"])))
        grade_columns[2].metric("Final period", format_grade(float(selected["G3"])))

        profile_left, profile_right = st.columns(2)
        with profile_left:
            st.markdown("**Academic profile**")
            st.write(f"Course / subject: {selected['course']}")
            st.write(f"School code: {selected['school']}")
            st.write(f"Age in source record: {int(selected['age'])}")
            study_hours = {1: "Under 2 hours/week", 2: "2–5 hours/week", 3: "5–10 hours/week", 4: "Over 10 hours/week"}
            st.write(f"Reported study time: {study_hours.get(int(selected['studytime']), 'Not specified')}")
            st.write(f"Previous class failures: {int(selected['failures'])}")
        with profile_right:
            st.markdown("**Attendance and outcome context**")
            st.metric("Recorded absences", int(selected["absences"]))
            st.metric("Estimated attendance", f"{selected['attendance_estimate']:.1f}% · {selected['attendance_status']}")
            st.caption(
                f"Estimate assumes {int(school_days)} school days; absence counts come from the source. "
                "The pass/fail result is based on the recorded final grade, while the model prediction is an out-of-fold estimate."
            )
        trajectory = pd.DataFrame(
            {"Grade / 20": [selected["G1"], selected["G2"], selected["G3"]]},
            index=["Period 1", "Period 2", "Final"],
        )
        st.subheader("Marks across the year")
        st.line_chart(trajectory)

else:
    analysis = model_analysis_or_stop(source_records)
    st.subheader("Model lab")
    st.write(
        "A Random Forest classifier predicts whether the final course grade reaches 10/20. "
        "The holdout test uses records excluded from training; table predictions use cross-validation so each displayed record is held out when its estimate is produced."
    )
    st.caption(
        "Inputs: earlier-period marks (G1 and G2), course, absences, study time, prior failures, and age. "
        "Model scores are educational estimates, not guarantees or a substitute for teacher judgment."
    )
    retrain_column, size_column = st.columns([1, 2])
    with retrain_column:
        if st.button("Re-run training and test", type="primary"):
            analyze_records.clear()
            st.rerun()
    with size_column:
        st.caption(f"Dataset: {source_label} · {analysis['train_count']:,} training records · {analysis['test_count']:,} holdout records · 80/20 stratified split")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Holdout accuracy", f"{analysis['metrics']['accuracy']:.1%}")
    metric_columns[1].metric("Pass precision", f"{analysis['metrics']['precision']:.1%}")
    metric_columns[2].metric("Pass recall", f"{analysis['metrics']['recall']:.1%}")
    metric_columns[3].metric("Pass F1 score", f"{analysis['metrics']['f1']:.1%}")

    matrix_left, features_right = st.columns([1, 1.2])
    with matrix_left:
        st.markdown("**Holdout confusion matrix**")
        st.caption("Rows are recorded outcomes; columns are model predictions.")
        st.dataframe(analysis["confusion_matrix"], width="stretch")
    with features_right:
        st.markdown("**Most influential features in the fitted model**")
        st.bar_chart(analysis["feature_importance"].set_index("Feature")["Importance"])

    st.markdown("**Holdout sample**")
    st.caption("Only the 20% test split is shown below; it was not used to fit the evaluation model.")
    holdout = analysis["holdout_records"].copy()
    holdout["model_confidence_percent"] = holdout["model_confidence"] * 100
    holdout = holdout.rename(
        columns={
            "record_id": "Student record",
            "course": "Course",
            "G3": "Recorded final grade /20",
            "actual_outcome": "Recorded outcome",
            "model_prediction": "Predicted outcome",
            "model_confidence_percent": "Model confidence %",
        }
    )
    st.dataframe(
        holdout[["Student record", "Course", "Recorded final grade /20", "Recorded outcome", "Predicted outcome", "Model confidence %"]],
        width="stretch",
        hide_index=True,
        column_config={"Model confidence %": st.column_config.NumberColumn(format="%.1f%%")},
    )

    st.divider()
    st.markdown("**Use a different labeled dataset**")
    st.write(
        "Upload a CSV in the sidebar with columns `course`, `age`, `studytime`, `failures`, `absences`, `G1`, `G2`, and `G3`. "
        "An optional `record_id` column supplies your own anonymous IDs; otherwise the app creates them. "
        "Grades must use a 0–20 scale and the file must contain both pass and fail examples. Uploaded data stays in the current app session and is not saved."
    )

st.divider()
st.caption(
    "Source: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance), "
    "licensed CC BY 4.0. The source includes absence counts, not attendance rates; all displayed percentages are estimates."
)
