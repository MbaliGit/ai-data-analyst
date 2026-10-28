import json
import pandas as pd
import plotly.express as px
import streamlit as st
from ollama import chat

MODEL = "llama3.2"

def create_plan(question, df):
    prompt = f"""
Return JSON only with these fields:
operation, group_column, value_column.

Allowed operations:
summary, groupby_sum, groupby_mean, trend, anomalies.

Use null for unused columns.
Choose only columns from this list:
{df.columns.tolist()}

Question: {question}
"""

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "summary", "groupby_sum",
                        "groupby_mean", "trend", "anomalies"
                    ]
                },
                "group_column": {
                    "type": ["string", "null"]
                },
                "value_column": {
                    "type": ["string", "null"]
                }
            },
            "required": [
                "operation", "group_column", "value_column"
            ]
        }
    )

    return json.loads(response.message.content)

    def analyze(df, plan):
        operation = plan["operation"]
        group_col = plan["group_column"]
        value_col = plan["value_column"]

    if operation == "summary":
        return df.describe(include="all").fillna("").to_string(), None

    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("Invalid column selected.")

    if operation in ("groupby_sum", "groupby_mean"):
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise ValueError("Value column must be numeric.")

        grouped = df.groupby(group_col)[value_col]
        result = (
            grouped.sum() if operation == "groupby_sum"
            else grouped.mean()
        ).sort_values(ascending=False).head(20)

        return result.to_string(), px.bar(
            result, title=f"{operation}: {value_col}"
        )

    if operation == "trend":
        dates = pd.to_datetime(df[group_col], errors="coerce")
        values = pd.to_numeric(df[value_col], errors="coerce")
        result = pd.DataFrame({
            "date": dates, "value": values
        }).dropna().sort_values("date")

        if result.empty:
            raise ValueError("No valid dates or numeric values.")

        return result.head(1000).to_string(index=False), px.line(
            result, x="date", y="value", title="Trend over time"
        )

    if operation == "anomalies":
        values = pd.to_numeric(df[value_col], errors="coerce")
        mean, std = values.mean(), values.std()

        if pd.isna(std) or std == 0:
            return "No variation to evaluate.", None

        unusual = df[(values - mean).abs() > 3 * std]
        return unusual.head(100).to_string(index=False), None

    raise ValueError("Unsupported operation.")

    st.title("AI Data Analyst")
st.write("Upload a CSV and explore your data using natural language.")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)

        if df.empty or len(df.columns) == 0:
            st.error("The CSV contains no usable data.")
            st.stop()

        st.subheader("Dataset preview")
        st.dataframe(df.head(10))
        st.write(f"{len(df):,} rows · {len(df.columns)} columns")

        question = st.text_input(
            "Ask a question",
            placeholder="Which category has the highest sales?"
        )

        if st.button("Analyze") and question:
            with st.spinner("Analyzing your data..."):
                plan = create_plan(question, df)
                result, chart = analyze(df, plan)

            st.subheader("Analysis results")
            st.code(result)

            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

            with st.spinner("Summarizing findings..."):
                response = chat(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Summarize these computed results accurately. "
                            "Do not invent facts or causal explanations.\n"
                            + result[:6000]
                        )
                    }]
                )

            st.subheader("What the data suggests")
            st.write(response.message.content)str

    except Exception as exc:
        st.error(f"Could not analyze the file: {exc}")
