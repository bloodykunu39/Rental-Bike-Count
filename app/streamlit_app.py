"""Explore the day-ahead forecast on the two months the model never saw (Oct - Nov 2018).

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from bikeshare import data, features, serving, validation  # noqa: E402

# Steps chosen to stay readable on both the light and the dark Streamlit theme
BLUE, GREY, RED = "#3987e5", "#898781", "#e34948"

st.set_page_config(page_title="Seoul bike demand forecast", page_icon="🚲", layout="wide")


@st.cache_resource
def load_model():
    return serving.Forecaster()


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model.booster)


@st.cache_data
def load_test_period():
    frame = features.build(data.load())
    test = frame[frame["timestamp"] >= validation.TEST_START].copy()
    model = load_model()
    test["forecast"] = model.predict(test)
    test["lower"], test["upper"] = model.interval(test)
    return test


model, test = load_model(), load_test_period()
card = model.card
v2_name = f"v2: {card['model']}"
v2 = card["test_metrics"][v2_name]

st.title("Seoul bike demand: day-ahead forecast")
st.caption(f"{card['model']} trained on Dec 2017 to Sep 2018. Everything on this page comes from October and "
           "November 2018, which the model never saw. Observed weather stands in for a weather forecast.")

overall = st.columns(4)
overall[0].metric("Average miss per hour", f"{v2['MAE']:.0f} bikes", border=True)
naive_mae = card["test_metrics"]["naive: same hour yesterday"]["MAE"]
overall[1].metric("Better than copying yesterday", f"{1 - v2['MAE'] / naive_mae:.0%}", border=True)
overall[2].metric("R² on the test months", f"{v2['R2']:.2f}", border=True)
overall[3].metric("Hours inside the 90% range", f"{card['test_interval_coverage']:.1%}", border=True)

with st.sidebar:
    st.header("Pick a day")
    day = st.date_input("Date in the test period", value=pd.Timestamp("2018-10-16"),
                        min_value=pd.Timestamp("2018-10-01"), max_value=pd.Timestamp("2018-11-30"))
    open_days = set(test["date"].dt.date)
    closed = [d for d in pd.date_range("2018-10-01", "2018-11-30").date if d not in open_days]
    st.caption("Closed all day (no rentals): " + ", ".join(d.strftime("%d %b") for d in closed))

today = test[test["date"].dt.date == day]
if today.empty:
    st.info(f"The service was closed on {day:%A %d %B %Y}, so there is nothing to forecast.")
    st.stop()

st.subheader(f"{day:%A %d %B %Y}")
kpis = st.columns(4)
total_error = today["forecast"].sum() / today["rentals"].sum() - 1
kpis[0].metric("Forecast for the day", f"{today['forecast'].sum():,.0f}", delta=f"{total_error:+.0%} vs actual",
               delta_color="off")
kpis[1].metric("Actual rentals", f"{today['rentals'].sum():,.0f}")
kpis[2].metric("Average miss per hour", f"{(today['rentals'] - today['forecast']).abs().mean():.0f} bikes")
inside = ((today["rentals"] >= today["lower"]) & (today["rentals"] <= today["upper"])).sum()
kpis[3].metric("Hours inside the 90% range", f"{inside} of {len(today)}")

chart_data = today[["hour", "rentals", "forecast", "lower", "upper"]].rename(
    columns={"rentals": "Actual", "forecast": "Forecast"})
base = alt.Chart(chart_data).encode(x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23]),
                                            axis=alt.Axis(tickCount=12)))
band = base.mark_area(opacity=0.15, color=BLUE).encode(y=alt.Y("lower:Q", title="Bikes rented per hour"),
                                                       y2="upper:Q")
lines = base.transform_fold(["Actual", "Forecast"], as_=["series", "bikes"]).mark_line(strokeWidth=2).encode(
    y="bikes:Q",
    color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[GREY, BLUE]),
                    legend=alt.Legend(title=None, orient="top-left")))
hover = alt.selection_point(fields=["hour"], nearest=True, on="pointerover", empty=False)
rule = base.mark_rule(color=GREY, strokeWidth=1).encode(
    opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
    tooltip=[alt.Tooltip("hour:Q", title="Hour"), alt.Tooltip("Actual:Q", format=",.0f"),
             alt.Tooltip("Forecast:Q", format=",.0f"), alt.Tooltip("lower:Q", title="90% range from", format=",.0f"),
             alt.Tooltip("upper:Q", title="90% range to", format=",.0f")]).add_params(hover)
st.altair_chart(alt.layer(band, lines, rule).properties(height=340), width="stretch")
st.caption("Shaded: 90% prediction range. Hover for the numbers.")

st.subheader("Why this forecast?")
hours = today["hour"].tolist()
hour = st.select_slider("Hour", options=hours, value=8 if 8 in hours else hours[0])
row = today[today["hour"] == hour]
x_row = row[features.FEATURES]

left, right = st.columns([3, 2], gap="large")
with left:
    contributions = load_explainer(model)(x_row).values[0]
    explained = pd.DataFrame({"input": [features.LABELS[f] for f in features.FEATURES], "value": contributions})
    explained["effect"] = np.where(explained["value"] >= 0, "pushes up", "pushes down")
    explained = explained.reindex(explained["value"].abs().sort_values(ascending=False).index).head(8)
    bars = alt.Chart(explained).mark_bar(size=16, cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Contribution (model's square-root scale)"),
        y=alt.Y("input:N", sort=None, title=None),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["pushes up", "pushes down"], range=[BLUE, RED]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["input", alt.Tooltip("value:Q", title="contribution", format="+.2f")])
    st.altair_chart(bars.properties(height=300), width="stretch")
    st.caption("SHAP values for the 8 inputs that moved this hour's forecast most.")
with right:
    shown = {"Forecast": f"{row['forecast'].iloc[0]:,.0f}", "Actual": f"{row['rentals'].iloc[0]:,.0f}",
             "90% range": f"{row['lower'].iloc[0]:,.0f} to {row['upper'].iloc[0]:,.0f}"}
    for f in ["temperature", "humidity", "rainfall", "rain_prev_3h", "lag_24h", "same_hour_7d_mean"]:
        value = row[f].iloc[0]
        shown[features.LABELS[f]] = "unknown (closed)" if pd.isna(value) else f"{value:,.1f}"
    st.dataframe(pd.Series(shown, name="value").to_frame(), width="stretch")

st.subheader("What if the weather had been different?")
sliders = st.columns(3)
temperature = sliders[0].slider("Temperature (°C)", -20.0, 40.0, float(row["temperature"].iloc[0]), 0.5)
humidity = sliders[1].slider("Humidity (%)", 0, 100, int(row["humidity"].iloc[0]))
rain = sliders[2].slider("Rain this hour (mm)", 0.0, 20.0, float(row["rainfall"].iloc[0]), 0.1)
what_if = x_row.copy()
what_if["temperature"], what_if["humidity"], what_if["rainfall"] = temperature, humidity, rain
new_forecast = model.predict(what_if)[0]
new_lower, new_upper = (bound[0] for bound in model.interval(what_if))
st.metric(f"Forecast for {hour}:00 with these conditions", f"{new_forecast:,.0f} bikes",
          delta=f"{new_forecast - row['forecast'].iloc[0]:+,.0f} vs the actual weather", delta_color="off")
st.caption(f"90% range {new_lower:,.0f} to {new_upper:,.0f}. Rain in the previous 3 hours is kept as it was.")

with st.expander("Model card"):
    table = pd.DataFrame(card["test_metrics"]).T[["MAE", "RMSE", "R2"]].round(3)
    st.dataframe(table, width="stretch")
    st.markdown(
        f"""
- **Task:** forecast each hour of the next day, made the evening before.
- **Validation:** trained on {card['train_period'][0]} to {card['train_period'][1]}, tuned on monthly folds
  (June to September), tested once on {card['test_period'][0]} to {card['test_period'][1]}.
- **Intervals:** split conformal, calibrated separately for 10 time bands; {card['test_interval_coverage']:.1%}
  of test hours fell inside the 90% range.
- **Limits:** observed weather stands in for forecasts; the weekday 8 am peak is under-forecast; holidays and
  days after a closure are hard; one year of data.
"""
    )
