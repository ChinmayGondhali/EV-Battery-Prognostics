import os
import sys
import io
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.io as sio
import shap

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(
        0,
        PROJECT_DIR
    )


# ============================================================
# LOAD EXISTING SOH PREDICTOR
# ============================================================

from src.soh_predictor import (
    predict_soh,
    SOH_FEATURES
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Battery SOH & RUL",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.85rem;
    }

    .status-box {
        padding: 0.6rem 0.9rem;
        border-radius: 0.6rem;
        font-weight: 650;
        margin-top: 0.3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL METADATA
# ============================================================

metadata_path = os.path.join(
    PROJECT_DIR,
    "models",
    "random_forest_soh_model_metadata.json"
)

with open(
    metadata_path,
    "r",
    encoding="utf-8"
) as f:
    model_metadata = json.load(f)


# ============================================================
# OXFORD DATASET FUNCTIONS
# ============================================================

def load_mat_from_bytes(file_bytes):
    """
    Load Oxford MATLAB dataset from bytes.
    """

    return sio.loadmat(
        io.BytesIO(file_bytes),
        struct_as_record=True,
        squeeze_me=False
    )


def get_cells(data):

    cells = [
        key
        for key in data.keys()
        if key.startswith("Cell")
    ]

    return sorted(
        cells,
        key=lambda x: int(
            x.replace("Cell", "")
        )
    )


def get_cycles(data, cell_name):

    cell_struct = data[
        cell_name
    ][0, 0]

    cycle_names = [
        name
        for name in cell_struct.dtype.names
        if name.startswith("cyc")
    ]

    return sorted(
        cycle_names,
        key=lambda x: int(
            x.replace("cyc", "")
        )
    )


def get_cycle_number(cycle_name):

    return int(
        cycle_name.replace(
            "cyc",
            ""
        )
    )


def get_discharge(data, cell_name, cycle_name):

    cell_struct = data[
        cell_name
    ][0, 0]

    cycle_struct = cell_struct[
        cycle_name
    ][0, 0]

    c1dc = cycle_struct[
        "C1dc"
    ][0, 0]

    time = np.asarray(
        c1dc["t"]
    ).flatten()

    voltage = np.asarray(
        c1dc["v"]
    ).flatten()

    charge = np.asarray(
        c1dc["q"]
    ).flatten()

    temperature = np.asarray(
        c1dc["T"]
    ).flatten()

    valid = (
        np.isfinite(time)
        & np.isfinite(voltage)
        & np.isfinite(charge)
        & np.isfinite(temperature)
    )

    time = time[valid]
    voltage = voltage[valid]
    charge = charge[valid]
    temperature = temperature[valid]

    if len(time) < 10:

        raise ValueError(
            "Insufficient discharge measurements."
        )

    order = np.argsort(
        time
    )

    time = time[order]
    voltage = voltage[order]
    charge = charge[order]
    temperature = temperature[order]

    unique_time, unique_idx = np.unique(
        time,
        return_index=True
    )

    time = unique_time
    voltage = voltage[unique_idx]
    charge = charge[unique_idx]
    temperature = temperature[unique_idx]

    # MATLAB serial-day timestamp → elapsed seconds
    elapsed_seconds = (
        time - time[0]
    ) * 24 * 60 * 60

    return {
        "time_seconds": elapsed_seconds,
        "voltage": voltage,
        "charge": charge,
        "temperature": temperature
    }


def extract_model_features(
    data,
    cell_name,
    cycle_name
):

    signals = get_discharge(
        data,
        cell_name,
        cycle_name
    )

    voltage = signals[
        "voltage"
    ]

    temperature = signals[
        "temperature"
    ]

    duration = signals[
        "time_seconds"
    ][-1]

    features = {
        "cycle": get_cycle_number(
            cycle_name
        ),

        "initial_voltage_V": float(
            voltage[0]
        ),

        "final_voltage_V": float(
            voltage[-1]
        ),

        "mean_voltage_V": float(
            voltage.mean()
        ),

        "initial_temperature_C": float(
            temperature[0]
        ),

        "final_temperature_C": float(
            temperature[-1]
        ),

        "min_temperature_C": float(
            temperature.min()
        ),

        "max_temperature_C": float(
            temperature.max()
        ),

        "temperature_rise_C": float(
            temperature.max()
            - temperature[0]
        ),

        "discharge_duration_s": float(
            duration
        )
    }

    return features


def calculate_capacity(signals):

    charge = signals[
        "charge"
    ]

    return float(
        abs(
            charge[-1]
            - charge[0]
        )
    )


# ============================================================
# BUILD HISTORICAL SUMMARY
# ============================================================

@st.cache_data(
    show_spinner=False
)
def build_summary_from_bytes(
    file_bytes
):

    data = load_mat_from_bytes(
        file_bytes
    )

    records = []

    cells = get_cells(
        data
    )

    for cell_name in cells:

        cycles = get_cycles(
            data,
            cell_name
        )

        for cycle_name in cycles:

            signals = get_discharge(
                data,
                cell_name,
                cycle_name
            )

            features = extract_model_features(
                data,
                cell_name,
                cycle_name
            )

            capacity = calculate_capacity(
                signals
            )

            records.append({
                "cell": cell_name,
                "cycle": get_cycle_number(
                    cycle_name
                ),
                "capacity_mAh": capacity,
                **features
            })

    summary = pd.DataFrame(
        records
    )

    summary = summary.sort_values(
        ["cell", "cycle"]
    ).reset_index(
        drop=True
    )

    initial_capacity = (
        summary
        .groupby("cell")[
            "capacity_mAh"
        ]
        .transform("first")
    )

    summary["SOH_percent"] = (
        summary["capacity_mAh"]
        / initial_capacity
        * 100
    )

    summary["terminal_cycle"] = (
        summary
        .groupby("cell")[
            "cycle"
        ]
        .transform("max")
    )

    summary["RUL_cycles"] = (
        summary["terminal_cycle"]
        - summary["cycle"]
    )

    return data, summary


# ============================================================
# SIDEBAR DATA SOURCE
# ============================================================

with st.sidebar:

    st.header("🔋 Controls")

    st.subheader("Data")

    project_dataset_path = os.path.join(
        PROJECT_DIR,
        "Dataset",
        "Oxford_Battery_Degradation_Dataset_1.mat"
    )

    use_project_dataset = False

    if os.path.exists(
        project_dataset_path
    ):

        use_project_dataset = st.checkbox(
            "Use Oxford project dataset",
            value=True
        )

    uploaded_file = None

    if not use_project_dataset:

        uploaded_file = st.file_uploader(
            "Upload Oxford .mat file",
            type=["mat"],
            max_upload_size=400
        )


# ============================================================
# LOAD DATASET
# ============================================================

dataset_loaded = False


if use_project_dataset:

    try:

        with open(
            project_dataset_path,
            "rb"
        ) as f:

            dataset_bytes = f.read()

        data, summary_df = build_summary_from_bytes(
            dataset_bytes
        )

        dataset_loaded = True

    except Exception as error:

        st.sidebar.error(
            f"Project dataset error: {error}"
        )


elif uploaded_file is not None:

    try:

        dataset_bytes = uploaded_file.getvalue()

        data, summary_df = build_summary_from_bytes(
            dataset_bytes
        )

        dataset_loaded = True

    except Exception as error:

        st.sidebar.error(
            f"Uploaded dataset error: {error}"
        )


# ============================================================
# SIDEBAR BATTERY CONTROLS
# ============================================================

if dataset_loaded:

    cells = get_cells(
        data
    )

    selected_cell = st.sidebar.selectbox(
        "Battery",
        cells,
        index=0
    )

    cell_history = (
        summary_df[
            summary_df["cell"] == selected_cell
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    cycle_options = (
        cell_history["cycle"]
        .astype(int)
        .tolist()
    )

    selected_cycle_number = st.sidebar.select_slider(
        "Cycle",
        options=cycle_options,
        value=cycle_options[0]
    )

    selected_cycle = (
        f"cyc{selected_cycle_number:04d}"
    )

    st.sidebar.subheader(
        "Model"
    )

    selected_model = st.sidebar.selectbox(
        "SOH Model",
        [
            "Random Forest"
        ]
    )

    st.sidebar.caption(
        "Selected model: Random Forest"
    )

else:

    selected_cell = None
    selected_cycle_number = None
    selected_cycle = None
    cell_history = None
    selected_model = "Random Forest"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔋 EV Battery State of Health & Remaining Useful Life'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Oxford Battery Degradation Dataset 1 · '
    'Kokam lithium-ion pouch cells · 40°C testing · '
    'Random Forest SOH model'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATA NOT LOADED
# ============================================================

if not dataset_loaded:

    st.info(
        "Select **Use Oxford project dataset** in the sidebar "
        "or upload the Oxford .mat dataset."
    )

    st.subheader(
        "Dashboard workflow"
    )

    st.write(
        """
        **Battery selection → Characterization cycle →
        Automatic feature extraction → SOH prediction →
        Battery health analysis → Explainability**
        """
    )

    st.stop()


# ============================================================
# CURRENT RECORD
# ============================================================

current_record = summary_df[
    (summary_df["cell"] == selected_cell)
    &
    (summary_df["cycle"] == selected_cycle_number)
].iloc[0]


current_features = extract_model_features(
    data,
    selected_cell,
    selected_cycle
)


predicted_soh = float(
    predict_soh(
        current_features
    )[0]
)


measured_soh = float(
    current_record["SOH_percent"]
)

measured_rul = int(
    current_record["RUL_cycles"]
)

terminal_cycle = int(
    current_record["terminal_cycle"]
)

peak_temperature = float(
    current_record["max_temperature_C"]
)

prediction_difference = (
    predicted_soh
    - measured_soh
)


# ============================================================
# EXPERIMENTAL RUL MODEL
# ============================================================

rul_features = [
    "cycle",
    "SOH_percent",
    "initial_voltage_V",
    "final_voltage_V",
    "mean_voltage_V",
    "initial_temperature_C",
    "final_temperature_C",
    "min_temperature_C",
    "max_temperature_C",
    "temperature_rise_C",
    "discharge_duration_s"
]

try:

    rul_train = summary_df[
        summary_df["cell"] != selected_cell
    ].copy()

    rul_test = summary_df[
        summary_df["cell"] == selected_cell
    ].copy()

    rul_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )

    rul_model.fit(
        rul_train[rul_features],
        rul_train["RUL_cycles"]
    )

    rul_input = (
        rul_test[
            rul_test["cycle"]
            == selected_cycle_number
        ][rul_features]
    )

    predicted_rul = float(
        rul_model.predict(
            rul_input
        )[0]
    )

except Exception:

    predicted_rul = np.nan


# ============================================================
# HEALTH INDICATOR
# ============================================================

if predicted_soh >= 90:

    health_status = "✅ Healthy"

elif predicted_soh >= 80:

    health_status = "🟡 Aging"

else:

    health_status = "🔴 Attention"


# ============================================================
# TOP METRIC CARDS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "State of Health",
        f"{predicted_soh:.1f}%",
        f"{prediction_difference:+.2f}% vs measured"
    )


with metric2:

    if np.isfinite(
        predicted_rul
    ):

        st.metric(
            "RUL (estimated)",
            f"{predicted_rul:,.0f} cycles"
        )

        st.caption(
            f"Recorded endpoint RUL: "
            f"{measured_rul:,} cycles"
        )

    else:

        st.metric(
            "RUL (estimated)",
            "Unavailable"
        )


with metric3:

    st.metric(
        "Peak Temperature",
        f"{peak_temperature:.1f} °C"
    )


with metric4:

    st.metric(
        "Health Status",
        health_status
    )


st.caption(
    f"{selected_cell} · cycle {selected_cycle_number:,} · "
    f"model: {selected_model}"
)


# ============================================================
# NAVIGATION TABS
# ============================================================

(
    health_tab,
    prediction_tab,
    comparison_tab,
    insights_tab,
    explainability_tab,
    try_tab
) = st.tabs(
    [
        "Battery Health",
        "Predictions",
        "Model Comparison",
        "Insights",
        "Explainability",
        "Try It"
    ]
)


# ============================================================
# BATTERY HEALTH
# ============================================================

with health_tab:

    st.subheader(
        f"Capacity fade — {selected_cell}"
    )

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    ax.plot(
        cell_history["cycle"],
        cell_history["SOH_percent"],
        linewidth=2,
        label="Measured SOH"
    )

    ax.scatter(
        [selected_cycle_number],
        [measured_soh],
        s=70,
        zorder=5,
        label=f"Cycle {selected_cycle_number:,}"
    )

    ax.axvline(
        terminal_cycle,
        linestyle="--",
        linewidth=1.4,
        label=f"Recorded endpoint ({terminal_cycle:,})"
    )

    ax.set_xlabel(
        "Drive Cycle"
    )

    ax.set_ylabel(
        "SOH (%)"
    )

    ax.set_title(
        f"SOH degradation trajectory — {selected_cell}"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # Selected point information
    # --------------------------------------------------------

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.metric(
            "Measured SOH",
            f"{measured_soh:.2f}%"
        )

    with info2:

        st.metric(
            "Predicted SOH",
            f"{predicted_soh:.2f}%"
        )

    with info3:

        st.metric(
            "SOH Loss",
            f"{100 - measured_soh:.2f} pp"
        )

    with info4:

        st.metric(
            "Recorded RUL",
            f"{measured_rul:,} cycles"
        )


    # --------------------------------------------------------
    # Voltage / temperature curves
    # --------------------------------------------------------

    st.subheader(
        f"Discharge characterization — cycle {selected_cycle_number:,}"
    )

    signals = get_discharge(
        data,
        selected_cell,
        selected_cycle
    )

    curve1, curve2 = st.columns(2)


    with curve1:

        fig_v, ax_v = plt.subplots(
            figsize=(7, 4)
        )

        ax_v.plot(
            signals["time_seconds"],
            signals["voltage"],
            linewidth=1.8
        )

        ax_v.set_xlabel(
            "Elapsed Time (s)"
        )

        ax_v.set_ylabel(
            "Voltage (V)"
        )

        ax_v.set_title(
            "Voltage discharge curve"
        )

        ax_v.grid(
            True,
            alpha=0.25
        )

        fig_v.tight_layout()

        st.pyplot(
            fig_v,
            use_container_width=True
        )

        plt.close(fig_v)


    with curve2:

        fig_t, ax_t = plt.subplots(
            figsize=(7, 4)
        )

        ax_t.plot(
            signals["time_seconds"],
            signals["temperature"],
            linewidth=1.8
        )

        ax_t.set_xlabel(
            "Elapsed Time (s)"
        )

        ax_t.set_ylabel(
            "Temperature (°C)"
        )

        ax_t.set_title(
            "Temperature during discharge"
        )

        ax_t.grid(
            True,
            alpha=0.25
        )

        fig_t.tight_layout()

        st.pyplot(
            fig_t,
            use_container_width=True
        )

        plt.close(fig_t)


# ============================================================
# PREDICTIONS
# ============================================================

with prediction_tab:

    st.subheader(
        "Battery prediction summary"
    )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "Measured SOH",
            f"{measured_soh:.3f}%"
        )

        st.metric(
            "Predicted SOH",
            f"{predicted_soh:.3f}%"
        )

        st.metric(
            "Absolute SOH Error",
            f"{abs(prediction_difference):.3f} pp"
        )


    with p2:

        st.metric(
            "Current Cycle",
            f"{selected_cycle_number:,}"
        )

        st.metric(
            "Recorded Terminal Cycle",
            f"{terminal_cycle:,}"
        )

        st.metric(
            "Recorded-Endpoint RUL",
            f"{measured_rul:,} cycles"
        )


    st.divider()

    st.subheader(
        "Current model input features"
    )

    feature_display = pd.DataFrame({
        "Feature": SOH_FEATURES,
        "Value": [
            current_features[
                feature
            ]
            for feature in SOH_FEATURES
        ]
    })

    st.dataframe(
        feature_display,
        use_container_width=True,
        hide_index=True
    )


    st.warning(
        "The recorded-endpoint RUL is calculated from the "
        "complete degradation history. It is not a prospective "
        "80%-SOH EOL prediction. The experimental RUL model "
        "should also be interpreted cautiously because its "
        "cell-wise validation showed substantial variation."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

with comparison_tab:

    st.subheader(
        "SOH model evaluation"
    )

    st.write(
        "The selected Random Forest was evaluated using "
        "Cell7–Cell8 as completely unseen cells."
    )

    model_comparison_path = os.path.join(
        PROJECT_DIR,
        "results",
        "metrics",
        "model_comparison_unseen_cells.csv"
    )

    if os.path.exists(
        model_comparison_path
    ):

        try:

            comparison_df = pd.read_csv(
                model_comparison_path
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.warning(
                f"Could not read model comparison file: {error}"
            )

    else:

        st.info(
            "The saved multi-model comparison table is not "
            "currently available in results/metrics."
        )


    st.subheader(
        "Selected Random Forest — unseen-cell holdout"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "MAE",
            "0.9233%"
        )

    with c2:

        st.metric(
            "RMSE",
            "0.9493%"
        )

    with c3:

        st.metric(
            "R²",
            "0.9771"
        )

    st.caption(
        "Evaluation on Cell7–Cell8 held-out data."
    )


    st.subheader(
        "Model features"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": SOH_FEATURES
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INSIGHTS
# ============================================================

with insights_tab:

    st.subheader(
        f"Battery insights — {selected_cell}"
    )

    first_soh = float(
        cell_history.iloc[0]["SOH_percent"]
    )

    soh_loss = (
        first_soh
        - measured_soh
    )

    # Long-term degradation slope
    if len(cell_history) >= 2:

        long_slope = np.polyfit(
            cell_history["cycle"],
            cell_history["SOH_percent"],
            1
        )[0]

    else:

        long_slope = 0.0


    # Recent degradation slope
    recent_history = cell_history[
        cell_history["cycle"]
        >= selected_cycle_number - 1000
    ]

    if len(recent_history) >= 2:

        recent_slope = np.polyfit(
            recent_history["cycle"],
            recent_history["SOH_percent"],
            1
        )[0]

    else:

        recent_slope = 0.0


    i1, i2, i3, i4 = st.columns(4)

    with i1:

        st.metric(
            "Current SOH",
            f"{measured_soh:.2f}%"
        )

    with i2:

        st.metric(
            "SOH Loss",
            f"{soh_loss:.2f} pp"
        )

    with i3:

        st.metric(
            "Long-term Rate",
            f"{long_slope * 100:.3f} pp/100 cycles"
        )

    with i4:

        st.metric(
            "Recent Rate",
            f"{recent_slope * 100:.3f} pp/100 cycles"
        )


    st.subheader(
        "Battery characterization"
    )

    insight_table = pd.DataFrame({
        "Measurement": [
            "Cell",
            "Current cycle",
            "Measured SOH",
            "Predicted SOH",
            "Initial voltage",
            "Final voltage",
            "Mean voltage",
            "Initial temperature",
            "Peak temperature",
            "Temperature rise",
            "Discharge duration",
            "Recorded terminal cycle"
        ],

        "Value": [
            selected_cell,
            f"{selected_cycle_number:,}",
            f"{measured_soh:.3f}%",
            f"{predicted_soh:.3f}%",
            f"{current_features['initial_voltage_V']:.6f} V",
            f"{current_features['final_voltage_V']:.6f} V",
            f"{current_features['mean_voltage_V']:.6f} V",
            f"{current_features['initial_temperature_C']:.3f} °C",
            f"{current_features['max_temperature_C']:.3f} °C",
            f"{current_features['temperature_rise_C']:.3f} °C",
            f"{current_features['discharge_duration_s']:.2f} s",
            f"{terminal_cycle:,}"
        ]
    })

    st.dataframe(
        insight_table,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Cell-to-cell recorded terminal comparison"
    )

    terminal_df = (
        summary_df
        .sort_values(
            ["cell", "cycle"]
        )
        .groupby("cell")
        .tail(1)
        [
            [
                "cell",
                "cycle",
                "SOH_percent",
                "capacity_mAh"
            ]
        ]
        .copy()
    )

    terminal_df.columns = [
        "Cell",
        "Terminal Cycle",
        "Terminal SOH (%)",
        "Terminal Capacity (mAh)"
    ]

    terminal_df = terminal_df.reset_index(
        drop=True
    )

    st.dataframe(
        terminal_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

with explainability_tab:

    st.subheader(
        "Why did the Random Forest make this prediction?"
    )

    predictor_module = __import__(
        "src.soh_predictor",
        fromlist=["soh_model"]
    )

    loaded_model = predictor_module.soh_model

    explanation_input = pd.DataFrame(
        [current_features]
    )[SOH_FEATURES]

    try:

        explainer = shap.TreeExplainer(
            loaded_model
        )

        local_values = explainer.shap_values(
            explanation_input
        )

        local_values = np.asarray(
            local_values
        ).reshape(-1)

        local_df = pd.DataFrame({
            "Feature": SOH_FEATURES,
            "Value": [
                current_features[
                    feature
                ]
                for feature in SOH_FEATURES
            ],
            "SHAP Contribution": local_values
        })

        local_df["Absolute SHAP"] = (
            local_df[
                "SHAP Contribution"
            ].abs()
        )

        local_df = local_df.sort_values(
            "Absolute SHAP",
            ascending=False
        ).reset_index(
            drop=True
        )

        st.dataframe(
            local_df,
            use_container_width=True,
            hide_index=True
        )


        fig_s, ax_s = plt.subplots(
            figsize=(10, 6)
        )

        plot_df = local_df.sort_values(
            "SHAP Contribution"
        )

        ax_s.barh(
            plot_df["Feature"],
            plot_df["SHAP Contribution"]
        )

        ax_s.axvline(
            0,
            linewidth=1
        )

        ax_s.set_xlabel(
            "SHAP contribution to predicted SOH"
        )

        ax_s.set_title(
            f"Local SHAP explanation — "
            f"{selected_cell}, cycle {selected_cycle_number:,}"
        )

        fig_s.tight_layout()

        st.pyplot(
            fig_s,
            use_container_width=True
        )

        plt.close(fig_s)

        st.info(
            "SHAP explains model behavior. It does not establish "
            "causality between a feature and battery degradation."
        )

    except Exception as error:

        st.error(
            f"SHAP explanation failed: {error}"
        )


# ============================================================
# TRY IT
# ============================================================

with try_tab:

    st.subheader(
        "Try the SOH predictor"
    )

    st.write(
        "Enter the same 10 measurements used to train the "
        "selected Random Forest model."
    )

    t1, t2 = st.columns(2)


    with t1:

        manual_cycle = st.number_input(
            "Cycle",
            min_value=0.0,
            value=0.0,
            step=100.0
        )

        manual_initial_voltage = st.number_input(
            "Initial Voltage (V)",
            min_value=0.0,
            value=4.19,
            step=0.001,
            format="%.6f"
        )

        manual_final_voltage = st.number_input(
            "Final Voltage (V)",
            min_value=0.0,
            value=2.70,
            step=0.001,
            format="%.6f"
        )

        manual_mean_voltage = st.number_input(
            "Mean Voltage (V)",
            min_value=0.0,
            value=3.74,
            step=0.001,
            format="%.6f"
        )

        manual_initial_temperature = st.number_input(
            "Initial Temperature (°C)",
            value=40.0,
            step=0.001,
            format="%.6f"
        )


    with t2:

        manual_final_temperature = st.number_input(
            "Final Temperature (°C)",
            value=41.0,
            step=0.001,
            format="%.6f"
        )

        manual_min_temperature = st.number_input(
            "Minimum Temperature (°C)",
            value=39.8,
            step=0.001,
            format="%.6f"
        )

        manual_max_temperature = st.number_input(
            "Maximum Temperature (°C)",
            value=41.2,
            step=0.001,
            format="%.6f"
        )

        manual_temperature_rise = st.number_input(
            "Temperature Rise (°C)",
            min_value=0.0,
            value=1.2,
            step=0.001,
            format="%.6f"
        )

        manual_duration = st.number_input(
            "Discharge Duration (s)",
            min_value=0.0,
            value=3500.0,
            step=1.0,
            format="%.6f"
        )


    if st.button(
        "Predict SOH",
        type="primary",
        use_container_width=True
    ):

        manual_input = {
            "cycle": manual_cycle,
            "initial_voltage_V": manual_initial_voltage,
            "final_voltage_V": manual_final_voltage,
            "mean_voltage_V": manual_mean_voltage,
            "initial_temperature_C": manual_initial_temperature,
            "final_temperature_C": manual_final_temperature,
            "min_temperature_C": manual_min_temperature,
            "max_temperature_C": manual_max_temperature,
            "temperature_rise_C": manual_temperature_rise,
            "discharge_duration_s": manual_duration
        }

        try:

            manual_prediction = float(
                predict_soh(
                    manual_input
                )[0]
            )

            st.success(
                f"Predicted SOH: "
                f"**{manual_prediction:.2f}%**"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EV Battery Prognostics · Oxford Battery Degradation Dataset 1 · "
    "Random Forest SOH estimation · SHAP explainability"
)
