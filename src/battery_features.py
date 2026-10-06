import io
import re

import numpy as np
import scipy.io as sio


SOH_FEATURES = [
    "cycle",
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


def load_oxford_mat(file_source):

    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)

    if hasattr(file_source, "seek"):
        file_source.seek(0)

    return sio.loadmat(
        file_source,
        struct_as_record=True,
        squeeze_me=False
    )


def get_cell_names(data):

    cells = [
        name
        for name in data.keys()
        if re.fullmatch(r"Cell\d+", name)
    ]

    return sorted(
        cells,
        key=lambda x: int(
            x.replace("Cell", "")
        )
    )


def get_cycle_names(data, cell_name):

    cell_struct = data[
        cell_name
    ][0, 0]

    cycles = [
        name
        for name in cell_struct.dtype.names
        if re.fullmatch(r"cyc\d+", name)
    ]

    return sorted(
        cycles,
        key=lambda x: int(
            x.replace("cyc", "")
        )
    )


def extract_discharge_signals(
    data,
    cell_name,
    cycle_name
):

    cell_struct = data[
        cell_name
    ][0, 0]

    cycle_struct = cell_struct[
        cycle_name
    ][0, 0]

    if "C1dc" not in cycle_struct.dtype.names:

        raise ValueError(
            f"C1dc discharge data not found in "
            f"{cell_name} {cycle_name}"
        )

    c1dc = cycle_struct[
        "C1dc"
    ][0, 0]

    time = np.asarray(
        c1dc["t"]
    ).flatten()

    voltage = np.asarray(
        c1dc["v"]
    ).flatten()

    temperature = np.asarray(
        c1dc["T"]
    ).flatten()

    valid = (
        np.isfinite(time)
        & np.isfinite(voltage)
        & np.isfinite(temperature)
    )

    time = time[valid]
    voltage = voltage[valid]
    temperature = temperature[valid]

    if len(time) < 10:

        raise ValueError(
            f"Insufficient discharge measurements "
            f"in {cell_name} {cycle_name}"
        )

    order = np.argsort(time)

    time = time[order]
    voltage = voltage[order]
    temperature = temperature[order]

    unique_time, unique_idx = np.unique(
        time,
        return_index=True
    )

    time = unique_time
    voltage = voltage[unique_idx]
    temperature = temperature[unique_idx]

    elapsed_seconds = (
        time - time[0]
    ) * 24 * 60 * 60

    return {
        "time_seconds": elapsed_seconds,
        "voltage": voltage,
        "temperature": temperature
    }


def extract_soh_features(
    data,
    cell_name,
    cycle_name
):

    signals = extract_discharge_signals(
        data,
        cell_name,
        cycle_name
    )

    time_seconds = signals[
        "time_seconds"
    ]

    voltage = signals[
        "voltage"
    ]

    temperature = signals[
        "temperature"
    ]

    cycle_number = int(
        cycle_name.replace(
            "cyc",
            ""
        )
    )

    initial_voltage = float(
        voltage[0]
    )

    final_voltage = float(
        voltage[-1]
    )

    mean_voltage = float(
        voltage.mean()
    )

    initial_temperature = float(
        temperature[0]
    )

    final_temperature = float(
        temperature[-1]
    )

    min_temperature = float(
        temperature.min()
    )

    max_temperature = float(
        temperature.max()
    )

    temperature_rise = (
        max_temperature
        - initial_temperature
    )

    discharge_duration = float(
        time_seconds[-1]
    )

    return {
        "cycle": cycle_number,
        "initial_voltage_V": initial_voltage,
        "final_voltage_V": final_voltage,
        "mean_voltage_V": mean_voltage,
        "initial_temperature_C": initial_temperature,
        "final_temperature_C": final_temperature,
        "min_temperature_C": min_temperature,
        "max_temperature_C": max_temperature,
        "temperature_rise_C": temperature_rise,
        "discharge_duration_s": discharge_duration
    }
