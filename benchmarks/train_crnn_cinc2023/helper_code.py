#!/usr/bin/env python

import os

import numpy as np
import scipy as sp
import scipy.io


# Find the folders with data files.
def find_data_folders(root_folder):
    data_folders = list()
    for x in sorted(os.listdir(root_folder)):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_file = os.path.join(data_folder, x + ".txt")
            if os.path.isfile(data_file):
                data_folders.append(x)
    return sorted(data_folders)


# Load the patient metadata: age, sex, etc.
def load_challenge_data(data_folder, patient_id):
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + ".txt")
    patient_metadata = load_text_file(patient_metadata_file)
    return patient_metadata


# Find the record names.
def find_recording_files(data_folder, patient_id):
    record_names = set()
    patient_folder = os.path.join(data_folder, patient_id)
    for file_name in sorted(os.listdir(patient_folder)):
        if not file_name.startswith(".") and file_name.endswith(".hea"):
            root, ext = os.path.splitext(file_name)
            record_name = "_".join(root.split("_")[:-1])
            record_names.add(record_name)
    return sorted(record_names)


# Load the WFDB data for the Challenge (but not all possible WFDB files).
def load_recording_data(record_name, check_values=False):
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext == "":
        header_file = record_name + ".hea"
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError("{} recording not found.".format(record_name))

    with open(header_file, "r") as f:
        header = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Parse the header file.
    record_name = None
    num_signals = None
    sampling_frequency = None
    num_samples = None
    signal_files = list()
    gains = list()
    baselines = list()
    adc_zeros = list()
    channels = list()
    initial_values = list()
    checksums = list()

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(" ")]
        # Parse the record line.
        if i == 0:
            record_name = arrs[0]
            num_signals = int(arrs[1])
            sampling_frequency = float(arrs[2])
            num_samples = int(arrs[3])
        # Parse the signal specification lines.
        elif not l.startswith("#") or len(l.strip()) == 0:
            signal_file = arrs[0]
            if "(" in arrs[2] and ")" in arrs[2]:
                gain = float(arrs[2].split("/")[0].split("(")[0])
                baseline = float(arrs[2].split("/")[0].split("(")[1].split(")")[0])
            else:
                gain = float(arrs[2].split("/")[0])
                baseline = 0.0
            adc_zero = int(arrs[4])
            initial_value = int(arrs[5])
            checksum = int(arrs[6])
            channel = arrs[8]
            signal_files.append(signal_file)
            gains.append(gain)
            baselines.append(baseline)
            adc_zeros.append(adc_zero)
            initial_values.append(initial_value)
            checksums.append(checksum)
            channels.append(channel)

    # Check that the header file only references one signal file. WFDB format allows for multiple signal files, but, for
    # simplicity, we have not done that here.
    num_signal_files = len(set(signal_files))
    if num_signal_files != 1:
        raise NotImplementedError(
            "The header file {}".format(header_file)
            + " references {} signal files; one signal file expected.".format(num_signal_files)
        )

    # Load the signal file.
    head, tail = os.path.split(header_file)
    signal_file = os.path.join(head, list(signal_files)[0])
    data = np.asarray(sp.io.loadmat(signal_file)["val"])

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    if np.shape(data) != (num_channels, num_samples):
        raise ValueError("The header file {}".format(header_file) + " is inconsistent with the dimensions of the signal file.")

    # Check that the initial value and checksums in the signal file are consistent with the initial value and checksums in the
    # header file.
    if check_values:
        for i in range(num_channels):
            if data[i, 0] != initial_values[i]:
                raise ValueError(
                    "The initial value in header file {}".format(header_file)
                    + " is inconsistent with the initial value for channel {} in the signal data".format(channels[i])
                )
            if np.sum(data[i, :], dtype=np.int16) != checksums[i]:
                raise ValueError(
                    "The checksum in header file {}".format(header_file)
                    + " is inconsistent with the checksum value for channel {} in the signal data".format(channels[i])
                )

    # Rescale the signal data using the gains and offsets.
    rescaled_data = np.zeros(np.shape(data), dtype=np.float32)
    for i in range(num_channels):
        rescaled_data[i, :] = (np.asarray(data[i, :], dtype=np.float64) - baselines[i] - adc_zeros[i]) / gains[i]

    return rescaled_data, channels, sampling_frequency


# Choose the channels.
def reduce_channels(current_data, current_channels, requested_channels):
    if current_channels == requested_channels:
        reduced_data = current_data
        reduced_channels = current_channels
    else:
        reduced_indices = [current_channels.index(channel) for channel in requested_channels if channel in current_channels]
        reduced_channels = [current_channels[i] for i in reduced_indices]
        reduced_data = current_data[reduced_indices, :]
    return reduced_data, reduced_channels


# Choose the channels.
def expand_channels(current_data, current_channels, requested_channels):
    if current_channels == requested_channels:
        expanded_data = current_data
    else:
        num_current_channels, num_samples = np.shape(current_data)
        num_requested_channels = len(requested_channels)
        expanded_data = np.zeros((num_requested_channels, num_samples))
        for i, channel in enumerate(requested_channels):
            if channel in current_channels:
                j = current_channels.index(channel)
                expanded_data[i, :] = current_data[j, :]
            else:
                expanded_data[i, :] = float("nan")
    return expanded_data


# Load text file as a string.
def load_text_file(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data


# Get a variable from the patient metadata.
def get_variable(text, variable_name, variable_type):
    variable = None
    for ln in text.split("\n"):
        if ln.startswith(variable_name):
            variable = ":".join(ln.split(":")[1:]).strip()
            variable = cast_variable(variable, variable_type)
            return variable


# Get the patient ID variable from the patient data.
def get_patient_id(string):
    return get_variable(string, "Patient", str)


# Get the patient ID variable from the patient data.
def get_hospital(string):
    return get_variable(string, "Hospital", str)


# Get the age variable (in years) from the patient data.
def get_age(string):
    return get_variable(string, "Age", int)


# Get the sex variable from the patient data.
def get_sex(string):
    return get_variable(string, "Sex", str)


# Get the ROSC variable (in minutes) from the patient data.
def get_rosc(string):
    return get_variable(string, "ROSC", int)


# Get the OHCA variable from the patient data.
def get_ohca(string):
    return get_variable(string, "OHCA", bool)


# Get the shockable rhythm variable from the patient data.
def get_shockable_rhythm(string):
    return get_variable(string, "Shockable Rhythm", bool)


# Get the TTM variable (in Celsius) from the patient data.
def get_ttm(string):
    return get_variable(string, "TTM", int)


# Get the Outcome variable from the patient data.
def get_outcome(string):
    variable = get_variable(string, "Outcome", str)
    if variable is None or is_nan(variable):
        raise ValueError("No outcome available. Is your code trying to load labels from the hidden data?")
    if variable == "Good":
        variable = 0
    elif variable == "Poor":
        variable = 1
    return variable


# Get the Outcome probability variable from the patient data.
def get_outcome_probability(string):
    variable = sanitize_scalar_value(get_variable(string, "Outcome Probability", str))
    if variable is None or is_nan(variable):
        raise ValueError("No outcome available. Is your code trying to load labels from the hidden data?")
    return variable


# Get the CPC variable from the patient data.
def get_cpc(string):
    variable = sanitize_scalar_value(get_variable(string, "CPC", str))
    if variable is None or is_nan(variable):
        raise ValueError("No CPC score available. Is your code trying to load labels from the hidden data?")
    return variable


# Get the utility frequency (in Hertz) from the recording data.
def get_utility_frequency(string):
    return get_variable(string, "#Utility frequency", int)


# Get the start time (in hh:mm:ss format) from the recording data.
def get_start_time(string):
    variable = get_variable(string, "#Start time", str)
    times = tuple(int(value) for value in variable.split(":"))
    return times


# Get the end time (in hh:mm:ss format) from the recording data.
def get_end_time(string):
    variable = get_variable(string, "#End time", str)
    times = tuple(int(value) for value in variable.split(":"))
    return times


# Convert seconds to days, hours, minutes, seconds.
def convert_seconds_to_hours_minutes_seconds(seconds):
    days = int(seconds / 86400)
    hours = int(seconds / 3600 - 24 * days)
    minutes = int(seconds / 60 - 24 * 60 * days - 60 * hours)
    seconds = int(seconds - 24 * 3600 * days - 3600 * hours - 60 * minutes)
    return hours, minutes, seconds


# Convert hours, minutes, and seconds to seconds.
def convert_hours_minutes_seconds_to_seconds(hours, minutes, seconds):
    return 3600 * hours + 60 * minutes + seconds


# Save the Challenge outputs for one file.
def save_challenge_outputs(filename, patient_id, outcome, outcome_probability, cpc):
    # Sanitize values, e.g., in case they are a singleton array.
    outcome = sanitize_boolean_value(outcome)
    outcome_probability = sanitize_scalar_value(outcome_probability)
    cpc = sanitize_scalar_value(cpc)

    # Format Challenge outputs.
    patient_string = "Patient: {}".format(patient_id)
    if outcome == 0:
        outcome = "Good"
    elif outcome == 1:
        outcome = "Poor"
    outcome_string = "Outcome: {}".format(outcome)
    outcome_probability_string = "Outcome Probability: {:.3f}".format(outcome_probability)
    cpc_string = "CPC: {:.3f}".format(cast_int_if_int_else_float(cpc))
    output_string = patient_string + "\n" + outcome_string + "\n" + outcome_probability_string + "\n" + cpc_string + "\n"

    # Write the Challenge outputs.
    if filename is not None:
        with open(filename, "w") as f:
            f.write(output_string)

    return output_string


# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False


# Check if a variable is a boolean or represents a boolean.
def is_boolean(x):
    if (is_number(x) and float(x) == 0) or (remove_extra_characters(x) in ("False", "false", "FALSE", "F", "f")):
        return True
    elif (is_number(x) and float(x) == 1) or (remove_extra_characters(x) in ("True", "true", "TRUE", "T", "t")):
        return True
    else:
        return False


# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False


# Remove any quotes, brackets (for singleton arrays), and/or invisible characters.
def remove_extra_characters(x):
    return str(x).replace('"', "").replace("'", "").replace("[", "").replace("]", "").replace(" ", "").strip()


# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x) == 0) or (remove_extra_characters(x) in ("False", "false", "FALSE", "F", "f")):
        return 0
    elif (is_number(x) and float(x) == 1) or (remove_extra_characters(x) in ("True", "true", "TRUE", "T", "t")):
        return 1
    else:
        return float("nan")


# Sanitize integer values.
def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float("nan")


# Sanitize scalar values.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float("nan")


# Cast a value to a particular type.
def cast_variable(variable, variable_type, preserve_nan=True):
    if preserve_nan and is_nan(variable):
        variable = float("nan")
    else:
        if variable_type == bool:
            variable = sanitize_boolean_value(variable)
        elif variable_type == int:
            variable = sanitize_integer_value(variable)
        elif variable_type == float:
            variable = sanitize_scalar_value(variable)
        else:
            variable = variable_type(variable)
    return variable


# Cast a value to an integer if the value is an integer, a float if the value is a non-integer float, and itself otherwise.
def cast_int_if_int_else_float(x):
    if is_integer(x):
        return int(float(x))
    elif is_number(x):
        return float(x)
    else:
        return x
