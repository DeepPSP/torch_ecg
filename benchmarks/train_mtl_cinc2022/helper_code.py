#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile  # noqa: E401


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


# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()


# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith(".") and extension == ".txt":
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(
            filenames, key=lambda filename: int(os.path.split(filename)[1][:-4])
        )

    return filenames


# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data


# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency


# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1 : num_locations + 1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(" ")
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            try:
                patient_id = l.split(" ")[0]
            except Exception:
                pass
        else:
            break
    return patient_id


# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            try:
                num_locations = int(l.split(" ")[1])
            except Exception:
                pass
        else:
            break
    return num_locations


# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            try:
                frequency = float(l.split(" ")[2])
            except Exception:
                pass
        else:
            break
    return frequency


# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split("\n")):
        entries = l.split(" ")
        if i == 0:
            pass
        elif 1 <= i <= num_locations:
            locations.append(entries[0])
        else:
            break
    return locations


# Get age from patient data.
def get_age(data):
    age = None
    for line in data.split("\n"):
        if line.startswith("#Age:"):
            try:
                age = line.split(": ")[1].strip()
            except Exception:
                pass
    return age


# Get sex from patient data.
def get_sex(data):
    sex = None
    for line in data.split("\n"):
        if line.startswith("#Sex:"):
            try:
                sex = line.split(": ")[1].strip()
            except Exception:
                pass
    return sex


# Get height from patient data.
def get_height(data):
    height = None
    for line in data.split("\n"):
        if line.startswith("#Height:"):
            try:
                height = float(line.split(": ")[1].strip())
            except Exception:
                pass
    return height


# Get weight from patient data.
def get_weight(data):
    weight = None
    for line in data.split("\n"):
        if line.startswith("#Weight:"):
            try:
                weight = float(line.split(": ")[1].strip())
            except Exception:
                pass
    return weight


# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for line in data.split("\n"):
        if line.startswith("#Pregnancy status:"):
            try:
                is_pregnant = bool(sanitize_binary_value(line.split(": ")[1].strip()))
            except Exception:
                pass
    return is_pregnant


# Get murmur from patient data.
def get_murmur(data):
    murmur = None
    for line in data.split("\n"):
        if line.startswith("#Murmur:"):
            try:
                murmur = line.split(": ")[1]
            except Exception:
                pass
    if murmur is None:
        raise ValueError(
            "No murmur available. Is your code trying to load labels from the hidden data?"
        )
    return murmur


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for line in data.split("\n"):
        if line.startswith("#Outcome:"):
            try:
                outcome = line.split(": ")[1]
            except Exception:
                pass
    if outcome is None:
        raise ValueError(
            "No outcome available. Is your code trying to load labels from the hidden data?"
        )
    return outcome


# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = (
        str(x).replace('"', "").replace("'", "").strip()
    )  # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x) == 1) or (x in ("True", "true", "T", "t")):
        return 1
    else:
        return 0


# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = (
        str(x).replace('"', "").replace("'", "").strip()
    )  # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and np.isinf(float(x))):
        return float(x)
    else:
        return 0.0


# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    patient_string = "#{}".format(patient_id)
    class_string = ",".join(str(c) for c in classes)
    label_string = ",".join(str(lb) for lb in labels)
    probabilities_string = ",".join(str(p) for p in probabilities)
    output_string = (
        patient_string
        + "\n"
        + class_string
        + "\n"
        + label_string
        + "\n"
        + probabilities_string
        + "\n"
    )

    # Write the Challenge outputs.
    with open(filename, "w") as f:
        f.write(output_string)


# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, "r") as f:
        for i, l in enumerate(f):
            if i == 0:
                patient_id = l.replace("#", "").strip()
            elif i == 1:
                classes = tuple(entry.strip() for entry in l.split(","))
            elif i == 2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(","))
            elif i == 3:
                probabilities = tuple(
                    sanitize_scalar_value(entry) for entry in l.split(",")
                )
            else:
                break
    return patient_id, classes, labels, probabilities
