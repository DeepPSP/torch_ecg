#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

import numpy as np, os
from scipy.io import loadmat


# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
six_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
four_leads = ("I", "II", "III", "V2")
three_leads = ("I", "II", "V2")
two_leads = ("I", "II")
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)


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


# (Re)sort leads using the standard order of leads for the standard twelve-lead ECG.
def sort_leads(leads):
    x = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    leads = sorted(
        leads,
        key=lambda lead: (x.index(lead) if lead in x else len(x) + leads.index(lead)),
    )
    return tuple(leads)


# Find header and recording files.
def find_challenge_files(data_directory):
    header_files = list()
    recording_files = list()
    for f in sorted(os.listdir(data_directory)):
        root, extension = os.path.splitext(f)
        if not root.startswith(".") and extension == ".hea":
            header_file = os.path.join(data_directory, root + ".hea")
            recording_file = os.path.join(data_directory, root + ".mat")
            if os.path.isfile(header_file) and os.path.isfile(recording_file):
                header_files.append(header_file)
                recording_files.append(recording_file)
    return header_files, recording_files


# Load header file as a string.
def load_header(header_file):
    with open(header_file, "r") as f:
        header = f.read()
    return header


# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key="val"):
    from scipy.io import loadmat

    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording


# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording


# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split("\n")):
        if i == 0:
            try:
                recording_id = l.split(" ")[0]
            except:
                pass
        else:
            break
    return recording_id


# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split("\n")):
        entries = l.split(" ")
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)


# Get age from header.
def get_age(header):
    age = None
    for l in header.split("\n"):
        if l.startswith("#Age"):
            try:
                age = float(l.split(": ")[1].strip())
            except:
                age = float("nan")
    return age


# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split("\n"):
        if l.startswith("#Sex"):
            try:
                sex = l.split(": ")[1].strip()
            except:
                pass
    return sex


# Get number of leads from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split("\n")):
        if i == 0:
            try:
                num_leads = float(l.split(" ")[1])
            except:
                pass
        else:
            break
    return num_leads


# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split("\n")):
        if i == 0:
            try:
                frequency = float(l.split(" ")[2])
            except:
                pass
        else:
            break
    return frequency


# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split("\n")):
        if i == 0:
            try:
                num_samples = float(l.split(" ")[3])
            except:
                pass
        else:
            break
    return num_samples


# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split("\n")):
        entries = l.split(" ")
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split("/")[0])
                except:
                    pass
        else:
            break
    return adc_gains


# Get baselines from header.
def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split("\n")):
        entries = l.split(" ")
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4].split("/")[0])
                except:
                    pass
        else:
            break
    return baselines


# Get labels from header.
def get_labels(header):
    labels = list()
    for l in header.split("\n"):
        if l.startswith("#Dx"):
            try:
                entries = l.split(": ")[1].split(",")
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels


# Save outputs from model.
def save_outputs(output_file, recording_id, classes, labels, probabilities):
    # Format the model outputs.
    recording_string = "#{}".format(recording_id)
    class_string = ",".join(str(c) for c in classes)
    label_string = ",".join(str(l) for l in labels)
    probabilities_string = ",".join(str(p) for p in probabilities)
    output_string = (
        recording_string
        + "\n"
        + class_string
        + "\n"
        + label_string
        + "\n"
        + probabilities_string
        + "\n"
    )

    # Save the model outputs.
    with open(output_file, "w") as f:
        f.write(output_string)


# Load outputs from model.
def load_outputs(output_file):
    with open(output_file, "r") as f:
        for i, l in enumerate(f):
            if i == 0:
                recording_id = l[1:] if len(l) > 1 else None
            elif i == 1:
                classes = tuple(entry.strip() for entry in l.split(","))
            elif i == 2:
                labels = tuple(entry.strip() for entry in l.split(","))
            elif i == 3:
                probabilities = tuple(
                    float(entry) if is_finite_number(entry) else float("nan")
                    for entry in l.split(",")
                )
            else:
                break
    return recording_id, classes, labels, probabilities
