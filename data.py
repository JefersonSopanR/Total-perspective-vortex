"""
data.py - PhysioNet EEGBCI loading, filtering and epoching.

Defines the 6 canonical "experiments" used throughout this project (each one
a binary classification task built from a group of PhysioNet runs), and
turns raw recordings into epoched (X, y) arrays ready for the CSP + LDA
pipeline.
"""

from dataclasses import dataclass

import numpy as np
import mne
from mne.datasets import eegbci

LOW_FREQ = 7.0
HIGH_FREQ = 30.0
POWERLINE_FREQ = 60.0

# tmin/tmax relative to each cue's onset, in seconds. PhysioNet cues last 4s;
# we keep a couple of seconds fully inside the cue window.
TMIN, TMAX = 0.0, 2.0

# with @dataclass, Python automatically generates __init__(), __repr__(), and __eq__() for you behind the scenes
# Passing frozen=True tells Python to make instances of the class read-only (immutable) after creation
# frozen=True -> read only!
@dataclass(frozen=True)
class Experiment:
    """One binary classification task: a group of PhysioNet runs plus the
    mapping from that group's annotation labels to class 0 / class 1."""
    runs: tuple
    event_id: dict  # e.g. {"T1": 0, "T2": 1}
    description: str

# Runs 3/7/11 and 4/8/12 contrast left vs right fist (real vs imagined).
# Runs 5/9/13 and 6/10/14 contrast both fists vs both feet (real vs imagined).
# T1/T2 keep one consistent meaning across every combined run:
#   experiment 4: T1 = left fist,  T2 = right fist  (runs 3,7,11 real + 4,8,12 imagined)
#   experiment 5: T1 = both fists, T2 = both feet   (runs 5,9,13 real + 6,10,14 imagined)

EXPERIMENTS = {
    0: Experiment((3, 7, 11), {"T1": 0, "T2": 1}, "left vs right fist (real)"),
    1: Experiment((4, 8, 12), {"T1": 0, "T2": 1}, "left vs right fist (imagined)"),
    2: Experiment((5, 9, 13), {"T1": 0, "T2": 1}, "fists vs feet (real)"),
    3: Experiment((6, 10, 14), {"T1": 0, "T2": 1}, "fists vs feet (imagined)"),
    4: Experiment((3, 7, 11, 4, 8, 12), {"T1": 0, "T2": 1}, "left vs right fist (real+imagined combined)"),
    5: Experiment((5, 9, 13, 6, 10, 14), {"T1": 0, "T2": 1}, "fists vs feet (real+imagined combined)"),
}

# Important!!!
# During Run 3, 7, 11 (Left vs. Right Hand), a single recording looks like this sequentially:
# [T0: Rest] ──► [T1: Move Left Hand] ──► [T0: Rest] ──► [T2: Move Right Hand] ──► [T0: Rest] ...
# T0 we are gonna ignore it, we just want the actions

# The question now, if each run do the same then why have 3, and not just a single long one?
# this is because each run is a separate recording, and the subject has to take a break between runs,
# so we have 3 runs to have more data and to have the subject not get tired
# remember each run is 4 minutes long, so 3 runs is 12 minutes of data, which is a lot for a subject to do in one sitting


def load_filtered_raw(subject: int, runs: tuple) -> mne.io.Raw:
    """Download (if needed), concatenate and band-pass filter the given runs."""
    edf_paths = eegbci.load_data(subject, list(runs), update_path=True, verbose=False)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in edf_paths]
    raw = mne.concatenate_raws(raws, verbose=False)

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    nyquist = raw.info["sfreq"] / 2.0
    notch_freqs = np.arange(POWERLINE_FREQ, nyquist, POWERLINE_FREQ).tolist()
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, verbose=False)
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design="firwin", verbose=False)

    return raw


def make_epochs(raw: mne.io.Raw, event_id: dict) -> mne.Epochs:
    events, all_event_id = mne.events_from_annotations(raw, verbose=False)
    missing = set(event_id) - set(all_event_id)
    if missing:
        raise ValueError(
            f"Annotations {missing} not found in this recording (found: {list(all_event_id)})"
        )
    # MNE raw files use arbitrary, unpredictable integer codes every time a file is recorded.
    # In one EDF file, MNE might assign "T1" -> 2 and "T2" -> 3.
    # In another file, "T1" might be assigned 10 and "T2" assigned 11
    # In our events_id's we have values 0 and 1 (for machine learning),
    # So with this picked_event_id we temporarily translates our labels so
    # MNE knows which continuous slices to extract from the raw file
    picked_event_id = {label: all_event_id[label] for label in event_id}
    print("Picked event_id mapping for this recording:", picked_event_id)
	# tmin=TMIN, tmax=TMAX (0.0 to 2.0 seconds): Cuts a 2-second window starting at the onset of each cue (t = 0s) up to t = 2s.
	
    # baseline=None: Disables baseline correction (subtracting pre-stimulus mean voltage).
    # Usually, researchers explicitly force the mean to zero when they cut the epochs. MNE has a built-in tool for this called baseline correction
    # So no baseline correction because the epoch starts at the cue onset (t=0),
    # so there is no pre-cue interval available for baseline estimation.
    
	# preload=True: Loads all epoched slice data into RAM memory as a 3D NumPy array immediately.
    epochs = mne.Epochs(
        raw, events, event_id=picked_event_id,
        tmin=TMIN, tmax=TMAX, baseline=None, preload=True, verbose=False,
    )
    return epochs


def load_experiment_data(subject: int, experiment: int):
    """Return (X, y, epochs) for one subject and one of the 6 experiments.

    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : ndarray, shape (n_epochs,) with values in {0, 1}
    """
    exp = EXPERIMENTS[experiment]
    raw = load_filtered_raw(subject, exp.runs)
    epochs = make_epochs(raw, exp.event_id)

    # the values in X are voltages in volts
    X = epochs.get_data(copy=True)
    # epochs.event_id maps each annotation label (e.g. "T1") to the integer
    # code MNE assigned it; remap those codes to our clean 0/1 class labels.
    # # 2. We match MNE's codes (2, 3) back to our binary target values (0, 1):
	# e.g -> code_to_label = {2: 0, 3: 1}
    code_to_label = {epochs.event_id[label]: exp.event_id[label] for label in exp.event_id}
    print(f"code_to_label mapping for this recording: {code_to_label}")
    y = np.array([code_to_label[code] for code in epochs.events[:, -1]])
    # when X shape: (45, 64, 321), y shape: (45,)
    # 45 is becasue -> in each 2-minute PhysioNet run, there are 15 cue events (either T1 or T2 motor tasks).
    # so Across all 3 concatenated runs, we have 45
    # 321 is because -> 2 seconds of data at 160 Hz sampling rate gives 2 * 160 = 320 samples, plus the initial sample at t=0 gives 321 samples per epoch.
    # Each epoch corresponds to one single cue event (either T1 or T2 motor tasks)
    # e.g Epoch 0 (Row 0 of X): The 2-second EEG recording for Cue #1.
    # Epoch 1 (Row 1 of X): The 2-second EEG recording for Cue #2.
    # ...
    # Epoch 44 (Row 44 of X): The 2-second EEG recording for Cue #45.

    # and the target vector y simply stores the correct answer (the label) for each of those individual cue events:
    # y[0] = 0 -> Epoch 0 was a Left Fist cue
    # y[1] = 1 -> Epoch 1 was a Right Fist cue.
    # so when we feed it 45 individual trial examples, and the model learns to look at
    # a single 2-second epoch to predict whether the person was thinking "Left" or "Right"!
    return X, y, epochs

def load_single_run_data(subject: int, run: int):
    """Return (X, y, epochs) for one individual PhysioNet run.

    The T1/T2 annotations are mapped to the same binary labels used by
    the experiments: T1 -> 0, T2 -> 1.
    """
    raw = load_filtered_raw(subject, (run,))
    epochs = make_epochs(raw, {"T1": 0, "T2": 1})

    X = epochs.get_data(copy=True)

    code_to_label = {
        epochs.event_id[label]: label_id
        for label, label_id in {"T1": 0, "T2": 1}.items()
    }

    y = np.array([
        code_to_label[code]
        for code in epochs.events[:, -1]
    ])

    return X, y, epochs
