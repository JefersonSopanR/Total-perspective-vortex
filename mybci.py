"""
mybci.py

Part V.1.2 — Treatment pipeline / V.1.4 — Train, Validation and Test
=======================================================================
Builds the full sklearn Pipeline (our own CSP + a classifier), trains it
with cross-validation, and "plays back" a run epoch-by-epoch to simulate
real-time prediction (each epoch must be classified in under 2 seconds).

Now driven by data.py's canonical 0-5 experiment definitions instead of
raw run-number lists.

Usage
-----
    python mybci.py <subject> <experiment|run> train
    python mybci.py <subject> <experiment|run> predict 
    python mybci.py                                    # all subjects, all 6 experiments
    python mybci.py --quick                             # fast sanity check: subjects 1-10 only
    python mybci.py --restart                           # clear checkpoint and start the full sweep over

The full sweep (no args) checkpoints every (experiment, subject) result to
models/all_subjects_checkpoint.csv as it goes, and resumes from there
automatically if re-run after being interrupted. Use --restart to discard
an existing checkpoint and start clean.

Examples
--------
    python mybci.py 4 1 train      # subject 4, experiment 1 (imagined left/right fist)
    python mybci.py 4 1 predict
    python mybci.py 4 14 train       # individual run 14
    python mybci.py 4 14 predict
    python mybci.py --quick
    python mybci.py
"""

import csv
import os
import sys
import time

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from data import EXPERIMENTS, load_experiment_data, load_single_run_data
from csp import CSP
from customLDA import CustomLDA
from wavelets import WaveletBandPower

N_COMPONENTS = 6
MODEL_DIR = "models"
CV_SPLITS_SINGLE_RUN = 10
CV_SPLITS_FULL_SWEEP = 5 
CHECKPOINT_PATH = "models/all_subjects_checkpoint.csv"

TRAIN_SIZE = 0.85
TEST_SIZE = 0.15
CV_SIZE = 0.15
RANDOM_STATE = 42

VALID_SUBJECTS = range(1, 110)  # PhysioNet EEGBCI: subjects 1-109


# "csp", CSP(n_components=N_COMPONENTS)
# "wavelets", WaveletBandPower()
# "clf", CustomLDA(plot=True, verbose=True)
# "lda", LinearDiscriminantAnalysis()
# "lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
def build_pipeline() -> Pipeline:
    """The pipeline required by V.1.2: our own CSP (BaseEstimator +
    TransformerMixin) feeding a standard sklearn classifier."""
    return Pipeline([
        ("csp", CSP(n_components=N_COMPONENTS)),
        ("clf", CustomLDA()),
    ])


def parse_task(task: str):
    """
    Return ("experiment", id) or ("run", id).

    Existing numeric 0-5 values remain experiments.
    Numeric 6-14 are interpreted as individual runs.
    Runs 3-5 can be requested explicitly as run3, run4, run5.
    """
    if task.startswith("run"):
        run_id = int(task[3:])

        if run_id not in range(3, 15):
            raise ValueError(
                "Run must be between 3 and 14."
            )

        return "run", run_id

    task_id = int(task)

    if task_id in EXPERIMENTS:
        return "experiment", task_id

    if task_id in range(6, 15):
        return "run", task_id

    raise ValueError(
        "Task must be an experiment 0-5, "
        "a run 6-14, or an explicit run3-run14."
    )


def load_task_data(subject: int, task_type: str, task_id: int):
    """Load either one canonical experiment or one individual run."""
    if task_type == "experiment":
        return load_experiment_data(subject, task_id)

    return load_single_run_data(subject, task_id)


def task_model_path(subject: int, task_type: str, task_id: int) -> str:
    """Return a separate model path for experiments and individual runs."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if task_type == "experiment":
        filename = (
            f"subject{subject:03d}_exp{task_id}_pipeline.joblib"
        )
    else:
        filename = (
            f"subject{subject:03d}_run{task_id}_pipeline.joblib"
        )

    return os.path.join(MODEL_DIR, filename)


def task_test_indices_path(
    subject: int,
    task_type: str,
    task_id: int
) -> str:
    """Return the path used to save the fixed held-out test indices."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if task_type == "experiment":
        filename = (
            f"subject{subject:03d}_exp{task_id}_test_indices.npy"
        )
    else:
        filename = (
            f"subject{subject:03d}_run{task_id}_test_indices.npy"
        )

    return os.path.join(MODEL_DIR, filename)

def split_data(X, y):
    """
    Split the dataset deterministically into:

        85% train
        15% test

    The test set is kept completely separate until final evaluation.
    """
    if not np.isclose(TRAIN_SIZE + TEST_SIZE, 1.0):
        raise ValueError("Train and test sizes must sum to 1.")

    # First separate the final 15% test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def train(subject: int, task_type: str, task_id: int):
    if task_type == "experiment":
        label = EXPERIMENTS[task_id].description
    else:
        label = f"individual run {task_id}"
    print(f"Loading + epoching subject {subject}, {task_type} {task_id} ({label}) ...")
    X, y, _epochs = load_task_data(subject,task_type,task_id)
    print(f"{X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} time samples")

    (X_train, X_test, y_train, y_test) = split_data(X, y)
    print("\nDataset split:")
    print(f"Train:      {X_train.shape[0]} epochs")
    print(f"Test:       {X_test.shape[0]} epochs")
    pipeline = build_pipeline()
    cv = StratifiedShuffleSplit(n_splits=CV_SPLITS_SINGLE_RUN, test_size=CV_SIZE, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv)

    print(np.array2string(scores, precision=4))
    print(f"cross_val_score: {scores.mean():.4f}")

    # Fit the final pipeline on all development/training data
    pipeline.fit(X_train, y_train)

    # TEST has not been used anywhere above.
    # Final evaluation on NEVER-LEARNED TEST data.
    test_predictions = pipeline.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Test accuracy:       {test_accuracy:.4f}")

    # The saved pipeline has been trained on TRAIN,
    # but it has never seen TEST
    
    # Save trained pipeline.
    path = task_model_path(subject,task_type,task_id)
    joblib.dump(pipeline, path)
    print(f"Saved trained pipeline to {path}")
    # Save the original indices so predict() can replay precisely
    # the same held-out test epochs.
    all_indices = np.arange(X.shape[0])

    # Recreate the same deterministic split on indices.
    _, test_indices = train_test_split(all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    np.save(task_test_indices_path(subject, task_type, task_id), test_indices)
    return test_accuracy


def predict(subject: int, task_type: str, task_id: int):
    path = task_model_path(subject, task_type, task_id)

    try:
        pipeline = joblib.load(path)
    except FileNotFoundError:
        print(f"No trained model found at {path}. Run 'train' for this subject/experiment first.")
        return None

    indices_path = task_test_indices_path(subject, task_type, task_id)
    try:
        test_indices = np.load(indices_path)
    except FileNotFoundError:
        print(
            "No saved test split found. "
            "Run 'train' for this subject/experiment first."
        )
        return None
    
    if task_type == "experiment":
        label = EXPERIMENTS[task_id].description
    else:
        label = f"individual run {task_id}"
    print(f"Loading + epoching subject {subject}, {task_type} {task_id} ({label}) (playback) ...")
    print("NOTE: playback uses the held-out test set that was never used to fit the saved pipeline.")
    X, y, _epochs = load_task_data(subject, task_type, task_id)
    X_test = X[test_indices]
    y_test = y[test_indices]
    print(f"Number of held-out test epochs: {X_test.shape[0]}")

    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i in range(X_test.shape[0]):
        epoch = X_test[i:i + 1]  # keep the (1, n_channels, n_times) batch shape

        t0 = time.perf_counter()
        pred = pipeline.predict(epoch)[0]
        elapsed = time.perf_counter() - t0

        truth = y_test[i]
        equal = bool(pred == truth)
        correct += equal

        flag = "  ! exceeded 2s budget" if elapsed > 2.0 else ""
        print(f"epoch {i:02d}: [{pred}] [{truth}] {equal}{flag}")

    accuracy = correct / X_test.shape[0]
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def _load_checkpoint(path: str) -> dict:
    """Read already-computed (experiment, subject) -> accuracy pairs from a
    previous (possibly interrupted) run, so run_all_subjects() can resume
    instead of redoing hours of work."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done[(int(row["experiment"]), int(row["subject"]))] = float(row["accuracy"])
    return done


def _append_checkpoint(path: str, exp_id: int, subject: int, accuracy: float):
    """Append one result immediately -- so if the process dies mid-sweep,
    everything computed so far is already safely on disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["experiment", "subject", "accuracy"])
        writer.writerow([exp_id, subject, f"{accuracy:.6f}"])
        f.flush()
        os.fsync(f.fileno())


def run_all_subjects(subjects=range(1, 110), checkpoint_path: str = CHECKPOINT_PATH, resume: bool = True):
    """No-argument mode: train+evaluate all 6 experiments across subjects,
    matching the mean-accuracy table format from the project subject.

    Writes every (experiment, subject) result to `checkpoint_path` as soon
    as it's computed, and -- if resume=True (the default) -- skips any
    pairs already present in that file on startup. This is a multi-hour
    sweep (109 subjects x 6 experiments); without this, an interruption at
    any point would mean starting over from scratch.
    """
    done = _load_checkpoint(checkpoint_path) if resume else {}
    if done:
        print(f"Resuming from checkpoint: {len(done)} (experiment, subject) results already computed.")

    experiment_means = {}

    for exp_id, exp in EXPERIMENTS.items():
        subject_scores = []
        for subject in subjects:
            if (exp_id, subject) in done:
                acc = done[(exp_id, subject)]
                subject_scores.append(acc)
                print(f"experiment {exp_id}: subject {subject:03d}: accuracy = {acc:.4f}  (from checkpoint)")
                continue

            try:
                X, y, _epochs = load_experiment_data(subject, exp_id)
                (X_train, X_test,y_train, y_test) = split_data(X, y)
                pipeline = build_pipeline()
                cv = StratifiedShuffleSplit(n_splits=CV_SPLITS_FULL_SWEEP, test_size=CV_SIZE, random_state=RANDOM_STATE)
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
                cv_accuracy = scores.mean()
                pipeline.fit(X_train, y_train)

                # TEST remains completely unseen until this point.
                test_predictions = pipeline.predict(X_test)

                test_accuracy = np.mean(test_predictions == y_test)

                acc = test_accuracy
            except Exception as exc:  # missing/corrupt data for a subject, skip it
                print(f"experiment {exp_id}: subject {subject:03d}: FAILED ({exc})")
                continue

            subject_scores.append(acc)
            print(
                f"experiment {exp_id}: "
                f"subject {subject:03d}: "
                f"CV = {cv_accuracy:.4f}, "
                f"test = {test_accuracy:.4f}")
            # _append_checkpoint(checkpoint_path, exp_id, subject, acc)

        experiment_means[exp_id] = np.mean(subject_scores) if subject_scores else float("nan")

    print("\nMean accuracy of the six different experiments:")
    for exp_id, exp in EXPERIMENTS.items():
        print(f"experiment {exp_id} ({exp.description}): accuracy = {experiment_means[exp_id]:.4f}")

    overall = np.nanmean(list(experiment_means.values()))
    print(f"\nMean accuracy of 6 experiments: {overall:.4f}")
    return experiment_means


def main():
    argv = sys.argv[1:]

    if len(argv) == 0 or argv[0] in ("--quick", "--restart"):
        quick = "--quick" in argv
        restart = "--restart" in argv
        subjects = range(1, 11) if quick else range(1, 110)
        checkpoint = "models/quick_checkpoint.csv" if quick else CHECKPOINT_PATH
        if restart and os.path.exists(checkpoint):
            os.remove(checkpoint)
            print(f"--restart: cleared existing checkpoint at {checkpoint}")
        if quick:
            print("--quick: running only subjects 1-10 for a fast sanity check "
                  "(not the real evaluation number -- drop --quick for the full sweep).")
        try:
            run_all_subjects(subjects=subjects, checkpoint_path=checkpoint)
        except Exception as exc:
            # Per-subject failures are already handled inside run_all_subjects();
            # this is a last-resort net for anything outside that loop
            # (e.g. a corrupted checkpoint file).
            print(f"Unexpected error during the full sweep: {exc}")
            sys.exit(1)
        return

    if len(argv) != 3 or argv[-1] not in ("train", "predict"):
        print(__doc__)
        sys.exit(1)

    try:
        subject = int(argv[0])
    except ValueError:
        print(f"Invalid subject: {argv[0]!r} is not a number.")
        sys.exit(1)

    if subject not in VALID_SUBJECTS:
        print(f"Invalid subject: {subject}. Must be between "
              f"{VALID_SUBJECTS.start} and {VALID_SUBJECTS.stop - 1}.")
        sys.exit(1)

    task = argv[1]
    mode = argv[2]

    try:
        task_type, task_id = parse_task(task)
    except (ValueError, TypeError) as exc:
        print(f"Invalid task: {exc}")
        sys.exit(1)

    try:
        if mode == "train":
            train(subject, task_type, task_id)
        else:
            predict(subject, task_type, task_id)
    except (ValueError, OSError) as exc:
        # ValueError: e.g. PhysioNet has no data for this subject/task combo.
        # OSError: covers filesystem-related errors.
        # Network errors are handled by the final Exception safety net.
        print(f"Error loading/processing data for subject {subject}, {task_type} {task_id}: {exc}")
        print("Check that the subject and task are valid, and that physionet.org is reachable.")
        sys.exit(1)
    except Exception as exc:
        # Final safety net so nothing else can surface a raw traceback.
        print(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
