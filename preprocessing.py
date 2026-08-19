import argparse
import os
import sys

import matplotlib
import mne
from mne.datasets import eegbci
import numpy as np
import matplotlib.pyplot as plt

from wavelets import cwt_scalogram, DEFAULT_FREQS

# Frequency band kept for motor imagery: mu (~8-12 Hz) + beta (~13-30 Hz)
LOW_FREQ = 7.0
HIGH_FREQ = 30.0
POWERLINE_FREQ = 60.0  # PhysioNet data was recorded in the US (60 Hz mains)

VALID_SUBJECTS = range(1, 110)  # PhysioNet EEGBCI: subjects 1-109
VALID_RUNS = range(3, 15)       # runs 3-14 (1-2 are eyes-open/closed baseline, not motor imagery)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse, visualize and filter EEG motor-imagery data.")
    parser.add_argument("subject", type=int, help="PhysioNet subject id (1-109)")
    parser.add_argument("runs", type=int, nargs="+", help="One or more PhysioNet run numbers (3-14)")
    parser.add_argument("--show", action="store_true", help="Display figures interactively instead of saving them")
    parser.add_argument("--outdir", default="outputs", help="Directory to save figures into (default: outputs)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose debug information")
    parser.add_argument("--wavelet", action="store_true", help="Bonus: also plot a wavelet scalogram (time-frequency, not just frequency)")
    args = parser.parse_args()

    # Validate up front so a bad subject/run gives an immediate, clear
    # message instead of a raw traceback from deep inside MNE/pooch.
    if args.subject not in VALID_SUBJECTS:
        parser.error(f"subject must be between {VALID_SUBJECTS.start} and {VALID_SUBJECTS.stop - 1} (got {args.subject})")
    invalid_runs = [r for r in args.runs if r not in VALID_RUNS]
    if invalid_runs:
        parser.error(f"runs must be between {VALID_RUNS.start} and {VALID_RUNS.stop - 1} (got invalid: {invalid_runs})")

    return args


def load_raw(subject: int, runs: list[int], verbose: bool = False) -> mne.io.Raw:
    """Download (if needed) and load the requested runs, concatenated into one Raw object."""
    def _log(message):
        if verbose:
            print(message)
    _log(f"Fetching subject {subject}, runs {runs} from PhysioNet ...")
    edf_paths = eegbci.load_data(subject, runs, update_path=True, verbose=False)
    _log(f"EDF files found: {edf_paths}")

    raws = [mne.io.read_raw_edf(path, preload=True, verbose=False) for path in edf_paths]
    for path, r in zip(edf_paths, raws):
        events, event_id = mne.events_from_annotations(r, verbose=False)
        _log(f"File: {r.filenames[0].name} - {len(events)} events - types: {event_id}")

    # Standardize channel names and montage per-file before concatenating,
    # since individual runs can otherwise disagree slightly.
    for r in raws:
        eegbci.standardize(r)
        montage = mne.channels.make_standard_montage("standard_1020")
        r.set_montage(montage, on_missing="ignore")
    raw = mne.concatenate_raws(raws, on_mismatch="ignore", verbose=False)

    _log(f"sfreq={raw.info['sfreq']} Hz, duration={raw.times[-1] - raw.times[0]:.1f}s, shape={raw.get_data().shape}")

    return raw


def plot_and_save(fig, outdir: str, name: str, show: bool):
    if show:
        fig.show()
    else:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")


def plot_scalogram(raw: mne.io.Raw, outdir: str, tag: str, stage: str, show: bool, duration: float = 10.0, channel: int = 0):
    """Bonus: wavelet scalogram (time-frequency power) of one channel, the
    wavelet analogue of the Fourier-based PSD plots above. Unlike a PSD
    plot -- which only tells you a frequency was present *somewhere* in the
    window -- this shows *when* mu/beta power rises and falls, which is
    what actually happens during a motor-imagery cue."""

    sfreq = raw.info["sfreq"]
    # This needs the amount of samples
    n_samples = int(duration * sfreq)
    # The [0] is because this always returns a 2D array of shape (1, n_samples)
    # and we just want the 1D array of shape (n_samples,)
    data = raw.get_data(picks=[channel], start=0, stop=n_samples)[0]

    power, freqs, times = cwt_scalogram(data, sfreq, freqs=DEFAULT_FREQS)

    fig, ax = plt.subplots(figsize=(9, 4))
    # with this we create a nice heatmap of the power across time and frequency, with a smooth gradient
    # vmin=0.0, vmax=1e-8,   Locks the colorbar ceiling across all calls
    mesh = ax.pcolormesh(times, freqs, power, vmin=0.0, vmax=1e-8, shading="gouraud", cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Wavelet scalogram - {stage} - channel {raw.ch_names[channel]}")
    fig.colorbar(mesh, ax=ax, label="Power")
    plot_and_save(fig, outdir, f"{tag}_wavelet_scalogram_{stage.replace(' ', '_')}", show)


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")  # headless backend, must be set before pyplot is touched

    def _log(message):
        if args.verbose:
            print(message)

    try:
        raw = load_raw(args.subject, args.runs, verbose=args.verbose)
    except ValueError as e:
        # e.g. a subject/run combination PhysioNet doesn't actually have on record
        print(f"Error: no PhysioNet data for subject {args.subject}, runs {args.runs}.")
        print(f"  ({e})")
        print(f"Valid ranges: subject {VALID_SUBJECTS.start}-{VALID_SUBJECTS.stop - 1}, "
              f"runs {VALID_RUNS.start}-{VALID_RUNS.stop - 1}.")
        sys.exit(1)
    except OSError as e:
        # Handles OS/file-related download and access errors
        print(f"Error: could not download EEG data for subject {args.subject}, runs {args.runs}.")
        print(f"  ({e})")
        print("Check your internet connection and that physionet.org is reachable, then try again.")
        sys.exit(1)

    try:
        # Unique tag for this subject/run combination, used in every saved
        # filename so different invocations don't overwrite each other's output.
        runs_str = "_".join(f"r{r:02d}" for r in sorted(args.runs))
        tag = f"subject{args.subject:03d}_{runs_str}"
        _log(f"--- Tag for this run: {tag} ---")

        _log("\n--- Raw info ---")
        _log(raw.info)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        _log(f"\nEvents found: {event_id}, total annotated events: {len(events)}")

        # 1) Visualize the raw, unfiltered signal.
        # BLUE is when the subject was relaxing and PINK is when the subject was performing motor imagery (or real movement)
        fig_raw = raw.plot(n_channels=20, duration=10, scalings=dict(eeg=50e-6), title="Raw EEG (unfiltered)", show=False)
        plot_and_save(fig_raw, args.outdir, f"{tag}_01_raw_timeseries", args.show)

		# the fmax=80 argument limits the x-axis to 0-80 Hz, which is the relevant range for EEG analysis and avoids clutter from higher frequencies that are not of interest.
        fig_psd_raw = raw.compute_psd(fmax=80).plot(show=False)
        fig_psd_raw.suptitle("Power spectral density - before filtering")
        plot_and_save(fig_psd_raw, args.outdir, f"{tag}_02_raw_psd", args.show)

        if args.wavelet:
            plot_scalogram(raw, args.outdir, tag, "before filtering", args.show)

        # 2) Filter: keep the mu/beta band useful for motor imagery, and notch
        #    out the 60 Hz US mains hum. PhysioNet's EEGBCI data is sampled at
        #    160 Hz (Nyquist = 80 Hz), so only the fundamental fits below Nyquist;
        #    harmonics above it are skipped automatically.
        nyquist = raw.info["sfreq"] / 2.0
        _log(f"raw data frequency: {raw.info['sfreq']} Hz | Nyquist frequency: {nyquist} Hz")
        notch_freqs = np.arange(POWERLINE_FREQ, nyquist, POWERLINE_FREQ).tolist()
        _log(f"Notch filtering out {notch_freqs} Hz ...")

        raw_filtered = raw.copy()
        if notch_freqs:
            raw_filtered.notch_filter(freqs=notch_freqs, verbose=False)
        # method: Selects the overall filter family/architecture. Its primary choices are 'fir' (the default) or 'iir'.
		# in this case method we use the default 'fir', which is a Finite Impulse Response filter, which is generally recommended for EEG data.
		# fir_design: Selects the specific mathematical algorithm used to construct an FIR filter when method='fir' is selected.
		# fir_design supports two values: "firwin" and "firwin2". "firwin" is the default and is generally recommended for most applications.
		# and here comes the million dollar question: If I just want to filter my data, why do I care about specifying fir_design="firwin"?
		# FIRWIN stands for FIR (Finite Impulse Response) Windowing.
		# firwin designs the FIR filter using the window-method approach,
		# providing a smooth transition instead of an ideal brick-wall cutoff.
        raw_filtered.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design="firwin", verbose=False)

        # 3) Visualize again, after filtering.
        fig_filtered = raw_filtered.plot(
            n_channels=20, duration=10, scalings=dict(eeg=50e-6),
            title=f"Filtered EEG ({LOW_FREQ}-{HIGH_FREQ} Hz band-pass + {POWERLINE_FREQ} Hz notch)",
            show=False,
        )
        plot_and_save(fig_filtered, args.outdir, f"{tag}_03_filtered_timeseries", args.show)

        fig_psd_filtered = raw_filtered.compute_psd(fmax=80).plot(show=False)
        fig_psd_filtered.suptitle("Power spectral density - after filtering")
        plot_and_save(fig_psd_filtered, args.outdir, f"{tag}_04_filtered_psd", args.show)
        _log(f"raw_filtered shape: {raw_filtered.get_data().shape}, "
             f"duration: {raw_filtered.times[-1] - raw_filtered.times[0]:.1f}s")

        if args.wavelet:
            plot_scalogram(raw_filtered, args.outdir, tag, "after filtering", args.show)

        # Save the filtered raw so the next stage (pipeline/training script) can
        # load it directly instead of re-downloading and re-filtering.
        os.makedirs(args.outdir, exist_ok=True)
        fif_path = os.path.join(args.outdir, f"{tag}_filtered_raw.fif")
        raw_filtered.save(fif_path, overwrite=True)
        _log(f"\nSaved filtered raw to {fif_path}")

        if args.show:
            plt.show()

    except Exception as e:
        # Final safety net: nothing past this point should ever surface a
        # raw traceback -- print a clean message and exit instead.
        print(f"Unexpected error while processing subject {args.subject}, runs {args.runs}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
