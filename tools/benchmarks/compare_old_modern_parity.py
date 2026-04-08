import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pywt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter


DT_MS = 0.1
LOWCUT_HZ = 30
HIGHCUT_HZ = 120
WAVELET = "cgau5"
SCALE_LOW = 3
SCALE_HIGH = 32
N_SCALES = 50
SNIFF_DURATION_MS = 200
SKIP_FIRST_N_SNIFFS = 1
NUM_SNIFFS = 8


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def interpolate(x, y, dt):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = interp1d(x, y, kind="linear")
    newx = np.arange(x.min(), x.max(), step=dt)
    return newx, f(newx)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def load_wavelet(result_dir):
    t, lfp = load_pickle(result_dir / "lfp.pkl")
    t = np.asarray(t, dtype=float)
    lfp = np.asarray(lfp, dtype=float)
    t, lfp = interpolate(t, lfp, DT_MS)
    fs_hz = 1 / DT_MS * 1000
    lfp_bp = butter_bandpass_filter(lfp, LOWCUT_HZ, HIGHCUT_HZ, fs_hz, order=4)
    scales = np.linspace(SCALE_LOW / DT_MS, SCALE_HIGH / DT_MS, N_SCALES)
    cfs, freqs = pywt.cwt(lfp_bp, scales, WAVELET, DT_MS / 1000.0)
    power = np.log(1 + np.abs(cfs))
    step = int(round(SNIFF_DURATION_MS / DT_MS))
    chunks = [
        power[:, i * step:(i + 1) * step - 2]
        for i in range(NUM_SNIFFS + SKIP_FIRST_N_SNIFFS)[SKIP_FIRST_N_SNIFFS:]
    ]
    avg = sum(chunks)
    return t, lfp, lfp_bp, freqs, power, avg


def build_raw_report(old_dir, new_dir):
    old_inputs = load_pickle(old_dir / "input_times.pkl")
    new_inputs = load_pickle(new_dir / "input_times.pkl")
    old_inputs_sorted = sorted(
        ((seg, np.asarray(times, dtype=float)) for seg, times in old_inputs),
        key=lambda item: item[0],
    )
    new_inputs_sorted = sorted(
        ((seg, np.asarray(times, dtype=float)) for seg, times in new_inputs),
        key=lambda item: item[0],
    )
    segment_names_match = [seg for seg, _ in old_inputs_sorted] == [seg for seg, _ in new_inputs_sorted]
    max_input_diff = 0.0
    input_length_mismatch = False
    input_first_segment_mismatch = None
    for index, ((old_seg, old_times), (new_seg, new_times)) in enumerate(zip(old_inputs_sorted, new_inputs_sorted)):
        if old_seg != new_seg:
            input_first_segment_mismatch = {
                "index": index,
                "old": old_seg,
                "new": new_seg,
            }
            break
        n = min(len(old_times), len(new_times))
        if n:
            max_input_diff = max(max_input_diff, float(np.max(np.abs(old_times[:n] - new_times[:n]))))
        if len(old_times) != len(new_times):
            input_length_mismatch = True
            break

    old_t, old_lfp = load_pickle(old_dir / "lfp.pkl")
    new_t, new_lfp = load_pickle(new_dir / "lfp.pkl")
    old_t = np.asarray(old_t, dtype=float)
    new_t = np.asarray(new_t, dtype=float)
    old_lfp = np.asarray(old_lfp, dtype=float)
    new_lfp = np.asarray(new_lfp, dtype=float)
    n_lfp = min(len(old_t), len(new_t), len(old_lfp), len(new_lfp))

    old_vs = load_pickle(old_dir / "soma_vs.pkl")
    new_vs = load_pickle(new_dir / "soma_vs.pkl")
    old_labels = [row[0] for row in old_vs]
    new_labels = [row[0] for row in new_vs]
    old_dup = Counter(old_labels)
    new_dup = Counter(new_labels)
    rowwise = []
    label_mismatch_index = None
    for i, ((old_label, old_t_vec, old_v), (new_label, new_t_vec, new_v)) in enumerate(zip(old_vs, new_vs)):
        if old_label != new_label:
            label_mismatch_index = i
            break
        old_t_vec = np.asarray(old_t_vec, dtype=float)
        new_t_vec = np.asarray(new_t_vec, dtype=float)
        old_v = np.asarray(old_v, dtype=float)
        new_v = np.asarray(new_v, dtype=float)
        n = min(len(old_t_vec), len(new_t_vec), len(old_v), len(new_v))
        rowwise.append(
            {
                "label": old_label,
                "max_abs_time_diff": float(np.max(np.abs(old_t_vec[:n] - new_t_vec[:n]))) if n else 0.0,
                "max_abs_voltage_diff": float(np.max(np.abs(old_v[:n] - new_v[:n]))) if n else 0.0,
            }
        )
    worst_row = max(rowwise, key=lambda item: item["max_abs_voltage_diff"]) if rowwise else None

    return {
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "input_times": {
            "rows_old": len(old_inputs),
            "rows_new": len(new_inputs),
            "segment_names_match_after_sort": segment_names_match,
            "input_length_mismatch": input_length_mismatch,
            "first_segment_mismatch_after_sort": input_first_segment_mismatch,
            "max_abs_time_diff_after_sort": max_input_diff,
            "first_row_old": old_inputs[0][0],
            "first_row_new": new_inputs[0][0],
        },
        "lfp": {
            "len_old": int(len(old_t)),
            "len_new": int(len(new_t)),
            "common_prefix_len": int(n_lfp),
            "max_abs_time_diff_common_prefix": float(np.max(np.abs(old_t[:n_lfp] - new_t[:n_lfp]))) if n_lfp else 0.0,
            "max_abs_value_diff_common_prefix": float(np.max(np.abs(old_lfp[:n_lfp] - new_lfp[:n_lfp]))) if n_lfp else 0.0,
            "extra_tail_old": int(len(old_t) - n_lfp),
            "extra_tail_new": int(len(new_t) - n_lfp),
        },
        "soma_vs": {
            "rows_old": len(old_vs),
            "rows_new": len(new_vs),
            "unique_labels_old": len(old_dup),
            "unique_labels_new": len(new_dup),
            "label_sequences_match_exactly": old_labels == new_labels,
            "first_label_mismatch_index": label_mismatch_index,
            "rowwise_common_prefix": len(rowwise),
            "worst_rowwise_voltage_diff": worst_row,
            "top_duplicate_count_old": old_dup.most_common(10),
            "top_duplicate_count_new": new_dup.most_common(10),
        },
    }


def build_wavelet_report(old_dir, new_dir):
    old_t, old_lfp, old_bp, _freqs, old_power, old_avg = load_wavelet(old_dir)
    new_t, new_lfp, new_bp, _freqs2, new_power, new_avg = load_wavelet(new_dir)
    n = min(len(old_lfp), len(new_lfp))
    pn = min(old_power.shape[1], new_power.shape[1])
    an = min(old_avg.shape[1], new_avg.shape[1])
    return {
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "dt_ms": DT_MS,
        "bandpass_hz": [LOWCUT_HZ, HIGHCUT_HZ],
        "wavelet": WAVELET,
        "raw_lfp_max_abs_diff": float(np.max(np.abs(old_lfp[:n] - new_lfp[:n]))),
        "bp_lfp_max_abs_diff": float(np.max(np.abs(old_bp[:n] - new_bp[:n]))),
        "full_wavelet_power_max_abs_diff": float(np.max(np.abs(old_power[:, :pn] - new_power[:, :pn]))),
        "sniff_average_wavelet_power_max_abs_diff": float(np.max(np.abs(old_avg[:, :an] - new_avg[:, :an]))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_dir")
    parser.add_argument("new_dir")
    parser.add_argument("raw_out")
    parser.add_argument("wavelet_out")
    args = parser.parse_args()

    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    raw_out = Path(args.raw_out)
    wavelet_out = Path(args.wavelet_out)

    raw_report = build_raw_report(old_dir, new_dir)
    wavelet_report = build_wavelet_report(old_dir, new_dir)

    raw_out.write_text(json.dumps(raw_report, indent=2, sort_keys=True))
    wavelet_out.write_text(json.dumps(wavelet_report, indent=2, sort_keys=True))

    print(raw_out)
    print(wavelet_out)


if __name__ == "__main__":
    main()
