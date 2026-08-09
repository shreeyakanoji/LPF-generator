"""
Streamlit UI for lfpgen: configure the LIF network + LFP proxy, run it, and
inspect/validate the resulting multi-channel LFP interactively.

Run locally:
    pip install -e .[app]
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    - push this repo to GitHub
    - on share.streamlit.io, point at streamlit_app.py
    - requirements.txt (included) covers the dependencies
    - Streamlit Cloud has no C/C++ compiler, so this app forces Brian2's
      "numpy" code-generation target below, which is pure-Python and works
      everywhere (a bit slower than the default "cython" target, which is
      fine for the demo-sized networks a Streamlit slider will realistically
      use).
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Make sure the local `lfpgen` package is importable regardless of how the
# app was launched or deployed. This is a defensive fallback: the primary
# fix is that requirements.txt installs the package itself (see below), but
# this guarantees `import lfpgen` works even if that step is ever skipped.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# --- Work around a Brian2 / Streamlit threading clash -----------------------
# Brian2's __init__.py calls signal.signal(signal.SIGINT, ...) at import time
# to install a Ctrl-C handler. signal.signal() only works when called from
# the main thread of the main interpreter -- but Streamlit runs the app
# script in a worker thread, so this raises:
#   ValueError: signal only works in main thread of the main interpreter
# We temporarily patch signal.signal to swallow that specific error for the
# duration of the brian2 import, then restore the real signal.signal so the
# rest of the app behaves normally.
import signal as _signal

_real_signal = _signal.signal


def _thread_safe_signal(sig, handler):
    try:
        return _real_signal(sig, handler)
    except ValueError:
        return None  # not in the main thread; skip installing the handler


_signal.signal = _thread_safe_signal
try:
    import brian2
    brian2.prefs.codegen.target = "numpy"  # portable; no C++ compiler needed
except ImportError:
    pass  # surfaced with a clear error message in the UI below
finally:
    _signal.signal = _real_signal
# -----------------------------------------------------------------------------


from lfpgen import NetworkParams, run_network, LinearProbe, LFPParams, compute_lfp, utils
from lfpgen import validation

st.set_page_config(page_title="lfpgen", layout="wide")
st.title("lfpgen — biophysically-grounded LFP generator")
st.caption(
    "LIF spiking network (Brian2) → multi-channel local field potential via the "
    "Mazzoni et al. (2015) AMPA/GABA current-summation proxy, on a virtual "
    "linear (Neuropixels-like) probe."
)

try:
    import brian2  # noqa: F401
except ImportError:
    st.error(
        "brian2 isn't installed in this environment. Install with "
        "`pip install -e .[app]` (or `pip install brian2`) and restart."
    )
    st.stop()

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Network")
    n_exc = st.slider("Excitatory neurons", 200, 8000, 2000, step=200)
    n_inh = st.slider("Inhibitory neurons", 50, 2000, 500, step=50)
    duration_ms = st.slider("Duration (ms)", 200, 5000, 1500, step=100)
    seed = st.number_input("Random seed", value=42, step=1)

    st.header("Probe")
    n_channels = st.slider("Channels", 4, 64, 32, step=4)
    span_um = st.slider("Probe span (um)", 200, 2000, 1000, step=100)

    st.header("LFP proxy")
    gamma = st.slider("GABA weight (gamma)", 0.0, 4.0, 1.65, step=0.05)
    delay_ms = st.slider("GABA delay (ms)", 0.0, 10.0, 3.0, step=0.5)
    gain = st.slider("Gain (uV per proxy unit)", 1.0, 200.0, 40.0, step=1.0)
    noise_uv = st.slider("Measurement noise (uV std)", 0.0, 30.0, 3.0, step=0.5)
    pink_uv = st.slider("Background 1/f noise (uV std)", 0.0, 40.0, 8.0, step=0.5)

    run_button = st.button("Run simulation", type="primary")

# --------------------------------------------------------------- session --
if run_button:
    net_params = NetworkParams(
        n_exc=n_exc, n_inh=n_inh, duration_ms=float(duration_ms), seed=int(seed),
    )
    with st.spinner("Simulating LIF network (numpy codegen; small nets run in seconds)..."):
        result = run_network(net_params, record_subset=min(1000, n_exc))

    probe = LinearProbe(n_channels=n_channels, span_um=float(span_um))
    lfp_params = LFPParams(
        gamma=gamma, delay_ms=delay_ms, gain_uv_per_na=gain,
        noise_uv_std=noise_uv, pink_noise_uv_std=pink_uv, seed=int(seed),
    )
    t_ms, lfp_uv = compute_lfp(result, probe, lfp_params)
    fs_hz = 1000.0 / (t_ms[1] - t_ms[0])

    st.session_state["result"] = result
    st.session_state["t_ms"] = t_ms
    st.session_state["lfp_uv"] = lfp_uv
    st.session_state["fs_hz"] = fs_hz
    st.session_state["probe"] = probe

# ---------------------------------------------------------------- output --
if "lfp_uv" in st.session_state:
    result = st.session_state["result"]
    t_ms = st.session_state["t_ms"]
    lfp_uv = st.session_state["lfp_uv"]
    fs_hz = st.session_state["fs_hz"]
    probe = st.session_state["probe"]

    st.success(f"Mean network firing rate: {result.firing_rate_hz:.2f} Hz  |  "
               f"{lfp_uv.shape[0]} channels x {lfp_uv.shape[1]} samples @ {fs_hz:.0f} Hz")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LFP traces")
        n_show = min(8, lfp_uv.shape[0])
        show_idx = np.linspace(0, lfp_uv.shape[0] - 1, n_show).astype(int)
        offset = np.max(np.abs(lfp_uv)) * 1.5 if lfp_uv.size else 1.0
        df = pd.DataFrame(
            {f"ch{ch}": lfp_uv[ch] + i * offset for i, ch in enumerate(show_idx)},
            index=t_ms,
        )
        st.line_chart(df, height=400)
        st.caption("Traces offset vertically for display; x-axis is time (ms).")

    with col2:
        st.subheader("Power spectrum")
        freqs, psd = utils.welch_psd(lfp_uv, fs_hz)
        psd_mean = psd.mean(axis=0)
        alpha = utils.spectral_slope(freqs, psd_mean)
        mask = (freqs > 0) & (freqs <= 200)
        psd_df = pd.DataFrame(
            {"PSD (uV^2/Hz)": psd_mean[mask]}, index=freqs[mask]
        )
        st.line_chart(np.log10(psd_df.clip(lower=1e-12)), height=400)
        st.caption(f"log10 PSD vs frequency (Hz). Fitted 1/f exponent: **{alpha:.2f}** "
                   f"(real cortical LFP is typically ~0.8-2 in the 2-100 Hz range).")

    st.subheader("Download")
    buf = io.BytesIO()
    np.savez(buf, t_ms=t_ms, lfp_uv=lfp_uv, fs_hz=fs_hz,
             channel_positions_um=probe.positions_um)
    st.download_button("Download simulated LFP (.npz)", buf.getvalue(),
                        file_name="lfp_output.npz")

    st.subheader("Validate against a real recording")
    st.caption(
        "Upload a reference LFP as .npy (shape: channels x samples, or 1-D) "
        "or .npz with keys `lfp_uv` and `fs_hz`."
    )
    ref_file = st.file_uploader("Reference recording", type=["npy", "npz"])
    ref_fs_manual = st.number_input("Reference sampling rate (Hz), if not in file",
                                     value=2500.0, step=100.0)

    if ref_file is not None:
        raw = ref_file.read()
        try:
            loaded = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            loaded = None

        if loaded is not None:
            if hasattr(loaded, "files"):  # npz
                ref_lfp = loaded["lfp_uv"] if "lfp_uv" in loaded.files else loaded[loaded.files[0]]
                ref_fs = float(loaded["fs_hz"]) if "fs_hz" in loaded.files else ref_fs_manual
            else:  # npy
                ref_lfp = loaded
                ref_fs = ref_fs_manual

            report = validation.compare_to_reference(lfp_uv, fs_hz, ref_lfp, ref_fs)
            st.text(report.summary())

            band_df = pd.DataFrame({
                "simulated": report.sim_band_power,
                "reference": report.ref_band_power,
            })
            st.bar_chart(band_df)
else:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
