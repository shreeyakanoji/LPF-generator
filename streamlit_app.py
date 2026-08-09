"""
lfpgen Streamlit app -- SELF-CONTAINED, single-file version.

This file has ZERO dependency on any other file in your repo (no `lfpgen`
package folder needed). Just drop this .py file into your repo root next to
a requirements.txt containing:

    brian2>=2.5
    numpy>=1.22
    scipy>=1.8
    streamlit>=1.30
    pandas

...and point Streamlit Cloud at this file. That's it.

It contains the same biophysically-grounded LFP method as the full lfpgen
package (LIF E/I network in Brian2 -> Mazzoni et al. 2015 AMPA/GABA
current-summation LFP proxy on a virtual linear probe), just inlined so
there's no import path to break.

Run locally:
    pip install brian2 numpy scipy streamlit pandas
    streamlit run streamlit_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

# --- Work around a Brian2 / Streamlit threading clash -----------------------
# Brian2's __init__.py calls signal.signal(signal.SIGINT, ...) at import time
# to install a Ctrl-C handler. signal.signal() only works when called from
# the main thread of the main interpreter -- but Streamlit runs the app
# script in a worker thread, so this raises:
#   ValueError: signal only works in main thread of the main interpreter
# We temporarily patch signal.signal to swallow that specific error for the
# duration of the brian2 import, then restore the real signal.signal.
import signal as _signal

_real_signal = _signal.signal


def _thread_safe_signal(sig, handler):
    try:
        return _real_signal(sig, handler)
    except ValueError:
        return None  # not in the main thread; skip installing the handler


_signal.signal = _thread_safe_signal
try:
    import brian2 as b2
    b2.prefs.codegen.target = "numpy"  # portable; no C++ compiler needed on cloud
    _BRIAN2_OK = True
except ImportError:
    _BRIAN2_OK = False
finally:
    _signal.signal = _real_signal
# -----------------------------------------------------------------------------

st.set_page_config(page_title="lfpgen", layout="wide")
st.title("lfpgen — biophysically-grounded LFP generator")
st.caption(
    "LIF spiking network (Brian2) -> multi-channel local field potential via the "
    "Mazzoni et al. (2015) AMPA/GABA current-summation proxy, on a virtual "
    "linear (Neuropixels-like) probe."
)

if not _BRIAN2_OK:
    st.error(
        "brian2 isn't installed in this environment. Make sure your "
        "requirements.txt includes `brian2>=2.5`, then reboot the app."
    )
    st.stop()


# ============================================================================
# Core simulation + LFP-proxy logic (inlined from the lfpgen package)
# ============================================================================

def run_network(n_exc, n_inh, duration_ms, seed, p_connect=0.02, dt_ms=0.1,
                 tau_m_ms=20.0, v_rest_mv=-70.0, v_reset_mv=-60.0, v_thresh_mv=-50.0,
                 t_refractory_ms=2.0, tau_ampa_ms=2.0, tau_gaba_ms=10.0,
                 g_ampa_nS=0.6, g_gaba_nS=2.7, e_ampa_mv=0.0, e_gaba_mv=-80.0,
                 r_m_Mohm=100.0, ext_rate_hz=2500.0, ext_g_nS=0.6,
                 probe_span_um=1000.0, record_subset=1000):
    """Simulate an E/I LIF network in Brian2, returning recorded synaptic
    currents (used downstream to build the LFP proxy) and firing rate."""
    from brian2 import ms, mV, nS, Hz

    rng = np.random.default_rng(seed)
    b2.seed(int(seed))
    b2.start_scope()
    b2.defaultclock.dt = dt_ms * ms

    n_record = min(record_subset, n_exc)

    eqs = """
    dv/dt = ((v_rest_mv*mV - v) + r_m*(g_ampa*(e_ampa_mv*mV - v) + g_gaba*(e_gaba_mv*mV - v))) / tau : volt (unless refractory)
    dg_ampa/dt = -g_ampa / tau_ampa : siemens
    dg_gaba/dt = -g_gaba / tau_gaba : siemens
    tau : second
    tau_ampa : second
    tau_gaba : second
    r_m : ohm
    v_rest_mv : 1
    e_ampa_mv : 1
    e_gaba_mv : 1
    """

    neurons = b2.NeuronGroup(
        n_exc + n_inh, eqs,
        threshold="v > v_thresh_mv_param*mV",
        reset="v = v_reset_mv_param*mV",
        refractory=t_refractory_ms * ms,
        method="euler",
        namespace={"v_thresh_mv_param": v_thresh_mv, "v_reset_mv_param": v_reset_mv},
    )
    neurons.tau = tau_m_ms * ms
    neurons.tau_ampa = tau_ampa_ms * ms
    neurons.tau_gaba = tau_gaba_ms * ms
    neurons.r_m = r_m_Mohm * b2.Mohm
    neurons.v_rest_mv = v_rest_mv
    neurons.e_ampa_mv = e_ampa_mv
    neurons.e_gaba_mv = e_gaba_mv
    neurons.v = v_rest_mv * mV + (rng.random(n_exc + n_inh) * 10 - 5) * mV

    exc = neurons[:n_exc]
    inh = neurons[n_exc:]

    syn_ee = b2.Synapses(exc, exc, on_pre="g_ampa_post += g_ampa_val*nS",
                          namespace={"g_ampa_val": g_ampa_nS})
    syn_ei = b2.Synapses(exc, inh, on_pre="g_ampa_post += g_ampa_val*nS",
                          namespace={"g_ampa_val": g_ampa_nS})
    syn_ie = b2.Synapses(inh, exc, on_pre="g_gaba_post += g_gaba_val*nS",
                          namespace={"g_gaba_val": g_gaba_nS})
    syn_ii = b2.Synapses(inh, inh, on_pre="g_gaba_post += g_gaba_val*nS",
                          namespace={"g_gaba_val": g_gaba_nS})
    for syn in (syn_ee, syn_ei, syn_ie, syn_ii):
        syn.connect(p=p_connect)

    b2.PoissonInput(neurons, "g_ampa", N=1, rate=ext_rate_hz * Hz, weight=ext_g_nS * nS)

    spikes_exc = b2.SpikeMonitor(exc)
    spikes_inh = b2.SpikeMonitor(inh)

    record_idx = np.sort(rng.choice(n_exc, size=n_record, replace=False))
    state_mon = b2.StateMonitor(exc, ["g_ampa", "g_gaba", "v"], record=record_idx,
                                 dt=dt_ms * ms)

    b2.run(duration_ms * ms)

    g_ampa = state_mon.g_ampa[:] / b2.nS
    g_gaba = state_mon.g_gaba[:] / b2.nS
    v_rec = state_mon.v[:] / b2.mV
    t_ms = state_mon.t[:] / b2.ms

    i_ampa = (g_ampa * (v_rec - e_ampa_mv)) / 1000.0  # nA
    i_gaba = (g_gaba * (v_rec - e_gaba_mv)) / 1000.0  # nA

    positions = rng.uniform(0.0, probe_span_um, size=n_exc)
    positions_recorded = positions[record_idx]

    total_spikes = spikes_exc.num_spikes + spikes_inh.num_spikes
    duration_s = duration_ms / 1000.0
    firing_rate_hz = total_spikes / ((n_exc + n_inh) * duration_s)

    return {
        "t_ms": t_ms, "i_ampa": i_ampa, "i_gaba": i_gaba,
        "positions_um": positions_recorded, "firing_rate_hz": firing_rate_hz,
    }


def probe_positions(n_channels, span_um):
    return np.linspace(0.0, span_um, n_channels)


def probe_distances(channel_positions_um, source_positions_um, min_distance_um=20.0):
    d = np.abs(channel_positions_um[:, None] - source_positions_um[None, :])
    return np.maximum(d, min_distance_um)


def _pink_noise(n_channels, n_steps, rng):
    white = rng.normal(size=(n_channels, n_steps))
    spec = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_steps)
    freqs[0] = freqs[1] if n_steps > 1 else 1.0
    shaping = 1.0 / np.sqrt(freqs)
    shaping[0] = shaping[1]
    pink = np.fft.irfft(spec * shaping[None, :], n=n_steps, axis=1)
    pink /= pink.std(axis=1, keepdims=True) + 1e-12
    return pink


def compute_lfp(sim_result, channel_positions_um, gamma=1.65, delay_ms=3.0,
                 noise_uv_std=3.0, pink_noise_uv_std=8.0, gain_uv_per_na=40.0, seed=0):
    """Mazzoni et al. (2015) AMPA/GABA current-summation LFP proxy, applied
    across a linear electrode array with inverse-distance weighting."""
    rng = np.random.default_rng(seed)

    i_ampa = sim_result["i_ampa"]
    i_gaba = sim_result["i_gaba"]
    t_ms = sim_result["t_ms"]
    dt_ms = t_ms[1] - t_ms[0] if len(t_ms) > 1 else 0.1

    shift = int(round(delay_ms / dt_ms))
    if shift > 0:
        i_gaba_delayed = np.zeros_like(i_gaba)
        i_gaba_delayed[:, shift:] = i_gaba[:, :-shift]
    else:
        i_gaba_delayed = i_gaba

    proxy_current = -(i_ampa + gamma * i_gaba_delayed)  # (n_record, n_steps)

    dist = probe_distances(channel_positions_um, sim_result["positions_um"])
    weights = 1.0 / dist
    weights = weights / weights.shape[1]

    lfp_core = weights @ proxy_current
    lfp_uv = lfp_core * gain_uv_per_na

    n_channels, n_steps = lfp_uv.shape
    lfp_uv = lfp_uv + rng.normal(0, noise_uv_std, size=(n_channels, n_steps))
    lfp_uv = lfp_uv + pink_noise_uv_std * _pink_noise(n_channels, n_steps, rng)

    return t_ms, lfp_uv


def welch_psd(x, fs_hz, nperseg=None):
    from scipy.signal import welch
    x = np.asarray(x)
    n = x.shape[-1]
    if nperseg is None:
        nperseg = min(n, 2048)
    freqs_hz, psd = welch(x, fs=fs_hz, nperseg=nperseg, axis=-1)
    return freqs_hz, psd


def spectral_slope(freqs_hz, psd, fmin_hz=2.0, fmax_hz=100.0):
    mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz) & (freqs_hz > 0)
    log_f = np.log10(freqs_hz[mask])
    psd = np.atleast_2d(psd)
    alphas = []
    for row in psd:
        log_p = np.log10(np.maximum(row[mask], 1e-20))
        slope, _ = np.polyfit(log_f, log_p, 1)
        alphas.append(-slope)
    return float(np.mean(alphas))


STANDARD_BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 100),
}


def _trapz(y, x, axis=-1):
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return trapz_fn(y, x, axis=axis)


def band_power(freqs_hz, psd, band):
    fmin, fmax = band
    mask = (freqs_hz >= fmin) & (freqs_hz <= fmax)
    psd = np.atleast_2d(psd)
    powers = _trapz(psd[:, mask], freqs_hz[mask], axis=1)
    return powers if powers.shape[0] > 1 else float(powers[0])


def _normalized_band_powers(freqs_hz, psd_mean):
    total = band_power(freqs_hz, psd_mean, (1, 100))
    total = total if np.isscalar(total) else float(total)
    out = {}
    for name, band in STANDARD_BANDS.items():
        p = band_power(freqs_hz, psd_mean, band)
        p = p if np.isscalar(p) else float(p)
        out[name] = p / total if total > 0 else 0.0
    return out


def compare_to_reference(sim_lfp, sim_fs_hz, ref_lfp, ref_fs_hz):
    """Compare simulated vs. real LFP: 1/f exponent, band power, RMS amplitude."""
    sim_lfp = np.atleast_2d(sim_lfp)
    ref_lfp = np.atleast_2d(ref_lfp)

    sim_f, sim_psd = welch_psd(sim_lfp, sim_fs_hz)
    ref_f, ref_psd = welch_psd(ref_lfp, ref_fs_hz)
    sim_psd_mean = sim_psd.mean(axis=0)
    ref_psd_mean = ref_psd.mean(axis=0)

    sim_alpha = spectral_slope(sim_f, sim_psd_mean)
    ref_alpha = spectral_slope(ref_f, ref_psd_mean)

    sim_bands = _normalized_band_powers(sim_f, sim_psd_mean)
    ref_bands = _normalized_band_powers(ref_f, ref_psd_mean)

    diffs = np.array([sim_bands[b] - ref_bands[b] for b in STANDARD_BANDS])
    band_power_rmse = float(np.sqrt(np.mean(diffs ** 2)))

    sim_rms = float(np.sqrt(np.mean(sim_lfp ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_lfp ** 2)))

    return {
        "sim_alpha": sim_alpha, "ref_alpha": ref_alpha,
        "alpha_diff": abs(sim_alpha - ref_alpha),
        "sim_band_power": sim_bands, "ref_band_power": ref_bands,
        "band_power_rmse": band_power_rmse,
        "sim_rms_uv": sim_rms, "ref_rms_uv": ref_rms,
        "amplitude_ratio": sim_rms / ref_rms if ref_rms > 0 else float("nan"),
    }


def report_summary(report):
    lines = [
        f"1/f exponent   : simulated={report['sim_alpha']:.2f}  reference={report['ref_alpha']:.2f}  "
        f"(|diff|={report['alpha_diff']:.2f})",
        f"RMS amplitude  : simulated={report['sim_rms_uv']:.1f} uV  reference={report['ref_rms_uv']:.1f} uV  "
        f"(ratio={report['amplitude_ratio']:.2f})",
        f"Band power RMSE (normalized profile): {report['band_power_rmse']:.4f}",
        "Relative band power (fraction of 1-100 Hz total):",
    ]
    for band in STANDARD_BANDS:
        lines.append(
            f"  {band:6s}: simulated={report['sim_band_power'][band]:.3f}  "
            f"reference={report['ref_band_power'][band]:.3f}"
        )
    return "\n".join(lines)


# ============================================================================
# Streamlit UI
# ============================================================================

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

if run_button:
    with st.spinner("Simulating LIF network (numpy codegen; small nets run in seconds)..."):
        sim_result = run_network(
            n_exc=n_exc, n_inh=n_inh, duration_ms=float(duration_ms), seed=int(seed),
            probe_span_um=float(span_um), record_subset=min(1000, n_exc),
        )

    ch_pos = probe_positions(n_channels, float(span_um))
    t_ms, lfp_uv = compute_lfp(
        sim_result, ch_pos, gamma=gamma, delay_ms=delay_ms,
        noise_uv_std=noise_uv, pink_noise_uv_std=pink_uv, gain_uv_per_na=gain,
        seed=int(seed),
    )
    fs_hz = 1000.0 / (t_ms[1] - t_ms[0])

    st.session_state["sim_result"] = sim_result
    st.session_state["t_ms"] = t_ms
    st.session_state["lfp_uv"] = lfp_uv
    st.session_state["fs_hz"] = fs_hz
    st.session_state["ch_pos"] = ch_pos

if "lfp_uv" in st.session_state:
    sim_result = st.session_state["sim_result"]
    t_ms = st.session_state["t_ms"]
    lfp_uv = st.session_state["lfp_uv"]
    fs_hz = st.session_state["fs_hz"]
    ch_pos = st.session_state["ch_pos"]

    st.success(f"Mean network firing rate: {sim_result['firing_rate_hz']:.2f} Hz  |  "
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
        freqs, psd = welch_psd(lfp_uv, fs_hz)
        psd_mean = psd.mean(axis=0)
        alpha = spectral_slope(freqs, psd_mean)
        mask = (freqs > 0) & (freqs <= 200)
        psd_df = pd.DataFrame({"PSD (uV^2/Hz)": psd_mean[mask]}, index=freqs[mask])
        st.line_chart(np.log10(psd_df.clip(lower=1e-12)), height=400)
        st.caption(f"log10 PSD vs frequency (Hz). Fitted 1/f exponent: **{alpha:.2f}** "
                   f"(real cortical LFP is typically ~0.8-2 in the 2-100 Hz range).")

    st.subheader("Download")
    buf = io.BytesIO()
    np.savez(buf, t_ms=t_ms, lfp_uv=lfp_uv, fs_hz=fs_hz, channel_positions_um=ch_pos)
    st.download_button("Download simulated LFP (.npz)", buf.getvalue(), file_name="lfp_output.npz")

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
            if hasattr(loaded, "files"):
                ref_lfp = loaded["lfp_uv"] if "lfp_uv" in loaded.files else loaded[loaded.files[0]]
                ref_fs = float(loaded["fs_hz"]) if "fs_hz" in loaded.files else ref_fs_manual
            else:
                ref_lfp = loaded
                ref_fs = ref_fs_manual

            report = compare_to_reference(lfp_uv, fs_hz, ref_lfp, ref_fs)
            st.text(report_summary(report))

            band_df = pd.DataFrame({
                "simulated": report["sim_band_power"],
                "reference": report["ref_band_power"],
            })
            st.bar_chart(band_df)
else:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
