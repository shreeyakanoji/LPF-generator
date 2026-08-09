# lfpgen

**A biophysically-grounded LFP generator for testing amplifiers, decoders, and spike-sorting/LFP-analysis pipelines.**

`lfpgen` simulates a sparsely-connected excitatory/inhibitory leaky
integrate-and-fire (LIF) network in [Brian2](https://brian2.readthedocs.io/),
then converts the network's synaptic currents into a realistic multi-channel
local field potential (LFP) using the current-summation proxy validated
against full compartmental models by Mazzoni et al. (2015, *PLoS
Computational Biology*, "Computing the Local Field Potential from
Integrate-and-Fire Network Models"). The result is deployed across a virtual
linear electrode array (a simplified Neuropixels-shank geometry), giving you
a multi-channel test signal with realistic spectral content, amplitude
scale, and channel-to-channel structure — useful for benchmarking your own
amplifier front-ends, spike/LFP decoders, or analysis pipelines against
something more physiologically honest than filtered white noise.

A `validation` module lets you quantitatively compare the simulated LFP
against real reference recordings — including a convenience loader for the
**Allen Institute Neuropixels Visual Coding** dataset — using standard
summary statistics from the LFP-modeling literature (1/f spectral exponent,
per-band power fractions, RMS amplitude).

> **Why this approach, not a full compartmental model?** Multicompartment +
> volume-conductor modeling (e.g. LFPy) is the gold standard, but it defeats
> the point of using a lightweight LIF network. The AMPA/GABA
> current-summation proxy used here was specifically developed and validated
> to bridge that gap: it reproduces the LFP of a full compartmental
> simulation from I&F network currents to a good approximation, and is the
> approach used in a number of published network-LFP studies.

## Install

```bash
pip install lfpgen
```

Or, from this repository:

```bash
git clone https://github.com/yourusername/lfpgen.git
cd lfpgen
pip install -e .
```

Optional extras:

```bash
pip install lfpgen[allen]  # adds allensdk, for real-data validation examples
pip install lfpgen[dev]    # adds pytest, matplotlib, for running tests/examples
```

Requires Python >= 3.9. Core dependencies are `brian2`, `numpy`, `scipy`.

## Quickstart

```python
from lfpgen import NetworkParams, run_network, LinearProbe, LFPParams, compute_lfp

# 1. simulate an E/I spiking network
params = NetworkParams(n_exc=4000, n_inh=1000, duration_ms=2000.0, seed=42)
result = run_network(params)
print(f"Mean firing rate: {result.firing_rate_hz:.1f} Hz")

# 2. place a 32-channel virtual linear probe and compute the LFP
probe = LinearProbe(n_channels=32, span_um=1000.0)
t_ms, lfp_uv = compute_lfp(result, probe, LFPParams())

# lfp_uv is a (32, n_timepoints) array in microvolts, ready to feed into
# your amplifier model / decoder / analysis pipeline
```

Or from the command line:

```bash
lfpgen simulate --n-exc 4000 --n-inh 1000 --duration-ms 2000 --channels 32 --out lfp.npz
```

See `examples/quickstart.py` for a full example that also plots the traces
and power spectrum.

## Streamlit app

An interactive UI is included (`streamlit_app.py`): sliders for network size,
duration, probe geometry, and LFP proxy parameters; live trace + PSD plots;
upload-a-reference-recording validation panel.

```bash
pip install -e .[app]      # adds streamlit, pandas
streamlit run streamlit_app.py
```

To deploy on **Streamlit Community Cloud**: push this repo to GitHub, then
point a new app at `streamlit_app.py` (the included `requirements.txt`
covers dependencies). Streamlit Cloud has no C/C++ compiler, so the app sets
`brian2.prefs.codegen.target = "numpy"` at startup — a pure-Python backend
that's slower than Brian2's default but works in any environment. This makes
demo-sized networks (a few thousand neurons, a couple seconds of simulated
time) run in seconds to tens of seconds; for large networks, run locally
where Brian2 can use its faster compiled backend instead.

## Validating against real data

```python
import numpy as np
from lfpgen import validation

# against any real reference trace you already have as a numpy array:
ref_lfp = np.load("my_real_recording.npy")   # (n_channels, n_samples), microvolts
report = validation.compare_to_reference(lfp_uv, sim_fs_hz=1000.0,
                                          ref_lfp=ref_lfp, ref_fs_hz=2500.0)
print(report.summary())
```

Or pull a real snippet directly from the Allen Institute Neuropixels dataset
(`pip install lfpgen[allen]` required, plus internet access — AllenSDK
downloads and caches an NWB file on first use):

```python
ref_lfp, ref_fs = validation.load_allen_lfp(session_id=<your_session_id>, duration_s=10.0)
report = validation.compare_to_reference(lfp_uv, sim_fs_hz, ref_lfp, ref_fs)
```

`ValidationReport` gives you:
- fitted 1/f spectral exponent (simulated vs. reference)
- normalized power in delta/theta/alpha/beta/gamma bands
- RMS amplitude and amplitude ratio

If your simulated amplitudes are off from a specific reference recording,
use `validation.fit_gain(...)` to recalibrate `LFPParams.gain_uv_per_na`.

See `examples/validate_against_allen.py` for a complete script.

## How the LFP is computed

1. `network.run_network` builds an 80%-excitatory / 20%-inhibitory
   conductance-based LIF network (Brunel-style random connectivity, AMPA/GABA
   synapses, external Poisson drive) and records each recorded neuron's AMPA
   and GABA synaptic currents.
2. `lfp.compute_lfp` applies the Mazzoni et al. (2015) proxy —
   `-(I_AMPA + gamma * I_GABA(delayed))` per neuron — then sums each
   neuron's contribution across a linear electrode array weighted by inverse
   distance (point-current-source approximation), and adds realistic
   measurement + background (1/f) noise.
3. `validation` lets you check the result's spectral shape and amplitude
   against real recordings.

All physical parameters (synaptic time constants, conductances, proxy gamma
and delay, probe geometry, noise levels) are exposed as dataclass fields
(`NetworkParams`, `LFPParams`, `LinearProbe`) so you can tune the signal for
your own use case (e.g. mimic a specific cortical layer, or push the network
into a more synchronous/bursty regime).

## Running the tests

```bash
pip install -e .[dev]
pytest
```

Tests that require `brian2` are skipped automatically if it isn't installed,
so `tests/test_utils.py`, `test_probe.py`, `test_lfp.py`, and
`test_validation.py` (all pure-numpy) will still run and validate the signal
processing / forward-model math on their own.

## Project layout

```
lfpgen/
    network.py     - Brian2 LIF E/I network
    probe.py        - linear electrode array geometry
    lfp.py          - synaptic-current -> LFP forward model
    utils.py        - PSD / spectral-slope / band-power helpers
    validation.py   - comparison against real recordings (incl. Allen Institute loader)
    cli.py          - `lfpgen simulate` / `lfpgen validate` commands
tests/               - unit tests (numpy-only tests run without brian2)
examples/            - quickstart + Allen-validation scripts
```

## Citing

If you use the LFP proxy for research, please cite:

Mazzoni A, Lindén H, Cutrone A, Donoghue JP, Mattia M, Panzeri S (2015).
"Computing the Local Field Potential (LFP) from Integrate-and-Fire Network
Models." *PLoS Computational Biology* 11(12): e1004584.

## License

MIT — see `LICENSE`.
