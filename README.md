# ETHQHack 2026

Submission repository for the ETHQHack 2026 quantum hackathon challenge, run on IQM Crystal-class quantum processors via [IQM Resonance](https://resonance.meetiqm.com/).

## Repository layout

- [challenge_tutorial/](challenge_tutorial/) — organizer-provided tutorial notebooks: connecting to IQM Resonance, routing to specific qubits, and the challenge brief.
- [mermin/](mermin/) — one of the pottential algorithm analysis: a generalized Mermin inequality test for arbitrary `n` qubits on IQM hardware, with hardware-aware chain selection and graceful simulator fallback.

## The challenge

Run a multi-qubit entanglement-witness experiment on a real IQM device and demonstrate a quantitative violation of a classical bound. One of the experiments was the **generalized Mermin inequality** (De Fabritiis, Roditi, Sorella, [arXiv:2305.04546](https://arxiv.org/abs/2305.04546)) because it scales cleanly to any `n`, has a clean analytic GHZ benchmark (`2^((n+1)/2)`), and stress-tests both connectivity selection and gate fidelity on the device.

## Quick start

1. Walk through the tutorial notebooks in [challenge_tutorial/](challenge_tutorial/) to set up IQM Resonance credentials.
2. Try the simulator path with no credentials:
   ```python
   from mermin_iqm import run_mermin_test
   result = run_mermin_test(n=3, shots=8192)
   print(result.summary())
   ```
3. See [mermin/README.md](mermin/README.md) for the full API, the math behind the protocol, hardware-failure handling, and the simulator sweep results for `n = 2 … 8`.

## Dependencies

- qiskit ≥ 1.0, qiskit-aer
- qiskit-iqm (only for real-hardware runs)
- numpy, scipy, matplotlib

Credentials for IQM Resonance are loaded from `mermin/.env` (gitignored).
