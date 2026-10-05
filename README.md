# Quantum Temporal Convolutional Network (QTCN)

This repository contains the implementation of [the Quantum Temporal Convolutional Network (QTCN)](https://link.springer.com/chapter/10.1007/978-3-032-32335-4_20) using PennyLane and PyTorch.
The quantum model implements the temporal convolutional component of TCNs with quantum dilated convolutional neural network (QDCNN).  

## Benchmarking Datasets

The benchmarking datasets used in this work are selected to represent different levels of data complexity : 

  - **The Adding Problem** with various T (we evaluated on T=200, 400, 600)
  - **Sequential MNIST** digit classification
  - **JSB Chorales** polyphonic music
  - **Nottingham** polyphonic music


## Usage

The repository respects the **same file organization as the original TCN [repository](https://github.com/locuslab/TCN)** to make exploration and comparison easier :

```
[TASK_NAME] /
    data/
    experiments.py
    utils.py
QTCN.py
```

## Setup

PennyLane 0.37 does not support Python ≥ 3.13, so use a Python 3.11 virtual environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Running the time-series prediction (Poly_Music)

`QTCN/Poly_Music/experiments.py` trains the QTCN to predict the next time step of polyphonic music sequences (Nottingham and JSB Chorales, 32-step input windows reduced to 2 features):

```bash
cd QTCN/Poly_Music
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg ../../.venv/bin/python -u experiments.py 2>&1 | tee train.log
```

- The script must be run from `QTCN/Poly_Music`, since the data is loaded from `./mdata/`.
- `CUDA_VISIBLE_DEVICES=""` forces CPU. The quantum circuits are simulated on the CPU anyway (via NumPy), so a GPU only adds host/device copies.
- `MPLBACKEND=Agg` keeps `plt.show()` from blocking between datasets; the plots are still saved as PNG files.
- Outputs are written to the working directory: `best_qtcn_<dataset>.pth`, `training_history_<dataset>.npz`, `test_losses_<dataset>.txt`, `qtcn_training_<dataset>.png`.

### Runtime

The quantum layers run one circuit per sample and per time step. A batch of 32 windows takes about 7.5 s on a desktop CPU:

| Dataset | Train windows | Approx. time per epoch |
|---|---|---|
| JSB Chorales | 6,486 | ~45 min (incl. evaluation) |
| Nottingham | 154,353 | ~10 h |

The default configuration (100 epochs on both datasets) would therefore take weeks. For a first test, set `datasets = ['JSB']` and `num_epochs=3` in the `__main__` block of `experiments.py`.

## Known issues

### Fixed: `calculate_metrics` crash

`run_experiment` calls `calculate_metrics(model, loader, device, criterion)`, but the function only accepted three arguments, so training crashed with a `TypeError` after the first epoch. The signature now accepts an (unused) `criterion=None` argument.

### The quantum weights are not trained

In `QuantumDilatedConvolution.forward` (`QTCN/Poly_Music/QTCN.py`), the circuit is run outside the autograd graph:

```python
weights_np = self.weights.detach().cpu().numpy()       # weights detached
input_data = x_slice[i, ...].detach().cpu().numpy()    # inputs detached
output[i, :, j] = torch.tensor(circuit_output, ...)    # result is a constant
```

Consequences (verified by running a backward pass):

- All `qdcnn*.weights` parameters receive **no gradient**. Adam skips them, so they stay at their random initialization (`0.01 * torch.randn(...)`) for the whole run.
- No gradient flows back *through* the quantum path either. The classical layers before each quantum layer are trained only via the 1×1 `Conv1d` skip connections of the residual blocks. In effect, the quantum layers act as a fixed, random feature transform, and the trainable part of the model is purely classical.

### The quantum weights have no effect on the output

With `hidden_dim=6`, each quantum layer has a single weight (`n_qubits // 4 = 1`). Because of the `weight_idx < len(weights)` condition, the circuit applies only one CNOT followed by `RZ(w)` on the target qubit. That `RZ` is the last gate on its wire, and the circuit then measures ⟨Z⟩. `RZ` is diagonal in the Z basis, so it does not change ⟨Z⟩. Evaluating the circuit with w = 0.0, 0.7 and 2.0 gives identical outputs.

Even with gradient flow restored, these weights would have nothing to learn.

### Poly_Music: inputs and targets refer to different notes

The poster describes JSB Chorales and Nottingham as predicting the next 88-bit piano vector. `PolyphonicMusicDataset` in `QTCN/Poly_Music/experiments.py` does something else:

- **Only 2 of 88 notes are used.** Each input window is reduced to the 2 notes with the highest variance *within that window*, so the selected notes differ from window to window.
- **The target notes are effectively fixed and unrelated to the inputs.** The target is a single time step, and `_reduce_dimensions` computes its variance over one row, which gives NaN (this is the source of the `var(): degrees of freedom is <= 0` warning). `topk` over the NaNs then always selects notes **58 and 60**. In a sample of 128 JSB test windows, the input notes matched the target notes in only 4 cases. The model is therefore asked to predict notes 58/60 from two other, changing notes.

### Poly_Music: losses are not comparable to the classical baselines

The classical JSB/Nottingham losses on the poster (e.g. TCN 8.10 / 3.07) follow the convention of the original TCN repository: negative log-likelihood summed over all 88 notes per time step. The QTCN numbers (0.0026 / 3.9e-5) come from `calculate_metrics`, which computes a loss over the 2 selected notes and divides by the number of outputs. The difference of several orders of magnitude reflects the different task and loss definition. It is not evidence of a quantum advantage.

### Adding Problem and MNIST do not run as-is

`QTCN/adding_problem/experiments.py` and `QTCN/MNIST/experiments.py` do `import QTCN` and then call `QTCN(...)`. However, those directories contain no `QTCN.py` (only `QTCN/Poly_Music/` does), and the call would have to be `QTCN.QTCN(...)`. Both scripts fail before training starts.

### Possible fixes (not applied)

- Call the QNode with torch tensors instead of `.detach().numpy()`, ideally batched via `qml.qnn.TorchLayer`, so that gradients flow.
- Use trainable gates that affect the measured observable, e.g. `RY`/`RX` instead of (or in addition to) `RZ`, or rotations before the entangling layer.
- Poly_Music: select the same note subset for inputs and targets (e.g. fixed per dataset), or use all 88 notes. Report the NLL summed over 88 notes per time step, as the classical TCN does.
- Adding Problem / MNIST: import the model from `QTCN/Poly_Music/QTCN.py` and call `QTCN.QTCN(...)`.

Either change alters the model relative to the published code, so the results would no longer be directly comparable to the paper.

To cite this work:

```
@inproceedings{hoceini2026quantum,
  title={Quantum Temporal Convolution Network},
  author={Hoceini, Rihab and Bouida, Ahmed},
  booktitle={German Conference on Artificial Intelligence (K{\"u}nstliche Intelligenz)},
  pages={244--251},
  year={2026},
  organization={Springer}
}
```
