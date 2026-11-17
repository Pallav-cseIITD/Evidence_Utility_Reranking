# Software environments

Use separate environments for CPU analysis and GPU reader evaluation. Run the commands below from the repository root in Bash on Linux or WSL.

| File | Purpose |
| :--- | :--- |
| `analysis.txt` | Selector training and inference on prepared features, and analysis of saved responses |
| `gpu.txt` | The A100 reader comparisons using 16 bit model weights and ALCE citation evaluation |
| `recorded_environment.json` | Versions and hardware recorded for the completed final HotpotQA additional reader run |

## CPU environment

Use Python 3.11 or 3.12. The reorganized CPU checks use NumPy 2.3.5. CPU passage selection uses tokenizers 0.21.4, matching the original HotpotQA preparation environment; the later GPU reader setup specifies tokenizers 0.21.1 separately.

```bash
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate
python3 -m pip install -r requirements/analysis.txt
python3 -m pip check
```

After the method and tests folders are available:

```bash
python3 -m unittest discover -s method/tests -v
```

CPU analysis uses prepared passage pools, features, labels or saved reader responses. It does not rebuild the retrieval index or generate new answers. The data preparation and experiment instructions specify which assets each command needs.

## GPU environment

Use a separate Linux environment with Python 3.11. The completed A100 run used Python 3.11.18, an A100 with 40 GB memory and the CUDA 11.8 build of PyTorch. CUDA support also requires a compatible NVIDIA driver on the host.

```bash
python3 -m venv .venv-reader
source .venv-reader/bin/activate
python3 -m pip install -r requirements/gpu.txt
python3 -m pip check
python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
```

The final value should be `True` on the allocated GPU node. On a cluster, run generation only after receiving the GPU allocation. Model downloads and data preparation are separate steps; the reader entry points will verify the specified model revisions and inputs.

This environment covers the later A100 readers. The primary smaller reader used a separate reduced memory loading configuration, which will be documented alongside that historical run. Do not substitute one loading format for the other when reproducing a particular result.

## Version records

The GPU requirements preserve the original reader setup requirements and pin NumPy and PyTorch to the versions observed in the final run. Packages that were originally specified as ranges remain ranges. `recorded_environment.json` contains the measured subset of package versions, rather than a complete package freeze. A fresh installation can therefore resolve different versions of the remaining dependencies.

The original run records will accompany the saved results. A full installation and reproduction check of the reorganized pipeline will be performed once its entry points and data manifests are included. Preparing these requirements does not constitute a new GPU run.

Retrieval index construction has additional dependencies and storage needs. Those will be supplied with the retrieval code rather than included in the lighter analysis environment.
