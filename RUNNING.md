# How to Run This Project

Quick reference for every entry point. Run all commands from the project
root (the folder containing `config.py`).

---

## 0. One-time setup (local PC)

```powershell
pip install -r requirements.txt      # pinned versions (Python 3.10+)
python download_dataset.py           # fetches NSL-KDD + UNSW-NB15 into data/raw/
```

CICIDS2017 is not auto-downloaded — its 8 `*pcap_ISCX.csv` files must be in
`data/raw/` (download from https://www.unb.ca/cic/datasets/ids-2017.html).
They are already present on this machine.

---

## 1. The main experiment (what produces the paper's numbers)

Full multi-seed suite — all 5 seeds x 14 methods, mean +/- std +
significance tests. This is the long one (~35-45 h for both datasets):

```powershell
python -u run_real_dataset.py --multiseed --dataset nsl-kdd
python -u run_real_dataset.py --multiseed --dataset cicids2017
```

Useful variations:

```powershell
# Specific seeds only (e.g. a quick 2-seed check)
python -u run_real_dataset.py --multiseed --dataset nsl-kdd --seeds 42 123

# Single algorithm, single seed (quick experiment)
python -u run_real_dataset.py --algorithm dqn --episodes 300 --dataset nsl-kdd

# Single algorithm with federated learning
python -u run_real_dataset.py --algorithm dqn --episodes 300 --federated

# Third dataset (downloaded but not yet part of the paper)
python -u run_real_dataset.py --multiseed --dataset unsw-nb15
```

Interrupted runs are safe: every method/seed saves a checkpoint
(`results/checkpoint_real_<dataset>_seed<N>.json`), so re-running the same
command resumes where it stopped, skipping completed work.

**Outputs:**
| What | Where |
|---|---|
| Aggregated mean +/- std results | `results/real_dataset_<dataset>_multiseed_results.json` |
| Pairwise significance tests | `results/real_dataset_<dataset>_significance.json` |
| Comparison table (CSV) | `results/tables/real_dataset_<dataset>_comparison_multiseed.csv` |
| Figures | `results/figures/real_dataset/` |
| Training logs (if redirected) | `results/logs/` |

---

## 1b. Reward ablation: Stackelberg tau vs. generic energy penalty

Isolates whether the game-theoretic deterrence threshold in the reward
actually matters. Trains the same agent under three reward modes
(`stackelberg` / `constant` / `none`), multi-seed, checkpointed/resumable:

```powershell
python -u run_ablation_tau.py --dataset cicids2017            # all config.SEEDS
python -u run_ablation_tau.py --dataset nsl-kdd --seeds 42 123
python -u run_ablation_tau.py --dataset cicids2017 --algorithm dqn --episodes 300
```

**Outputs:** `results/ablation_tau_<dataset>_results.json` (means, stds,
paired t-tests), `results/tables/ablation_tau_<dataset>.csv`, checkpoint at
`results/checkpoint_ablation_tau_<dataset>.json`.

---

## 2. Run on the HPC cluster instead (survives your PC turning off)

Everything is prepared — see **`hpc/README.md`** for the full guide. Short
version:

```powershell
# from your PC: upload the prepared bundle (on your Desktop)
scp $HOME\Desktop\prigmarl_hpc_bundle.tar.gz <user>@<cluster>:~/
```

```bash
# on the cluster:
tar xzf prigmarl_hpc_bundle.tar.gz && cd prigmarl
bash hpc/setup_env.sh        # one-time env setup + self-checks
sbatch hpc/job.slurm         # SLURM clusters
bash hpc/run_all.sh          # or: direct/interactive node (nohup-detached)
tail -f results/logs/multiseed_nsl-kdd_v2.log     # monitor
```

---

## 3. Synthetic-environment experiments (secondary pipeline)

```powershell
# Single algorithm on synthetic traffic
python main.py --algorithm dqn --episodes 500 --network_size 20

# Batch comparisons / ablations
python run_experiments.py --experiment compare
python run_experiments.py --experiment ablation

# Full publication suite (synthetic): 4 MARL + baselines + ablations
python run_publication.py
python run_publication.py --quick     # reduced episodes for a fast pass
```

Note: the synthetic pipeline's ablation figures predate the corrected
reward/detection model (flagged as illustrative in the paper) — the
real-dataset pipeline above is the source of truth for reported results.

---

## 4. Tests

```powershell
python -m unittest discover -s tests -v     # 22 assertion-based unit tests
```

---

## 5. Build the paper PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex   # -> paper/paper.pdf
```

(Same recipe works for `thesis/main.tex`.)

---

## 6. Monitoring a long local run

```powershell
# live progress of a background run
Get-Content "results\logs\multiseed_nsl-kdd_v2.log" -Wait -Tail 20

# is the python process alive and busy?
Get-Process python | Select-Object Id, CPU, WorkingSet
```
