# Final COMPASS rotation experiments

Three validation winners × seeds 1/2/3, 100 epochs on the exact train+validation
union. Initialization reuses the train-only seed-1 X-Fi backbones from tuning.
Evaluate the final checkpoint under all seven modality subsets after training.
Training curves are saved every epoch; no validation selection during retraining.

`campaign/manifest.json` records all 75 selection results, the nine tasks, and
source/backbone hashes. `membership-check.json` verifies the live split membership;
`submission.json` records the submitted array. Data, weights and run outputs stay
outside Git. The sibling `wireless-sensing` and `x-fi` repositories and the recorded
backbone artifacts are required; their source/artifact hashes are in the manifest.

`train-validation.patch` adds final retraining to an isolated copy of the tuning
trainer. The queued subject search keeps its original code and source hashes.
`launch.py --prepare` checks the tuning results and snapshots the executable sources.

From the COMPASS repository, with the recorded artifacts available:

```bash
.venv/bin/python final_training/launch.py --prepare
.venv/bin/python final_training/launch.py --task 0 --inspect
.venv/bin/python final_training/launch.py --submit
```

Preparation requires a new output directory; submission rejects duplicate launches.
The current campaign is already submitted as array 272178. Use `requirements.txt`
and `uv.lock` for the environment; cluster paths are recorded in the manifest.

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q final_training/test_final.py
```
