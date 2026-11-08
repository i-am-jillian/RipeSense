"""Run train_banana_model_colab.ipynb on this laptop (no Jupyter needed).

  cd ~/UB_Hack/ripesense/training && ../.venv/bin/python ../notebooks/run_notebook_local.py
Outputs go to training/ripesense_train/output; put own_data.zip in training/ to include robot photos.
"""
import json, os, sys, time
import matplotlib
matplotlib.use("Agg")
nb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_banana_model_colab.ipynb")))
cells = [c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}
for i, src in enumerate(cells, 1):
    t = time.time()
    print(f"\n===== cell {i}/{len(cells)} =====", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), g)
    print(f"===== cell {i} done in {time.time() - t:.0f}s =====", flush=True)
