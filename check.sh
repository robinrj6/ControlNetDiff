python - <<'PY'
from datasets import load_dataset
d="/home/hpc/rlvl/rlvl165v/Desktop/diffusers/examples/controlnet/shared/datasets/fill50k/extracted"
ds = load_dataset("imagefolder", data_dir=d, split="train")
print("columns:", ds.column_names)
print("types:", {k: type(ds[0][k]).__name__ for k in ds.column_names})
PY
