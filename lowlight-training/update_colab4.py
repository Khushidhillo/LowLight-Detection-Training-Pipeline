import json
from pathlib import Path

p = Path("notebooks/Colab_Training.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch cell-004 to apply inline fix before running pytest
for cell in nb["cells"]:
    if cell.get("id") == "cell-004":
        cell["source"] = [
            "import os, sys\n",
            "from pathlib import Path\n",
            "if 'target_dir' in globals() and target_dir and os.path.exists(target_dir):\n",
            "    os.chdir(target_dir)\n",
            "    if target_dir not in sys.path: sys.path.insert(0, target_dir)\n",
            "\n",
            "# Apply autograd loss patch for newer PyTorch/Ultralytics in Colab\n",
            "test_path = os.path.join(target_dir if 'target_dir' in globals() and target_dir else '/content/lowlight-training', 'tests/test_pipeline.py')\n",
            "if os.path.exists(test_path):\n",
            "    code = open(test_path).read()\n",
            "    code = code.replace('loss,_=v8DetectionLoss(detector)(detector(y),labels)\\n    loss.backward()', 'loss,_=v8DetectionLoss(detector)(detector(y),labels)\\n    if getattr(loss, \"ndim\", 0) > 0: loss = loss.sum()\\n    loss.backward()')\n",
            "    open(test_path, 'w').write(code)\n",
            "\n",
            "BASE = Path(\"/content/drive/MyDrive/LowLightResearch\")\n",
            "BASE.mkdir(parents=True, exist_ok=True)\n",
            "RAW = BASE / \"raw\"\n",
            "DATA = BASE / \"exdark\"\n",
            "RUNS = BASE / \"runs\"\n",
            "!python -m pytest -q"
        ]

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("COLAB NOTEBOOK PATCHED WITH AUTO TEST FIX")
