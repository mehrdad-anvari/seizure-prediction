# Analysis example (API)

This example shows how to run the analysis pipeline programmatically.

```python
import os
from glob import glob

from seizure_pred.analysis import analyze_run

# Find the newest split folder under runs/<run_name>/<stamp>/split_*/
candidates = sorted(glob("runs/*/*/split_*"))
if not candidates:
    raise RuntimeError("No runs found under ./runs. Train something first.")

run_dir = candidates[-1]
report = analyze_run(run_dir, make_plots=True)

print("run_dir:", run_dir)
print("report.json:", os.path.join(report["out_dir"], "report.json"))
print("metrics:", report.get("report", {}))
```
