"""Reuse sklearn's bundled OpenMP inside this venv if LightGBM cannot load it.

Run after pip installation on macOS. Modifies only the installed LightGBM
binary in this environment. No Homebrew or system-wide installation needed.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

if sys.platform != "darwin":
    sys.exit(0)
try:
    import lightgbm  # noqa: F401
except OSError as exc:
    if "libomp" not in str(exc):
        raise
    lgb = Path(importlib.util.find_spec("lightgbm").origin).parent / "lib/lib_lightgbm.dylib"
    sklearn = Path(importlib.util.find_spec("sklearn").origin).parent / ".dylibs/libomp.dylib"
    if not sklearn.is_file():
        raise RuntimeError(
            "No bundled OpenMP found; install libomp with your package manager"
        ) from exc
    subprocess.run(["install_name_tool", "-add_rpath", str(sklearn.parent), str(lgb)], check=True)
    subprocess.run([sys.executable, "-c", "import lightgbm"], check=True)
    print("Configured LightGBM to use this environment's sklearn OpenMP library.")
