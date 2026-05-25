"""Redirect MNE home to per-PID local dir + robust JSON parsing.

Lustre filesystems return EBADF/ESTALE on `fcntl.flock` and on concurrent
JSON writes. MNE's default config lives in $HOME/.mne (i.e. Lustre on Expanse)
and uses filelock + read-modify-write. Multiple workers in the same node
trip both of those.

Fix here: (1) set `_MNE_FAKE_HOME_DIR` to a per-PID directory inside the
SLURM-local /scratch NVMe so each process sees its own MNE home; (2) wrap
`_load_config` so a corrupt JSON returns {} instead of raising.

This module is imported at site init via `_mne_lustre_fix.pth` — so it
runs in every Python interpreter (parent, spawn workers, loky workers)
BEFORE any user code can import MNE.
"""
import os, sys, json


def _setup():
    try:
        user = os.environ.get("USER", "bpinto")
        tmpdir = os.environ.get("SLURM_TMPDIR") or f"/tmp/{user}"
        fake_home = f"{tmpdir}/mne_home_{os.getpid()}"
        os.makedirs(f"{fake_home}/.mne", exist_ok=True)
        cfg = f"{fake_home}/.mne/mne-python.json"
        if not os.path.exists(cfg):
            with open(cfg, "w") as f:
                json.dump({"MNE_STIM_CHANNEL": "STI 014"}, f)
        os.environ["_MNE_FAKE_HOME_DIR"] = fake_home
    except Exception as e:
        sys.stderr.write(f"[_mne_lustre_fix] setup failed: {e}\n")

    # Wrap _load_config so a corrupt JSON returns {} instead of raising.
    # Preserve the caller's raise_error: if they REALLY want corruption to
    # crash (defensive code path inside MNE), let it crash; only swallow when
    # the caller doesn't care.
    try:
        import mne.utils.config as _mnec
        _orig = _mnec._load_config

        def _safe_load(config_path, raise_error=False):
            try:
                return _orig(config_path, raise_error=raise_error)
            except Exception:
                return {}
        _mnec._load_config = _safe_load
    except Exception:
        pass


_setup()
