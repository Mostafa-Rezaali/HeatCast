import py_compile
from pathlib import Path


CORE_SCRIPTS = (
    "cfm_mesh_train.py",
    "mesh_backbone.py",
    "mode_dispatch.py",
    "icosahedral_mesh.py",
    "stitch_hindcast_tac.py",
    "tac_skill_maps.py",
    "summarize_hindcast_diagnostics.py",
    "compute_baselines.py",
    "export_per_year_stats.py",
    "bootstrap_significance.py",
)


def test_core_scripts_compile():
    repo_root = Path(__file__).resolve().parents[1]
    for script in CORE_SCRIPTS:
        py_compile.compile(str(repo_root / script), doraise=True)
