import py_compile
from pathlib import Path


CORE_SCRIPTS = (
    "cfm_mesh_train.py",
    "mesh_backbone.py",
    "mode_dispatch.py",
    "icosahedral_mesh.py",
    "exceedance_eval.py",
    "forecasts_of_opportunity.py",
    "recover_chunk_init_dates.py",
    "build_driver_tables.py",
    "ens_common.py",
    "download_ecmwf_s2s.py",
    "ens_ingest.py",
    "ens_score.py",
    "ens_compare.py",
    "ens_heatcast_stack_opportunity.py",
    "stitch_exceedance_folds.py",
    "build_paper_evidence_blocks.py",
    "build_paper_figures_tables.py",
    "build_paper_figures_extended.py",
    "export_w34_stack_netcdf.py",
    "figure_style.py",
    "publication_analysis_utils.py",
    "repo_integrity.py",
)


def test_core_scripts_compile():
    repo_root = Path(__file__).resolve().parents[1]
    for script in CORE_SCRIPTS:
        py_compile.compile(str(repo_root / script), doraise=True)
