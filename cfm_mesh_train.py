#!/usr/bin/env python3
"""
================================================================================
MeshFlowNet - DIRECT 15-DAY PREDICTION (No Autoregressive Rollout)
================================================================================
Physics-Guided Heat Wave Forecasting with ICOSAHEDRAL MESH GNN BACKBONE
(GraphCast-style Encoder-Processor-Decoder) + GLOBAL CONTEXT ENCODER

Key changes from rollout version:
- LEAD_TIME=15: model directly predicts day-15 field in one forward pass
- ROLLOUT_STEPS=1: no autoregressive rollout at inference
- Leave-k-years-out cross-validation instead of forward temporal split
- Exponential extreme loss option for transfer learning fine-tuning
- Validation uses single forward pass (15x faster than rollout)

Architecture unchanged: MeshFlowNet GNN backbone with per-round FiLM.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import gc
import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from tqdm import tqdm
import os
import math
import argparse
import ast
import csv
import random
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import shutil
mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy("file_system")

from datetime import datetime, timedelta

from icosahedral_mesh import IcosahedralMesh
from mesh_backbone import MeshFlowNet, count_parameters
from mode_dispatch import compute_loss, generate_sample
import pickle
try:
    from publication_analysis_utils import region_masks as publication_region_masks
except Exception:
    publication_region_masks = None

# ======================================================================================
# Logger
# ======================================================================================
import logging

def setup_spike_logger(output_dir):
    logger = logging.getLogger("spike_warnings")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, "spike_warnings.log"), mode="a")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ======================================================================================
# COMPUTED VARIABLE HELPERS
# ======================================================================================
def compute_doy_array(time_values):
    """Convert 'days since 1981-05-01' offsets to day-of-year (1-366)."""
    base = datetime(1981, 5, 1)
    doys = np.empty(len(time_values), dtype=np.float32)
    for i, tv in enumerate(time_values):
        dt = base + timedelta(days=float(tv))
        doys[i] = dt.timetuple().tm_yday
    return doys


def compute_toa_insolation(lat_deg, doy):
    """
    Daily-mean TOA insolation (W/m^2) for given latitudes and day-of-year.
    lat_deg: 1D array of latitudes in degrees (H,)
    doy: scalar day-of-year
    Returns: (H,) array
    """
    S0 = 1361.0
    lat_rad = np.radians(lat_deg)
    decl = np.radians(23.4393 * np.sin(np.radians(360.0 / 365.0 * (doy + 284))))
    cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    omega_s = np.arccos(cos_omega)
    Q = (S0 / np.pi) * (
        omega_s * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.sin(omega_s)
    )
    return Q.astype(np.float32)


# ======================================================================================
# DDP Setup
# ======================================================================================
def setup_ddp(rank, world_size):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}: entering init_process_group...", flush=True)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
        timeout=timedelta(hours=5),
    )
    print(f"Rank {rank}: DDP initialized", flush=True)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _safe_name(value):
    text = str(value).strip().replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def apply_run_name(run_name):
    run_name = _safe_name(run_name) if run_name else ""
    Config.RUN_NAME = run_name
    if run_name:
        Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "checkpoints", run_name)
        Config.PLOTS_DIR = os.path.join(Config.OUTPUT_DIR, "test_prediction_plots", run_name)
        Config.MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, f"trained_cfm_direct15_{run_name}.pth")
        Config.TAC_MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, f"trained_cfm_direct15_{run_name}_best_tac.pth")
        Config.MONITOR_MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, f"trained_cfm_direct15_{run_name}_best_monitor.pth")
    else:
        Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "checkpoints")
        Config.PLOTS_DIR = os.path.join(Config.OUTPUT_DIR, "test_prediction_plots")
        Config.MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, "trained_cfm_direct15.pth")
        Config.TAC_MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, "trained_cfm_direct15_best_tac.pth")
        Config.MONITOR_MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, "trained_cfm_direct15_best_monitor.pth")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.HINDCAST_STATS_DIR, exist_ok=True)
    os.makedirs(Config.HINDCAST_PAPER_DIR, exist_ok=True)


def _format_offsets(offsets):
    return "-".join(str(int(x)) for x in offsets) if offsets else "none"


def parse_cv_offsets(value, stride):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        parts = value
    else:
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
    offsets = sorted({int(p) % int(stride) for p in parts})
    return tuple(offsets)


def cv_split_tag(config=None):
    if config is None:
        config = Config
    return (
        f"cv{config.CV_STRIDE}"
        f"_val{_format_offsets(config.CV_VAL_OFFSETS)}"
        f"_test{_format_offsets(config.CV_TEST_OFFSETS)}"
    )


def get_norm_stats_path(config=None):
    if config is None:
        config = Config
    suffix = "_tube" + "-".join(str(x) for x in prediction_leads(config)) if getattr(config, "MULTI_LEAD_TUBE", False) else ""
    return os.path.join(
        config.OUTPUT_DIR,
        "data_cache",
        f"norm_stats_direct15{suffix}_{cv_split_tag(config)}.npz",
    )


# ======================================================================================
# CFM COMPONENTS
# ======================================================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

class FlowMatching(nn.Module):
    def sample_xt(self, x_0, x_1, t):
        t = t.view(-1, 1, 1, 1)
        x_flow = (1 - t) * x_0 + t * x_1
        return x_flow

    def velocity_target(self, x_0, x_1):
        return x_1 - x_0

# ======================================================================================
# CONFIGURATION
# ======================================================================================
class Config:
    # ==================== DATA PATHS ====================
    TRAINING_DATA_PATH = '/blue/nessie/mostafarezaali/Teleconnection/VDM_Training_Data_Extended_v2.nc'
    TARGET_VARIABLE_CANDIDATES = ("t2m_prism", "HeatIndex")
    OUTPUT_DIR = os.environ.get(
        "HEATCAST_OUTPUT_DIR",
        "/blue/nessie/mostafarezaali/Teleconnection/",
    )
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "test_prediction_plots")

    OUTPUT_NC_FILE = os.path.join(OUTPUT_DIR, "CFM_Forecasts_Improved.nc")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_cfm_direct15.pth")
    TAC_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_cfm_direct15_best_tac.pth")
    MONITOR_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_cfm_direct15_best_monitor.pth")

    # ==================== GLOBAL DATA PATHS ====================
    GLOBAL_DATA_PATH = '/blue/nessie/mostafarezaali/Teleconnection/Global_Coarse_Conditions_Extended.nc'
    EXTENDED_GLOBAL_VARIABLES_PATH = os.path.join(
        OUTPUT_DIR, "data_cache", "extended_global_variables.txt"
    )
    USE_EXTENDED_GLOBAL_FIELDS = True
    REQUIRE_EXTENDED_GLOBAL_FIELDS = True
    EXPECTED_NUM_GLOBAL_CHANNELS = 59
    GLOBAL_VARIABLES = [
        'sst',
        'olr',
        'geopotential_200',
        'u_wind_200',
        'total_column_water_vapour',
        'v_wind_200',
        'geopotential_500',
        'temperature_850',
        'temperature_2m_global',
    ]
    GLOBAL_SIZE = (181, 360)
    GLOBAL_LAG_DAYS = (0,)
    GLOBAL_DECOMPOSE_LOW_RESIDUAL = True
    GLOBAL_LOWPASS_WINDOW_DAYS = 20
    GLOBAL_COMPONENT_NAMES = ("low20", "residual")
    NUM_GLOBAL_CHANNELS = (
        EXPECTED_NUM_GLOBAL_CHANNELS
        * len(GLOBAL_LAG_DAYS)
        * (len(GLOBAL_COMPONENT_NAMES) if GLOBAL_DECOMPOSE_LOW_RESIDUAL else 1)
    )

    # Train on the daily day-15 z-score field. Train-year-only local daily
    # climatology is still built for TAC verification, not used as the target.
    TRAIN_ON_CLIMATOLOGY_ANOMALIES = False
    CLIMATOLOGY_WINDOW_DAYS = 30
    PREDICT_PERSISTENCE_RESIDUAL = True

    # ==================== MODEL ARCHITECTURE ====================
    IMAGE_SIZE = (621, 1405)
    IMAGE_CHANNELS = 1
    # Additional slow-state local memory channels. These are normalized exactly
    # like their current-day counterparts and appended to spatial_c.
    LOCAL_LAG_DAYS = (7, 14)
    LOCAL_LAG_VARIABLES = ("t2max", "soil_moisture", "temperature_2m")
    NUM_LOCAL_LAG_CHANNELS = len(LOCAL_LAG_DAYS) * len(LOCAL_LAG_VARIABLES)
    # spatial_c = physics(9) + local_lags(6) + topo/lat/lon/doy/toa/mask(7) = 22
    # model input = [x_t(1), x_tm1(1), x_tm2(1), spatial_c(22)] = 25 (deterministic)
    NUM_SPATIAL_CONDITIONS = 3 + 9 + NUM_LOCAL_LAG_CHANNELS + 7
    LEAD_TIME = 15             # Direct 15-day prediction (no rollout)
    MULTI_LEAD_TUBE = False
    PREDICTION_LEADS = (12, 13, 14, 15, 16, 17, 18)
    TUBE_LOSS_DAILY_WEIGHT = 0.80
    TUBE_LOSS_CENTER_WEIGHT = 0.10
    TUBE_LOSS_WEEKLY_WEIGHT = 0.10
    TUBE_TEMPORAL_HEADS = 4
    TUBE_DECODE_CHUNK_SIZE = 0
    GRADIENT_LOSS_WEIGHT = 0.0
    DISTRIBUTIONAL_HEAD = False
    CRPS_LOSS = False
    SIGMA_FLOOR = 0.1
    MSE_ANCHOR_WEIGHT = 0.0
    ENABLE_EXCEEDANCE_HEAD = False
    EXCEEDANCE_BCE_WEIGHT = 0.0
    EXCEEDANCE_COUNT_WEIGHT = 0.0
    EXCEEDANCE_POS_WEIGHT = 10.0
    EXCEEDANCE_FOCAL_GAMMA = 0.0
    EXCEEDANCE_INITIAL_PROB = 0.05
    ROLLOUT_STEPS = 1          # Single forward pass at inference
    CONDITION_DIM = 5

    BASE_DIM = 64
    DIM_MULTS = (1, 2, 4, 8)
    DROPOUT_RATE = 0.15

    GLOBAL_ENCODER_DIM = 64

    BATCH_SIZE = 4
    OCEAN_FILL = 0

    # ==================== CFM SCHEDULE ====================
    CFM_SAMPLING_STEPS = 50

    # ==================== TRAINING HYPERPARAMETERS ====================
    DEVICE = "cuda"
    SEED = 42
    LEARNING_RATE = 5e-5
    GRAD_CLIP_NORM = 1.0
    WINDOW_SIZE = 64

    # ==================== GENERATION SETTINGS ====================
    ENSEMBLE_SIZE = 20
    ENSEMBLE_MODE = False

    # ==================== TRAINING SCHEDULE ====================
    MAX_EPOCHS = 30
    TEST_FRACTION = 0.15
    VAL_FRACTION = 0.15
    CHECKPOINT_FREQ = 1
    NUM_VALIDATION_SAMPLES = 100  # Direct prediction is fast, use more samples
    SSIM_VALIDATION_SAMPLES = 128
    SSIM_CHUNK_SIZE = 8
    WARMUP_EPOCHS = 10
    MAX_TRAIN_BATCHES = None
    USE_VALIDATION_PATIENCE = True
    EARLY_STOP_METRIC = "weekly7_tac"
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_EPOCH = 12
    EARLY_STOP_MIN_DELTA = 1e-4

    # ==================== CROSS-VALIDATED HINDCASTS ====================
    CV_STRIDE = 5
    CV_VAL_OFFSETS = (3,)
    CV_TEST_OFFSETS = (4,)
    RUN_NAME = ""
    HINDCAST_STATS_DIR = os.path.join(OUTPUT_DIR, "hindcast_stats")
    HINDCAST_PAPER_DIR = os.path.join(OUTPUT_DIR, "hindcast_paper_data")
    TRAINING_METRICS_DIR = os.path.join(OUTPUT_DIR, "training_metrics")
    SAVE_HINDCAST_PAPER_DATA = True
    PAPER_MAPS_PER_MONTH_PER_FOLD = 8
    PAPER_HEAT_MAPS_PER_FOLD = 12

    # ==================== MESH CONFIG ====================
    MESH_REFINEMENT_LEVEL = 7
    MESH_PROCESSOR_ROUNDS = 8
    MESH_LATENT_DIM = 128
    MESH_BUFFER_DEG = 5.0
    K_GRID2MESH = 3
    K_MESH2GRID = 8
    CONUS_LAT_RANGE = (25.0, 50.0)
    CONUS_LON_RANGE = (-130.0, -60.0)

    # ==================== MODE ====================
    DETERMINISTIC = True  # Default to deterministic for direct prediction

    # ==================== EXTREME LOSS FINE-TUNING ====================
    USE_EXTREME_LOSS = False       # Enable after initial MSE training
    EXTREME_LOSS_WEIGHT = 0.3      # Blend: (1-w)*MSE + w*extreme
    EXTREME_LOSS_HEAT_BIAS = 1.0   # a=1.0: upweight positive extremes (heat)
    EXTREME_LOSS_COLD_BIAS = 0.0   # b=0.0: no cold extreme emphasis

    # ==================== ANOMALY-CORRELATION LOSS ====================
    USE_ANOMALY_CORR_LOSS = False
    ANOMALY_CORR_LOSS_WEIGHT = 0.0


def global_base_channel_count(config=Config):
    return len(config.GLOBAL_VARIABLES)


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def global_effective_channel_count(config=Config):
    n_components = len(config.GLOBAL_COMPONENT_NAMES) if config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 1
    return global_base_channel_count(config) * len(config.GLOBAL_LAG_DAYS) * n_components


def required_input_history(config=Config):
    lowpass_extra = int(config.GLOBAL_LOWPASS_WINDOW_DAYS) - 1 if config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 0
    global_history = max(int(lag) + lowpass_extra for lag in config.GLOBAL_LAG_DAYS)
    local_lags = getattr(config, "LOCAL_LAG_DAYS", ())
    local_history = max((int(lag) for lag in local_lags), default=0)
    return max(2, global_history, local_history)


def prediction_leads(config=Config):
    if getattr(config, "MULTI_LEAD_TUBE", False):
        return tuple(int(x) for x in getattr(config, "PREDICTION_LEADS", (config.LEAD_TIME,)))
    return (int(config.LEAD_TIME),)


def center_lead_index(config=Config):
    leads = prediction_leads(config)
    center = int(config.LEAD_TIME)
    if center not in leads:
        raise ValueError(f"LEAD_TIME={center} must be present in PREDICTION_LEADS={leads}")
    return leads.index(center)


def max_prediction_lead(config=Config):
    return max(prediction_leads(config))


def tube_mean_display_label(config=Config):
    """Human-readable tube-mean label; metric keys remain backward compatible."""
    return f"{len(prediction_leads(config))}-day mean"


def build_meshflow_model(config, mesh, device):
    """Build the shared MeshFlowNet used by training, export, and test paths."""
    model = MeshFlowNet(
        img_channels=config.IMAGE_CHANNELS,
        spatial_cond_channels=config.NUM_SPATIAL_CONDITIONS,
        condition_dim=config.CONDITION_DIM,
        latent_dim=config.MESH_LATENT_DIM,
        hidden_dim=config.MESH_LATENT_DIM * 2,
        num_processor_rounds=config.MESH_PROCESSOR_ROUNDS,
        mesh=mesh,
        image_size=config.IMAGE_SIZE,
        num_global_channels=config.NUM_GLOBAL_CHANNELS,
        global_encoder_dim=config.GLOBAL_ENCODER_DIM,
        deterministic=config.DETERMINISTIC,
        dropout=config.DROPOUT_RATE,
        predict_persistence_residual=config.PREDICT_PERSISTENCE_RESIDUAL,
        multi_lead_tube=config.MULTI_LEAD_TUBE,
        prediction_leads=prediction_leads(config),
        tube_temporal_heads=config.TUBE_TEMPORAL_HEADS,
        tube_decode_chunk_size=config.TUBE_DECODE_CHUNK_SIZE,
        tube_loss_weights=(
            config.TUBE_LOSS_DAILY_WEIGHT,
            config.TUBE_LOSS_CENTER_WEIGHT,
            config.TUBE_LOSS_WEEKLY_WEIGHT,
        ),
        gradient_loss_weight=config.GRADIENT_LOSS_WEIGHT,
        enable_exceedance_head=config.ENABLE_EXCEEDANCE_HEAD,
        exceedance_initial_logit=math.log(
            float(config.EXCEEDANCE_INITIAL_PROB)
            / max(1.0 - float(config.EXCEEDANCE_INITIAL_PROB), 1e-6)
        ),
        distributional_head=config.DISTRIBUTIONAL_HEAD,
        sigma_floor=config.SIGMA_FLOOR,
    ).to(device)
    model.crps_loss = bool(config.CRPS_LOSS)
    model.mse_anchor_weight = float(config.MSE_ANCHOR_WEIGHT)
    model.exceedance_bce_weight = float(config.EXCEEDANCE_BCE_WEIGHT)
    model.exceedance_count_weight = float(config.EXCEEDANCE_COUNT_WEIGHT)
    model.exceedance_pos_weight = float(config.EXCEEDANCE_POS_WEIGHT)
    model.exceedance_focal_gamma = float(config.EXCEEDANCE_FOCAL_GAMMA)
    return model


def validate_prediction_lead_config(config=Config, max_supported_lead=28):
    """Fail early when a requested tube cannot be indexed or shaped safely."""
    leads = prediction_leads(config)
    if any(int(lead) <= 0 for lead in leads):
        raise ValueError(f"Prediction leads must be positive, got {leads}.")
    if tuple(sorted(set(leads))) != leads:
        raise ValueError(f"Prediction leads must be unique and strictly increasing, got {leads}.")
    if max(leads) > int(max_supported_lead):
        raise ValueError(
            f"Maximum prediction lead {max(leads)} exceeds the validated campaign ceiling "
            f"of {int(max_supported_lead)} days."
        )
    if getattr(config, "MULTI_LEAD_TUBE", False):
        center_lead_index(config)
        if len(leads) < 2:
            raise RuntimeError("--multi_lead_tube needs at least two prediction leads.")
    lowpass_history = (
        int(config.GLOBAL_LOWPASS_WINDOW_DAYS) - 1
        if config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 0
    )
    history = required_input_history(config)
    if history < lowpass_history:
        raise RuntimeError(
            f"Required input history {history} does not cover the global low-pass history "
            f"of {lowpass_history} days."
        )
    return leads, history


def parse_int_tuple(text):
    values = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not values:
        raise ValueError("Expected a comma-separated list of integers.")
    return values


def early_stop_score(metric_name, improved_metrics, val_mse, val_ssim):
    metric_name = str(metric_name).lower()
    if metric_name == "tac":
        return float(improved_metrics.get("tac", float("nan")))
    if metric_name == "weekly7_tac":
        return float(improved_metrics.get("weekly7_tac", float("nan")))
    if metric_name == "tube_weekly7_tac":
        return float(improved_metrics.get("tube_weekly7_tac", improved_metrics.get("weekly7_tac", float("nan"))))
    if metric_name == "r2":
        return float(improved_metrics.get("r2", float("nan")))
    if metric_name == "mse_skill":
        return float(improved_metrics.get("mse_skill_vs_persistence", float("nan")))
    if metric_name == "spatial_anom_r2":
        return float(improved_metrics.get("spatial_anom_r2", float("nan")))
    if metric_name == "exceedance_bss":
        return float(improved_metrics.get("exceedance_bss", float("nan")))
    if metric_name == "ssim":
        return float(val_ssim)
    if metric_name == "val_mse":
        # Larger score is always better for the patience logic.
        return -float(val_mse)
    raise ValueError(f"Unknown EARLY_STOP_METRIC={metric_name!r}")


def early_stop_display_value(metric_name, score):
    return -score if str(metric_name).lower() == "val_mse" else score


def run_fold_id(run_name):
    match = re.search(r"cvfold(\d+)", str(run_name))
    if match:
        return int(match.group(1))
    return -1


def append_training_metrics(row):
    os.makedirs(Config.TRAINING_METRICS_DIR, exist_ok=True)
    path = os.path.join(
        Config.TRAINING_METRICS_DIR,
        f"fold_epoch_metrics_{Config.RUN_NAME or cv_split_tag(Config)}.csv",
    )
    fieldnames = [
        "run_name", "fold", "epoch", "train_loss", "val_mse", "val_rmse",
        "val_ssim", "val_r2", "val_tac", "val_sigma_min", "persistence_tac",
        "weekly7_tac", "weekly7_persistence_tac", "weekly7_n_samples",
        "weekly7_mse", "weekly7_persistence_mse",
        "tube_weekly7_tac", "tube_weekly7_persistence_tac", "tube_weekly7_n_samples",
        "tube_weekly7_mse", "tube_weekly7_persistence_mse",
        "exceedance_bss", "exceedance_brier", "exceedance_climo_brier",
        "exceedance_base_rate", "exceedance_pred_rate",
        "lead12_mse", "lead13_mse", "lead14_mse", "lead15_mse",
        "lead16_mse", "lead17_mse", "lead18_mse",
        "lead12_tac", "lead13_tac", "lead14_tac", "lead15_tac",
        "lead16_tac", "lead17_tac", "lead18_tac",
        "spatial_anom_r2", "persistence_spatial_anom_r2",
        "variance_ratio", "gradient_ratio", "extreme_bias", "correlation",
        "mae", "crps", "mse_skill_vs_persistence", "persistence_r2",
        "persistence_mae", "zero_r2", "train_base_mse",
        "train_anomaly_corr_loss", "train_gradient_loss", "train_exceedance_bce_loss",
        "train_exceedance_count_loss", "train_extreme_loss", "train_sigma_min", "lr",
        "early_stop_metric", "early_stop_value", "early_stop_best", "early_stop_failures",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_extended_global_variable_report(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    global_path = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("GLOBAL_DATA_PATH"):
            global_path = ast.literal_eval(stripped.split("=", 1)[1].strip())
            break

    marker = "NEW_GLOBAL_VARIABLES"
    if marker not in text:
        raise ValueError(f"{path} does not define NEW_GLOBAL_VARIABLES")
    start = text.index(marker)
    bracket_start = text.index("[", start)
    bracket_end = text.index("]", bracket_start) + 1
    new_vars = ast.literal_eval(text[bracket_start:bracket_end])
    if not isinstance(new_vars, list) or not all(isinstance(v, str) for v in new_vars):
        raise ValueError(f"Invalid NEW_GLOBAL_VARIABLES in {path}")
    return global_path, new_vars


def apply_extended_global_fields():
    if not Config.USE_EXTENDED_GLOBAL_FIELDS:
        Config.NUM_GLOBAL_CHANNELS = global_effective_channel_count(Config)
        if Config.REQUIRE_EXTENDED_GLOBAL_FIELDS:
            raise RuntimeError(
                "USE_EXTENDED_GLOBAL_FIELDS is False, but this CONUS run requires the "
                f"{Config.EXPECTED_NUM_GLOBAL_CHANNELS}-channel ERA5 global stack."
            )
        return

    report_path = Config.EXTENDED_GLOBAL_VARIABLES_PATH
    if not os.path.exists(report_path):
        Config.NUM_GLOBAL_CHANNELS = global_effective_channel_count(Config)
        if Config.REQUIRE_EXTENDED_GLOBAL_FIELDS:
            raise FileNotFoundError(
                f"Missing extended global-field report: {report_path}. "
                f"This CONUS run requires {Config.EXPECTED_NUM_GLOBAL_CHANNELS} global channels."
            )
        return

    global_path, new_vars = _read_extended_global_variable_report(report_path)
    if global_path:
        Config.GLOBAL_DATA_PATH = global_path

    seen = set(Config.GLOBAL_VARIABLES)
    added = []
    for name in new_vars:
        if name not in seen:
            Config.GLOBAL_VARIABLES.append(name)
            seen.add(name)
            added.append(name)

    base_channels = global_base_channel_count(Config)
    Config.NUM_GLOBAL_CHANNELS = global_effective_channel_count(Config)
    if Config.REQUIRE_EXTENDED_GLOBAL_FIELDS and base_channels != Config.EXPECTED_NUM_GLOBAL_CHANNELS:
        raise RuntimeError(
            f"Expected {Config.EXPECTED_NUM_GLOBAL_CHANNELS} global channels after loading "
            f"{report_path}, got {base_channels}."
        )
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Loaded extended global-field report: {report_path}")
        print(f"  GLOBAL_DATA_PATH: {Config.GLOBAL_DATA_PATH}")
        print(f"  Added global variables: {len(added)}")
        print(f"  Base global variables: {base_channels}")
        print(f"  Global lags: {Config.GLOBAL_LAG_DAYS}")
        if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL:
            print(
                f"  Global decomposition: trailing {Config.GLOBAL_LOWPASS_WINDOW_DAYS}-day "
                f"{Config.GLOBAL_COMPONENT_NAMES}"
            )
        print(f"  Total global input channels: {Config.NUM_GLOBAL_CHANNELS}")


def print_config_banner():
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    mode_name = "DETERMINISTIC (GraphCast)" if Config.DETERMINISTIC else "PROBABILISTIC (GenCast/CFM)"
    print(f"\n{'='*80}")
    print(f"DIRECT 15-DAY PREDICTION - {mode_name} + ICOSAHEDRAL MESH GNN")
    print(f"{'='*80}")
    print(f"Using device: {Config.DEVICE}")
    print(f"Mode: {mode_name}")
    if Config.MULTI_LEAD_TUBE:
        print(
            f"Prediction tube: leads {prediction_leads(Config)} days "
            f"(center t+{Config.LEAD_TIME}); same-init {tube_mean_display_label(Config)} enabled"
        )
        if Config.TUBE_DECODE_CHUNK_SIZE > 0:
            print(
                f"Tube decoder: {Config.TUBE_DECODE_CHUNK_SIZE} leads/chunk "
                "(activation checkpointing during training)"
            )
    else:
        print(f"Lead time: {Config.LEAD_TIME} days DIRECT (no rollout)")
    print(f"Sampling Steps: {Config.CFM_SAMPLING_STEPS} {'(ignored in deterministic)' if Config.DETERMINISTIC else ''}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Random seed: {Config.SEED}")
    print(f"Mesh refinement level: {Config.MESH_REFINEMENT_LEVEL}")
    print(f"Mesh processor rounds: {Config.MESH_PROCESSOR_ROUNDS}")
    print(f"Mesh latent dim: {Config.MESH_LATENT_DIM}")
    print(
        f"Global fields: {global_base_channel_count(Config)} variables x "
        f"{len(Config.GLOBAL_LAG_DAYS)} lags"
        f"{' x ' + str(len(Config.GLOBAL_COMPONENT_NAMES)) + ' components' if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else ''} "
        f"= {Config.NUM_GLOBAL_CHANNELS} "
        f"channels at {Config.GLOBAL_SIZE}"
    )
    print(f"Global data path: {Config.GLOBAL_DATA_PATH}")
    if Config.NUM_GLOBAL_CHANNELS > 0:
        print("Global encoder: circular longitude padding + periodic longitude sampling")
    if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL:
        print(
            "Global input decomposition: "
            f"low20=mean(t-{Config.GLOBAL_LOWPASS_WINDOW_DAYS - 1}:t), "
            "residual=field(t)-low20"
        )
    if Config.PREDICT_PERSISTENCE_RESIDUAL:
        print("Output head: predicts residual added to day-0 persistence")
    if Config.LOCAL_LAG_DAYS:
        print(
            "Local slow-state lags: "
            f"{Config.LOCAL_LAG_VARIABLES} at days {Config.LOCAL_LAG_DAYS} "
            f"({Config.NUM_LOCAL_LAG_CHANNELS} channels)"
        )
    if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
        print(
            "Training target: local climatological anomaly "
            f"(train-year-only {Config.CLIMATOLOGY_WINDOW_DAYS}-day daily climatology)"
        )
    else:
        print("Training target: daily PRISM T2max z-score field")
    print(f"Cross-validation: leave-k-years-out ({cv_split_tag(Config)})")
    if Config.RUN_NAME:
        print(f"Run name: {Config.RUN_NAME}")
    if Config.USE_EXTREME_LOSS:
        print(f"Extreme loss: ON (weight={Config.EXTREME_LOSS_WEIGHT}, a={Config.EXTREME_LOSS_HEAT_BIAS}, b={Config.EXTREME_LOSS_COLD_BIAS})")
    if Config.USE_ANOMALY_CORR_LOSS:
        print(
            "Training loss: "
            f"{1.0 - Config.ANOMALY_CORR_LOSS_WEIGHT:.2f}*MSE + "
            f"{Config.ANOMALY_CORR_LOSS_WEIGHT:.2f}*spatial anomaly-correlation loss"
        )
    if Config.GRADIENT_LOSS_WEIGHT > 0.0:
        print(
            "Spatial gradient loss: "
            f"ON (weight={Config.GRADIENT_LOSS_WEIGHT:.2f}; finite-difference dx/dy)"
        )
    if Config.MULTI_LEAD_TUBE:
        loss_name = "CRPS" if Config.DISTRIBUTIONAL_HEAD and Config.CRPS_LOSS else "MSE"
        print(
            "Tube training loss: "
            f"{Config.TUBE_LOSS_DAILY_WEIGHT:.2f}*mean_daily_{loss_name} + "
            f"{Config.TUBE_LOSS_CENTER_WEIGHT:.2f}*center_t+{Config.LEAD_TIME}_{loss_name} + "
            f"{Config.TUBE_LOSS_WEEKLY_WEIGHT:.2f}*"
            f"same_init_{tube_mean_display_label(Config).replace(' ', '_')}_{loss_name}"
        )
    if Config.DISTRIBUTIONAL_HEAD:
        print(
            "Distributional head: ON "
            f"(Gaussian CRPS={Config.CRPS_LOSS}, sigma_floor={Config.SIGMA_FLOOR:.3f}, "
            f"MSE anchor={Config.MSE_ANCHOR_WEIGHT:.3f})"
        )
    if Config.ENABLE_EXCEEDANCE_HEAD:
        print(
            "Exceedance head: ON "
            f"(BCE weight={Config.EXCEEDANCE_BCE_WEIGHT:.2f}, "
            f"count weight={Config.EXCEEDANCE_COUNT_WEIGHT:.2f}, "
            f"pos_weight={Config.EXCEEDANCE_POS_WEIGHT:.2f}, "
            f"focal_gamma={Config.EXCEEDANCE_FOCAL_GAMMA:.2f})"
        )
    if Config.USE_VALIDATION_PATIENCE:
        print(
            "Validation patience: "
            f"metric={Config.EARLY_STOP_METRIC}, patience={Config.EARLY_STOP_PATIENCE}, "
            f"min_epoch={Config.EARLY_STOP_MIN_EPOCH}, min_delta={Config.EARLY_STOP_MIN_DELTA}"
        )


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_DIR, exist_ok=True)
os.makedirs(Config.HINDCAST_STATS_DIR, exist_ok=True)
os.makedirs(Config.HINDCAST_PAPER_DIR, exist_ok=True)
os.makedirs(Config.TRAINING_METRICS_DIR, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA init failed, falling back to CPU: {e}")
    return torch.device("cpu")


def get_target_variable(nc, config=Config):
    for name in config.TARGET_VARIABLE_CANDIDATES:
        if name in nc.variables:
            return name, nc.variables[name]
    available = sorted(nc.variables.keys())
    raise KeyError(
        f"None of target variables {config.TARGET_VARIABLE_CANDIDATES} were found in "
        f"{config.TRAINING_DATA_PATH}. Available variables include: {available[:40]}"
    )


def target_to_hw_time(raw, target_name):
    if raw.ndim == 4:
        return np.array(raw[:, :, 0, :], dtype=np.float32)
    if raw.ndim == 3:
        return np.array(raw, dtype=np.float32)
    raise ValueError(f"Unexpected {target_name} shape: {raw.shape}")


# ======================================================================================
# SEASON BOUNDARY DETECTION
# ======================================================================================
def detect_continuous_runs(time_values):
    """
    Detect contiguous daily runs in the time array.
    Returns list of (start_idx, end_idx) for each continuous run.
    A gap is any jump > 1.5 days between consecutive entries.
    """
    runs = []
    start = 0
    for i in range(1, len(time_values)):
        if time_values[i] - time_values[i - 1] > 1.5:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(time_values) - 1))
    return runs


def build_valid_indices(runs, lead_time, min_history=2):
    """
    Build valid sample indices ensuring predictor history and the direct target
    day fall within a single continuous run.
    """
    indices = []
    for start, end in runs:
        first_valid = start + min_history
        last_valid = end - lead_time
        for t in range(first_valid, last_valid + 1):
            indices.append(t)
    return indices


# ======================================================================================
# CROSS-VALIDATION SPLIT
# ======================================================================================
def build_crossval_split(valid_indices, time_values, val_stride=None, test_stride=None,
                         val_offset=None, test_offset=None, val_offsets=None,
                         test_offsets=None):
    """
    Leave-k-years-out cross-validation.
    Assigns each index to its calendar year, then holds out every Nth year
    for val and test. This ensures warm and cool decades appear in both
    train and val, removing climate trend confound.

    Default: hold out years at positions 3,8,13,... for val
             and positions 4,9,14,... for test. For k-fold hindcasts,
             pass fold-specific offset tuples so every year can be predicted
             by a model that never trained on that offset group.
    """
    val_stride = Config.CV_STRIDE if val_stride is None else val_stride
    test_stride = Config.CV_STRIDE if test_stride is None else test_stride
    if val_offsets is None:
        if val_offset is None:
            val_offsets = Config.CV_VAL_OFFSETS
        else:
            val_offsets = (val_offset,)
    if test_offsets is None:
        if test_offset is None:
            test_offsets = Config.CV_TEST_OFFSETS
        else:
            test_offsets = (test_offset,)
    val_offsets = set(parse_cv_offsets(val_offsets, val_stride) or ())
    test_offsets = set(parse_cv_offsets(test_offsets, test_stride) or ())
    overlap = val_offsets & test_offsets
    if overlap:
        raise ValueError(f"Validation and test CV offsets overlap: {sorted(overlap)}")

    base_date = datetime(1981, 5, 1)
    idx_years = np.array([
        (base_date + timedelta(days=float(time_values[i]))).year
        for i in valid_indices
    ])
    unique_years = sorted(set(idx_years))

    val_years = set()
    for offset in val_offsets:
        val_years.update(unique_years[offset::val_stride])
    test_years = set()
    for offset in test_offsets:
        test_years.update(unique_years[offset::test_stride])
    train_years = set(unique_years) - val_years - test_years

    train_indices = [i for i, y in zip(valid_indices, idx_years) if y in train_years]
    val_indices_out = [i for i, y in zip(valid_indices, idx_years) if y in val_years]
    test_indices_out = [i for i, y in zip(valid_indices, idx_years) if y in test_years]

    return train_indices, val_indices_out, test_indices_out, train_years, val_years, test_years


# ======================================================================================
# BUILD MESH
# ======================================================================================
def build_mesh_once(config, conus_mask, device, ddp=False):
    cache_path = os.path.join(config.OUTPUT_DIR, "data_cache",
                               f"mesh_level{config.MESH_REFINEMENT_LEVEL}"
                               f"_g2m{config.K_GRID2MESH}_m2g{config.K_MESH2GRID}.pkl")

    if not ddp or dist.get_rank() == 0:
        if os.path.exists(cache_path):
            print(f"Loading cached mesh from {cache_path}")
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)
        else:
            H, W = config.IMAGE_SIZE
            grid_lat = np.linspace(config.CONUS_LAT_RANGE[0], config.CONUS_LAT_RANGE[1], H)
            grid_lon = np.linspace(config.CONUS_LON_RANGE[0], config.CONUS_LON_RANGE[1], W)
            mask_raw = conus_mask.squeeze().cpu().numpy() > 0.5

            mesh = IcosahedralMesh(
                refinement_level=config.MESH_REFINEMENT_LEVEL,
                lat_range=config.CONUS_LAT_RANGE,
                lon_range=config.CONUS_LON_RANGE,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                land_mask=mask_raw,
                buffer_deg=config.MESH_BUFFER_DEG,
                k_grid2mesh=config.K_GRID2MESH,
                k_mesh2grid=config.K_MESH2GRID,
            )

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(mesh, f)
            print(f"Cached mesh to {cache_path}")

    if ddp:
        dist.barrier()
        if dist.get_rank() != 0:
            with open(cache_path, 'rb') as f:
                mesh = pickle.load(f)

    mesh.to_torch(device)

    if is_main_process():
        s = mesh.summary()
        print(f"\nMesh summary:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    return mesh


# ======================================================================================
# EXPONENTIAL EXTREME LOSS (Lopez-Gomez et al., 2023)
# ======================================================================================
def exponential_extreme_loss(pred, target, mask, a=1.0, b=0.0):
    """
    Loss that upweights prediction errors for extreme values.
    a=1, b=0: emphasize positive extremes (heat waves)
    a=0, b=1: emphasize negative extremes (cold spells)
    a=0.5, b=0.5: emphasize both tails

    Reference: Lopez-Gomez et al. (2023) "Global Extreme Heat Forecasting
    Using Neural Weather Models", AI for Earth Systems.
    """
    diff_sq = (target - pred) ** 2
    weight = a * torch.exp(target.clamp(-5, 5)) + b * torch.exp(-target.clamp(-5, 5))
    loss = weight * diff_sq
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def masked_mse_loss(pred, target, mask):
    """Land-only MSE in normalized z-score units."""
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)

    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def finite_nonzero_mean_std_time_chunks(data_hw_time, indices, chunk_size=16):
    """Memory-safe mean/std over H,W,time slices for finite nonzero land values."""
    indices = np.asarray(indices, dtype=np.int64)
    total = 0.0
    total_sq = 0.0
    count = 0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = np.asarray(data_hw_time[:, :, chunk_idx], dtype=np.float32)
        valid = np.isfinite(chunk) & (chunk != 0.0)
        if np.any(valid):
            vals = chunk[valid].astype(np.float64, copy=False)
            total += float(vals.sum(dtype=np.float64))
            total_sq += float((vals * vals).sum(dtype=np.float64))
            count += int(vals.size)
        del chunk, valid
    if count <= 0:
        raise ValueError("No valid finite nonzero values found while computing mean/std.")
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return float(mean), float(np.sqrt(var))


def finite_mean_std_time_chunks(data_hw_time, indices, chunk_size=16):
    """Memory-safe nanmean/nanstd over H,W,time slices."""
    indices = np.asarray(indices, dtype=np.int64)
    total = 0.0
    total_sq = 0.0
    count = 0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = np.asarray(data_hw_time[:, :, chunk_idx], dtype=np.float32)
        valid = np.isfinite(chunk)
        if np.any(valid):
            vals = chunk[valid].astype(np.float64, copy=False)
            total += float(vals.sum(dtype=np.float64))
            total_sq += float((vals * vals).sum(dtype=np.float64))
            count += int(vals.size)
        del chunk, valid
    if count <= 0:
        raise ValueError("No valid finite values found while computing mean/std.")
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return float(mean), float(np.sqrt(var))


def anomaly_corr_loss(pred, target, climo, mask, eps=1e-8):
    """
    1 - per-sample spatial anomaly correlation, averaged over the batch.

    pred/target/climo are normalized z-score fields. climo is the train-year-only
    local daily climatology used by TAC, so this rewards spatial anomaly structure
    without using held-out climatology.
    """
    pred = pred.float()
    target = target.float()
    climo = climo.to(device=pred.device, dtype=pred.dtype)
    mask = mask.to(device=pred.device, dtype=pred.dtype)

    if climo.ndim == 3:
        climo = climo.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if climo.shape != pred.shape:
        climo = climo.expand_as(pred)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)

    pred_anom = (pred - climo) * mask
    truth_anom = (target - climo) * mask

    land_count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    pred_anom = pred_anom - pred_anom.sum(dim=(-2, -1), keepdim=True) / land_count
    truth_anom = truth_anom - truth_anom.sum(dim=(-2, -1), keepdim=True) / land_count
    pred_anom = pred_anom * mask
    truth_anom = truth_anom * mask

    cov = (pred_anom * truth_anom).sum(dim=(-2, -1))
    pred_std = (pred_anom.square().sum(dim=(-2, -1)).clamp_min(eps)).sqrt()
    truth_std = (truth_anom.square().sum(dim=(-2, -1)).clamp_min(eps)).sqrt()
    corr = cov / (pred_std * truth_std + eps)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return (1.0 - corr).mean()


def batch_target_climatology(t_indices, dataset, device):
    if dataset.target_climatology is None:
        return None

    if isinstance(t_indices, torch.Tensor):
        t_values = t_indices.detach().cpu().numpy().reshape(-1)
    else:
        t_values = np.asarray(t_indices).reshape(-1)

    if Config.MULTI_LEAD_TUBE:
        climo_np = np.stack([
            np.stack([
                np.asarray(dataset.target_climatology[int(dataset.doy_values[int(t) + int(lead)])], dtype=np.float32)
                for lead in prediction_leads(Config)
            ], axis=0)
            for t in t_values
        ], axis=0)
        return torch.from_numpy(climo_np).to(device=device, non_blocking=True)

    target_doys = [
        int(dataset.doy_values[int(t) + int(Config.LEAD_TIME)])
        for t in t_values
    ]
    climo_np = np.asarray(dataset.target_climatology[target_doys], dtype=np.float32)
    return torch.from_numpy(climo_np).unsqueeze(1).to(device=device, non_blocking=True)


# ======================================================================================
# IMPROVED METRICS
# ======================================================================================
@torch.inference_mode()
def compute_persistence_baseline(val_dataset, mask, n_samples=200):
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    h, w = Config.IMAGE_SIZE

    if isinstance(mask, torch.Tensor):
        mask_2d = mask[:h, :w].cpu()
    else:
        mask_2d = torch.from_numpy(np.array(mask[:h, :w]))

    all_pred = []
    all_truth = []
    count = 0

    for batch in loader:
        if count >= n_samples:
            break
        (y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, t_idx, batch_mask) = batch
        if Config.MULTI_LEAD_TUBE:
            x_tlead_true = y[0, center_lead_index(Config), :h, :w]
        else:
            x_tlead_true = y[0, 0, :h, :w]
        x_tlead_persist = x_t[0, 0, :h, :w]

        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES and val_dataset.target_climatology is not None:
            target_time_idx = int(t_idx.item()) + int(Config.LEAD_TIME)
            target_doy = int(val_dataset.doy_values[target_time_idx])
            climo = torch.from_numpy(
                np.array(val_dataset.target_climatology[target_doy], dtype=np.float32)
            )
            x_tlead_true = x_tlead_true + climo

        all_pred.append(x_tlead_persist)
        all_truth.append(x_tlead_true)
        count += 1

    preds = torch.stack(all_pred).unsqueeze(1)
    truth = torch.stack(all_truth).unsqueeze(1)

    metrics = calculate_improved_metrics(preds, truth, mask=mask_2d)
    return metrics


def calculate_improved_metrics(pred, truth, mask=None):
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred

    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = truth

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    elif mask is not None:
        mask_np = mask
    else:
        mask_np = None

    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]
        if mask_np is not None and mask_np.ndim == 4:
            mask_np = mask_np[:, 0]

    if mask_np is not None:
        if mask_np.ndim == 2:
            mask_np = np.broadcast_to(mask_np[None, :, :], pred_np.shape)

    if mask_np is not None:
        pred_masked = np.where(mask_np > 0.5, pred_np, np.nan)
        truth_masked = np.where(mask_np > 0.5, truth_np, np.nan)
    else:
        pred_masked = pred_np
        truth_masked = truth_np

    if mask_np is not None:
        pred_spatial_std = np.nanstd(pred_masked, axis=(1, 2)).mean()
        truth_spatial_std = np.nanstd(truth_masked, axis=(1, 2)).mean()
    else:
        pred_spatial_std = np.std(pred_np, axis=(1, 2)).mean()
        truth_spatial_std = np.std(truth_np, axis=(1, 2)).mean()

    variance_ratio = pred_spatial_std / (truth_spatial_std + 1e-8)

    if mask_np is not None:
        pred_for_grad = np.where(mask_np > 0.5, pred_np, 0.0)
        truth_for_grad = np.where(mask_np > 0.5, truth_np, 0.0)
    else:
        pred_for_grad = pred_np
        truth_for_grad = truth_np

    pred_grad_y = np.abs(np.diff(pred_for_grad, axis=1))
    pred_grad_x = np.abs(np.diff(pred_for_grad, axis=2))
    truth_grad_y = np.abs(np.diff(truth_for_grad, axis=1))
    truth_grad_x = np.abs(np.diff(truth_for_grad, axis=2))

    if mask_np is not None:
        mask_grad_y = mask_np[:, :-1, :] * mask_np[:, 1:, :]
        mask_grad_x = mask_np[:, :, :-1] * mask_np[:, :, 1:]
        pred_grad_y = np.where(mask_grad_y > 0.5, pred_grad_y, np.nan)
        pred_grad_x = np.where(mask_grad_x > 0.5, pred_grad_x, np.nan)
        truth_grad_y = np.where(mask_grad_y > 0.5, truth_grad_y, np.nan)
        truth_grad_x = np.where(mask_grad_x > 0.5, truth_grad_x, np.nan)
        pred_grad_mag = np.nanmean(pred_grad_y) + np.nanmean(pred_grad_x)
        truth_grad_mag = np.nanmean(truth_grad_y) + np.nanmean(truth_grad_x)
    else:
        pred_grad_mag = np.mean(pred_grad_y) + np.mean(pred_grad_x)
        truth_grad_mag = np.mean(truth_grad_y) + np.mean(truth_grad_x)

    gradient_ratio = pred_grad_mag / (truth_grad_mag + 1e-8)

    if mask_np is not None:
        valid_pred = pred_masked[~np.isnan(pred_masked)]
        valid_truth = truth_masked[~np.isnan(truth_masked)]
        truth_p95 = np.percentile(valid_truth, 95)
        pred_p95 = np.percentile(valid_pred, 95)
    else:
        truth_p95 = np.percentile(truth_np, 95)
        pred_p95 = np.percentile(pred_np, 95)

    extreme_bias = pred_p95 - truth_p95

    if mask_np is not None:
        corrs = []
        for b in range(pred_masked.shape[0]):
            vp = pred_masked[b][~np.isnan(pred_masked[b])]
            vt = truth_masked[b][~np.isnan(truth_masked[b])]
            if vt.std() > 1e-6 and vp.std() > 1e-6:
                corrs.append(np.corrcoef(vt, vp)[0, 1])
        correlation = float(np.mean(corrs)) if len(corrs) else 0.0
    else:
        corrs = []
        for b in range(pred_np.shape[0]):
            vp = pred_np[b].ravel()
            vt = truth_np[b].ravel()
            if vt.std() > 1e-6 and vp.std() > 1e-6:
                corrs.append(np.corrcoef(vt, vp)[0, 1])
        correlation = float(np.mean(corrs)) if len(corrs) else 0.0

    if mask_np is not None:
        valid_pred = pred_masked[~np.isnan(pred_masked)]
        valid_truth = truth_masked[~np.isnan(truth_masked)]
        ss_res = np.sum((valid_truth - valid_pred) ** 2)
        ss_tot = np.sum((valid_truth - valid_truth.mean()) ** 2) + 1e-8
    else:
        ss_res = np.sum((truth_np - pred_np) ** 2)
        ss_tot = np.sum((truth_np - truth_np.mean()) ** 2) + 1e-8

    r2 = 1 - (ss_res / ss_tot)

    # CRPS: for a deterministic (single-member) forecast, CRPS = MAE.
    # Computed per-sample then averaged, masked to land-only pixels.
    if mask_np is not None:
        mae_per_sample = []
        for b in range(pred_np.shape[0]):
            vp = pred_masked[b][~np.isnan(pred_masked[b])]
            vt = truth_masked[b][~np.isnan(truth_masked[b])]
            if len(vt) > 0:
                mae_per_sample.append(np.mean(np.abs(vt - vp)))
        mae = float(np.mean(mae_per_sample)) if mae_per_sample else 0.0
    else:
        mae = float(np.mean(np.abs(truth_np - pred_np)))

    # For a single deterministic forecast, CRPS = MAE exactly.
    # When ensemble members are available, use compute_ensemble_crps() instead.
    crps = mae

    return {
        'variance_ratio': float(variance_ratio),
        'gradient_ratio': float(gradient_ratio),
        'extreme_bias': float(extreme_bias),
        'correlation': float(correlation),
        'r2': float(r2),
        'mae': float(mae),
        'crps': float(crps),
        'pred_spatial_std': float(pred_spatial_std),
        'truth_spatial_std': float(truth_spatial_std),
    }


def masked_mse_value(pred, truth, mask=None):
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = truth
    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]

    if mask is None:
        return float(np.mean((truth_np - pred_np) ** 2))

    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
    if mask_np.ndim == 2:
        mask_np = np.broadcast_to(mask_np[None, :, :], pred_np.shape)
    elif mask_np.ndim == 4:
        mask_np = mask_np[:, 0]

    err2 = (truth_np - pred_np) ** 2
    valid = mask_np > 0.5
    return float(err2[valid].mean()) if np.any(valid) else 0.0


def calculate_spatial_anomaly_r2(pred, truth, mask=None):
    """
    R2 after removing each sample's masked spatial mean.
    This is a stricter diagnostic than absolute-temperature R2: it asks whether
    the forecast captures within-CONUS spatial anomaly structure, not just the
    easy domain-mean/seasonal/latitude signal.
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = truth
    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]

    mask_np = None
    if mask is not None:
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        if mask_np.ndim == 4:
            mask_np = mask_np[0, 0]
        elif mask_np.ndim == 3:
            mask_np = mask_np[0]
        mask_np = mask_np > 0.5

    ss_res = 0.0
    ss_tot = 0.0
    for b in range(pred_np.shape[0]):
        if mask_np is not None:
            p = pred_np[b][mask_np]
            t = truth_np[b][mask_np]
        else:
            p = pred_np[b].ravel()
            t = truth_np[b].ravel()
        if t.size == 0:
            continue
        p = p - np.mean(p)
        t = t - np.mean(t)
        ss_res += float(np.sum((t - p) ** 2))
        ss_tot += float(np.sum(t ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def build_train_daily_climatology_30day(shared_data, train_indices, norm_stats, config):
    """
    Train-year-only local daily climatology for TAC.

    The target date for a direct sample is t+LEAD_TIME, so climatology is built
    from the training targets' calendar days, then smoothed with a 30-day window.
    Values are returned in normalized units, so target anomalies are simply
    target_z - climatology_z.
    """
    heat_index = shared_data["heat_index"]
    time_values = np.array(shared_data["time_values"])
    doys = compute_doy_array(time_values).astype(np.int16)
    train_t_indices = np.array(train_indices, dtype=np.int64)
    if getattr(config, "MULTI_LEAD_TUBE", False):
        target_indices = np.concatenate(
            [train_t_indices + int(lead) for lead in prediction_leads(config)]
        )
    else:
        target_indices = train_t_indices + int(config.LEAD_TIME)
    target_doys = doys[target_indices].astype(np.int16)

    h, w = config.IMAGE_SIZE
    daily_sum = np.zeros((367, h, w), dtype=np.float32)
    daily_count = np.zeros((367, h, w), dtype=np.uint16)

    for ti, doy in zip(target_indices, target_doys):
        field = np.array(heat_index[:, :, ti], dtype=np.float32)
        valid = np.isfinite(field) & (field != 0.0)
        daily_sum[doy][valid] += field[valid]
        daily_count[doy][valid] += 1

    climo = np.zeros((367, h, w), dtype=np.float32)
    window = max(1, int(config.CLIMATOLOGY_WINDOW_DAYS))
    before = window // 2
    after = window - before - 1
    for doy in range(1, 367):
        lo = max(1, doy - before)
        hi = min(366, doy + after)
        sum_win = daily_sum[lo:hi + 1].sum(axis=0)
        count_win = daily_count[lo:hi + 1].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            climo[doy] = sum_win / np.maximum(count_win, 1)
        climo[doy][count_win == 0] = float(norm_stats["hi_mean"])

    del daily_sum, daily_count
    climo = (climo - float(norm_stats["hi_mean"])) / (float(norm_stats["hi_std"]) + 1e-8)
    climo = np.nan_to_num(climo, nan=0.0, posinf=0.0, neginf=0.0)
    return climo.astype(np.float16)


def get_climatology_cache_path(config=None):
    if config is None:
        config = Config
    suffix = "_tube" + "-".join(str(x) for x in prediction_leads(config)) if getattr(config, "MULTI_LEAD_TUBE", False) else ""
    return os.path.join(
        config.OUTPUT_DIR,
        "data_cache",
        f"local_daily_climo{config.CLIMATOLOGY_WINDOW_DAYS}_direct15{suffix}_{cv_split_tag(config)}.npy",
    )


def load_or_build_train_climatology(shared_data, train_indices, norm_stats, config, ddp=False):
    path = get_climatology_cache_path(config)
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stats_path = get_norm_stats_path(config)
        climo_ok = os.path.exists(path)
        if climo_ok and os.path.exists(stats_path):
            climo_ok = os.path.getmtime(path) >= os.path.getmtime(stats_path)
        if climo_ok:
            print(f"Loading train-year-only local climatology from {path}")
        else:
            print(
                "Building train-year-only local climatology for TAC "
                f"({config.CLIMATOLOGY_WINDOW_DAYS}-day running daily means)..."
            )
            climo = build_train_daily_climatology_30day(
                shared_data, train_indices, norm_stats, config
            )
            np.save(path, climo)
            print(f"  Saved local climatology to {path}")

    if ddp:
        dist.barrier()

    # Keep this in RAM. It is ~0.6 GB as float16 for CONUS, and reading one
    # full climatology slice per sample from network storage would slow training.
    climo = np.load(path)
    if is_main_process():
        print("  Local climatology ready: train years only, no held-out leakage.")
    return climo


MJJAS_MONTHS = (5, 6, 7, 8, 9)


def compute_month_array(time_values):
    base = datetime(1981, 5, 1)
    return np.array([
        (base + timedelta(days=float(tv))).month
        for tv in np.asarray(time_values)
    ], dtype=np.int16)


def _normalize_hi_np(field, norm_stats):
    return (
        np.asarray(field, dtype=np.float32) - float(norm_stats["hi_mean"])
    ) / (float(norm_stats["hi_std"]) + 1e-8)


def get_exceedance_stats_path(config=None):
    if config is None:
        config = Config
    suffix = "_tube" + "-".join(str(x) for x in prediction_leads(config)) if getattr(config, "MULTI_LEAD_TUBE", False) else ""
    fold = "-".join(str(int(x)) for x in getattr(config, "CV_TEST_OFFSETS", ()))
    return os.path.join(
        config.OUTPUT_DIR,
        "data_cache",
        f"month_q95_exceedance_direct15{suffix}_{cv_split_tag(config)}_fold{fold}.npz",
    )


def build_month_q95_exceedance_stats(shared_data, train_indices, norm_stats, config):
    heat_index = shared_data["heat_index"]
    time_values = np.asarray(shared_data["time_values"])
    months = compute_month_array(time_values)
    h, w = config.IMAGE_SIZE
    land_mask = np.isfinite(np.asarray(heat_index[:, :, 0])) & (np.asarray(heat_index[:, :, 0]) != 0.0)
    q95 = np.full((12, h, w), np.nan, dtype=np.float32)
    base_rate = np.full((12, h, w), np.nan, dtype=np.float32)
    train_t = np.asarray(train_indices, dtype=np.int64)

    for month in MJJAS_MONTHS:
        target_indices = []
        for lead in prediction_leads(config):
            idx = train_t + int(lead)
            idx = idx[months[idx] == month]
            if idx.size:
                target_indices.append(idx)
        if not target_indices:
            continue
        target_indices = np.concatenate(target_indices)
        stack = _normalize_hi_np(np.asarray(heat_index[:, :, target_indices], dtype=np.float32), norm_stats)
        stack[~land_mask, :] = np.nan
        q = np.nanpercentile(stack, 95.0, axis=2).astype(np.float32)
        q[~land_mask] = np.nan
        q95[month] = q
        exceed = stack > q[:, :, None]
        valid = np.isfinite(stack) & np.isfinite(q[:, :, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = exceed.sum(axis=2).astype(np.float32) / np.maximum(valid.sum(axis=2), 1)
        rate[~land_mask] = np.nan
        base_rate[month] = rate.astype(np.float32)
        mean_rate = float(np.nanmean(rate))
        print(f"  Exceedance train build check month={month}: mean rate={mean_rate:.4f}")
        if not (0.025 <= mean_rate <= 0.075):
            raise RuntimeError(f"Train exceedance rate for month {month} is not near 5%: {mean_rate:.4f}")
    return q95, base_rate


def load_or_build_exceedance_stats(shared_data, train_indices, norm_stats, config, ddp=False):
    path = get_exceedance_stats_path(config)
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stats_path = get_norm_stats_path(config)
        cache_ok = os.path.exists(path)
        if cache_ok and os.path.exists(stats_path):
            cache_ok = os.path.getmtime(path) >= os.path.getmtime(stats_path)
        if cache_ok:
            print(f"Loading train-year-only exceedance q95/base-rate stats from {path}")
        else:
            print("Building train-year-only month-q95 exceedance stats...")
            q95, base_rate = build_month_q95_exceedance_stats(shared_data, train_indices, norm_stats, config)
            np.savez_compressed(
                path,
                q95_z=q95.astype(np.float32),
                base_rate=base_rate.astype(np.float32),
                cv_split=np.array(cv_split_tag(config), dtype=object),
                cv_test_offsets=np.array(config.CV_TEST_OFFSETS, dtype=np.int16),
                months=np.array(MJJAS_MONTHS, dtype=np.int8),
                hi_mean=np.array(float(norm_stats["hi_mean"]), dtype=np.float32),
                hi_std=np.array(float(norm_stats["hi_std"]), dtype=np.float32),
            )
            print(f"  Saved exceedance stats to {path}")
    if ddp:
        dist.barrier()
    with np.load(path, allow_pickle=False) as data:
        q95 = np.asarray(data["q95_z"], dtype=np.float32)
        base_rate = np.asarray(data["base_rate"], dtype=np.float32)
    if is_main_process():
        for month in MJJAS_MONTHS:
            print(f"  Exceedance q95 ready month={month}: base_rate={np.nanmean(base_rate[month]):.4f}")
    return q95, base_rate


def batch_exceedance_thresholds(t_indices, dataset, q95_z, device):
    if q95_z is None:
        return None
    t_values = t_indices.detach().cpu().numpy().reshape(-1) if isinstance(t_indices, torch.Tensor) else np.asarray(t_indices).reshape(-1)
    month_values = compute_month_array(dataset.time_values)
    if Config.MULTI_LEAD_TUBE:
        q = np.stack([
            np.stack([
                np.asarray(q95_z[int(month_values[int(t) + int(lead)])], dtype=np.float32)
                for lead in prediction_leads(Config)
            ], axis=0)
            for t in t_values
        ], axis=0)
        return torch.from_numpy(q).to(device=device, non_blocking=True)
    q = np.stack([
        np.asarray(q95_z[int(month_values[int(t) + int(Config.LEAD_TIME)])], dtype=np.float32)
        for t in t_values
    ], axis=0)
    return torch.from_numpy(q).unsqueeze(1).to(device=device, non_blocking=True)


def build_exceedance_region_mask_tensor(config, conus_mask):
    mask_np = conus_mask.detach().cpu().numpy() > 0.5
    masks = []
    if publication_region_masks is not None:
        try:
            for _, region_mask in publication_region_masks(config.IMAGE_SIZE).items():
                masks.append((region_mask & mask_np).astype(np.float32))
        except Exception as exc:
            print(f"  Region masks unavailable for exceedance count loss; using CONUS only ({exc}).")
    if not masks:
        masks = [mask_np.astype(np.float32)]
    return torch.from_numpy(np.stack(masks, axis=0).astype(np.float32))


def restore_full_field_from_anomaly(field, target_doys, climo_by_doy, mask=None):
    if (
        not Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES
        or climo_by_doy is None
    ):
        return field

    doys = np.asarray(target_doys, dtype=np.int64)
    climo_np = np.asarray(climo_by_doy[doys], dtype=np.float32)
    climo = torch.from_numpy(climo_np).unsqueeze(1)
    climo = climo.to(device=field.device, dtype=field.dtype)
    out = field + climo
    if mask is not None:
        mask_t = mask.to(device=out.device, dtype=out.dtype)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(1)
        out = out * mask_t + Config.OCEAN_FILL * (1 - mask_t)
    return out


def temporal_anomaly_correlation_from_climo(pred, truth, target_doys, climo_by_doy, mask, chunk_pixels=20000):
    """
    Operational-style TAC:
      1. remove local daily climatology at every grid point,
      2. correlate forecast and observed anomalies across time at each grid point,
      3. average valid gridpoint correlations over CONUS land.
    """
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    truth_np = truth.cpu().numpy() if isinstance(truth, torch.Tensor) else np.asarray(truth)
    if pred_np.ndim == 4:
        pred_np = pred_np[:, 0]
        truth_np = truth_np[:, 0]
    target_doys = np.asarray(target_doys, dtype=np.int64)

    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if mask_np.ndim == 4:
        mask_np = mask_np[0, 0]
    elif mask_np.ndim == 3:
        mask_np = mask_np[0]
    pixel_ids = np.flatnonzero(mask_np.reshape(-1) > 0.5)
    if pixel_ids.size == 0 or pred_np.shape[0] < 2:
        return 0.0

    pred_flat = pred_np.reshape(pred_np.shape[0], -1)
    truth_flat = truth_np.reshape(truth_np.shape[0], -1)
    climo_flat = climo_by_doy.reshape(climo_by_doy.shape[0], -1)

    corr_sum = 0.0
    corr_count = 0
    for start in range(0, pixel_ids.size, chunk_pixels):
        cols = pixel_ids[start:start + chunk_pixels]
        climo = climo_flat[np.ix_(target_doys, cols)].astype(np.float32)
        pred_anom = pred_flat[:, cols].astype(np.float32) - climo
        truth_anom = truth_flat[:, cols].astype(np.float32) - climo

        pred_anom -= pred_anom.mean(axis=0, keepdims=True)
        truth_anom -= truth_anom.mean(axis=0, keepdims=True)
        cov = np.mean(pred_anom * truth_anom, axis=0)
        pred_std = np.sqrt(np.mean(pred_anom ** 2, axis=0))
        truth_std = np.sqrt(np.mean(truth_anom ** 2, axis=0))
        corr = cov / (pred_std * truth_std + 1e-8)
        valid = np.isfinite(corr) & (pred_std > 1e-6) & (truth_std > 1e-6)
        if np.any(valid):
            corr_sum += float(np.sum(corr[valid]))
            corr_count += int(np.sum(valid))

    return corr_sum / max(corr_count, 1)


def compute_ensemble_crps(ensemble_preds, truth, mask=None):
    """
    CRPS for an ensemble forecast using the energy form:
        CRPS = E|X - y| - 0.5 * E|X - X'|

    ensemble_preds: (M, N_samples, H, W) array, M ensemble members
    truth:          (N_samples, H, W) array
    mask:           (H, W) binary array, optional

    Returns per-sample CRPS averaged over samples and land pixels.
    For M=1, this reduces to MAE (the E|X-X'| term is zero).
    """
    if isinstance(ensemble_preds, torch.Tensor):
        ensemble_preds = ensemble_preds.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

    M = ensemble_preds.shape[0]
    N = ensemble_preds.shape[1]

    crps_values = []
    for n in range(N):
        obs = truth[n]  # (H, W)
        members = ensemble_preds[:, n]  # (M, H, W)

        # Term 1: E|X - y| = mean over members of |member - obs|
        abs_errors = np.abs(members - obs[None, :, :])  # (M, H, W)
        term1 = np.mean(abs_errors, axis=0)  # (H, W)

        # Term 2: E|X - X'| = mean over all pairs
        if M > 1:
            pair_diffs = 0.0
            count = 0
            for i in range(M):
                for j in range(i + 1, M):
                    pair_diffs = pair_diffs + np.abs(members[i] - members[j])
                    count += 1
            term2 = pair_diffs / count  # (H, W), average over unique pairs
        else:
            term2 = np.zeros_like(term1)

        crps_map = term1 - 0.5 * term2  # (H, W)

        if mask is not None:
            valid = mask > 0.5
            if valid.any():
                crps_values.append(float(np.mean(crps_map[valid])))
        else:
            crps_values.append(float(np.mean(crps_map)))

    return float(np.mean(crps_values)) if crps_values else 0.0


# ======================================================================================
# DATASET
# ======================================================================================
class ClimateDataset(Dataset):
    def __init__(self, config, mode='train', train_indices=None, val_indices=None, test_indices=None,
                 normalization_stats=None, shared_data=None, target_climatology=None):
        if is_main_process():
            print(f"Initializing ClimateDataset ({mode} mode)...")
        self.config = config
        self.mode = mode
        self.target_climatology = target_climatology

        if shared_data is None:
            print(f"  Loading data to RAM...")
            with NetCDFDataset(config.TRAINING_DATA_PATH, 'r') as nc:
                target_name, target_var = get_target_variable(nc, config)
                raw_shape = target_var.shape
                print(f"  Raw target variable: {target_name}, shape: {raw_shape}")
                self.heat_index = target_to_hw_time(target_var[:], target_name)
                self.n_timesteps = self.heat_index.shape[-1]
                self.geopotential = np.array(nc.variables['geopotential'][:], dtype=np.float32)
                self.soil_moisture = np.array(nc.variables['soil_moisture'][:], dtype=np.float32)
                self.slp = np.array(nc.variables['sea_level_pressure'][:], dtype=np.float32)
                self.cond_train = np.array(nc.variables['CondTrain'][:], dtype=np.float32)
                self.time_values = np.array(nc.variables['time'][:], dtype=np.float64)
            self.topography = None
            self.global_data = None
            print(f"  Data loaded to RAM")
        else:
            if is_main_process():
                print(f"  Using SHARED data arrays (zero-copy)")
            self.heat_index = shared_data['heat_index']
            self.geopotential = shared_data['geopotential']
            self.soil_moisture = shared_data['soil_moisture']
            self.slp = shared_data['slp']
            self.cond_train = shared_data['cond_train']
            self.topography = shared_data['topography']
            self.temperature_2m = shared_data['temperature_2m']
            self.specific_humidity_850 = shared_data['specific_humidity_850']
            self.temperature_850 = shared_data['temperature_850']
            self.u_wind_850 = shared_data['u_wind_850']
            self.v_wind_850 = shared_data['v_wind_850']
            self.geopotential_300 = shared_data['geopotential_300']
            self.global_data = shared_data.get('global_data', None)
            self.time_values = shared_data['time_values']
            self.n_timesteps = self.heat_index.shape[-1]

        # Precompute day-of-year arrays and latitude for TOA
        self.doy_values = compute_doy_array(self.time_values)
        self.doy_sin_arr = np.sin(2.0 * np.pi * self.doy_values / 365.25).astype(np.float32)
        self.doy_cos_arr = np.cos(2.0 * np.pi * self.doy_values / 365.25).astype(np.float32)
        self.lat_1d_deg = np.linspace(config.CONUS_LAT_RANGE[0], config.CONUS_LAT_RANGE[1],
                                       config.IMAGE_SIZE[0])

        if normalization_stats is None and mode == 'train':
            print("  Calculating Z-Score statistics (TRAINING SET ONLY)...")
            if train_indices is None:
                train_indices = list(
                    range(required_input_history(self.config), self.n_timesteps - max_prediction_lead(self.config))
                )

            train_t_indices = np.array(train_indices)
            if self.config.MULTI_LEAD_TUBE:
                target_indices = np.concatenate(
                    [train_t_indices + int(lead) for lead in prediction_leads(self.config)]
                )
            else:
                target_indices = train_t_indices + self.config.LEAD_TIME
            hi_mean, hi_std = finite_nonzero_mean_std_time_chunks(
                self.heat_index,
                target_indices,
                chunk_size=8 if self.config.MULTI_LEAD_TUBE else 16,
            )
            self.hi_mean = torch.tensor(hi_mean, dtype=torch.float32)
            self.hi_std = torch.tensor(hi_std, dtype=torch.float32)

            print(
                f"    Target -> Mean: {self.hi_mean:.4f}, "
                f"Std: {self.hi_std:.4f}"
            )

            predictor_fields = [
                self.geopotential[:, :, 0, :],
                self.soil_moisture,
                self.slp,
                self.temperature_2m,
                self.specific_humidity_850,
                self.temperature_850,
                self.u_wind_850,
                self.v_wind_850,
                self.geopotential_300,
            ]
            stats_means = []
            stats_stds = []
            for field in predictor_fields:
                mean_val, std_val = finite_mean_std_time_chunks(
                    field,
                    train_t_indices,
                    chunk_size=16,
                )
                stats_means.append(mean_val)
                stats_stds.append(std_val)
            self.stats_mean = torch.tensor(stats_means, dtype=torch.float32).view(9, 1, 1)
            self.stats_std = torch.tensor(stats_stds, dtype=torch.float32).view(9, 1, 1)

            cond_train_subset = self.cond_train[:, train_t_indices]
            self.cond_mean = torch.tensor(np.mean(cond_train_subset, axis=1), dtype=torch.float32)
            self.cond_std = torch.tensor(np.std(cond_train_subset, axis=1), dtype=torch.float32)

            if self.topography is not None:
                topo_land = self.topography[self.topography != 0.0]
                self.topo_mean = torch.tensor(float(np.mean(topo_land)), dtype=torch.float32)
                self.topo_std  = torch.tensor(float(np.std(topo_land)),  dtype=torch.float32)

            if self.global_data is not None:
                global_means = []
                global_stds = []
                for var_name, var_data in self.global_data.items():
                    var_train = var_data[:, :, train_t_indices]
                    global_means.append(float(np.nanmean(var_train)))
                    global_stds.append(float(np.nanstd(var_train)))
                self.global_mean = torch.tensor(global_means, dtype=torch.float32).view(-1, 1, 1)
                self.global_std = torch.tensor(global_stds, dtype=torch.float32).view(-1, 1, 1)
                del var_train
            else:
                self.global_mean = None
                self.global_std = None

            # TOA insolation normalization stats
            toa_subset_doys = self.doy_values[train_t_indices[:500]]
            toa_samples = np.stack([compute_toa_insolation(self.lat_1d_deg, doy)
                                    for doy in toa_subset_doys])
            self.toa_mean = torch.tensor(float(np.mean(toa_samples)), dtype=torch.float32)
            self.toa_std  = torch.tensor(float(np.std(toa_samples)),  dtype=torch.float32)
            print(f"    TOA insolation -> Mean: {self.toa_mean:.2f}, Std: {self.toa_std:.2f}")

            del predictor_fields, cond_train_subset
            gc.collect()

        elif normalization_stats is not None:
            self.hi_mean = normalization_stats['hi_mean']
            self.hi_std = normalization_stats['hi_std']
            self.stats_mean = normalization_stats['stats_mean']
            self.stats_std = normalization_stats['stats_std']
            self.cond_mean = normalization_stats['cond_mean']
            self.cond_std = normalization_stats['cond_std']
            self.topo_mean = normalization_stats['topo_mean']
            self.topo_std  = normalization_stats['topo_std']
            self.global_mean = normalization_stats.get('global_mean', None)
            self.global_std = normalization_stats.get('global_std', None)
            self.toa_mean = normalization_stats['toa_mean']
            self.toa_std  = normalization_stats['toa_std']
        else:
            raise ValueError(f"normalization_stats required for mode='{mode}'")

        h, w = self.config.IMAGE_SIZE
        lat_1d = np.linspace(-1.0, 1.0, h)
        lon_1d = np.linspace(-1.0, 1.0, w)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        self.lat_grid = torch.from_numpy(lat_grid).float().unsqueeze(0)
        self.lon_grid = torch.from_numpy(lon_grid).float().unsqueeze(0)

        if mode == 'train':
            self.indices = train_indices if train_indices is not None else list(
                range(required_input_history(self.config), self.n_timesteps - max_prediction_lead(self.config))
            )
        elif mode == 'val':
            self.indices = val_indices if val_indices is not None else list(
                range(required_input_history(self.config), self.n_timesteps - max_prediction_lead(self.config))
            )
        elif mode == 'test':
            self.indices = test_indices if test_indices is not None else list(
                range(required_input_history(self.config), self.n_timesteps - max_prediction_lead(self.config))
            )

        if (
            self.config.TRAIN_ON_CLIMATOLOGY_ANOMALIES
            and self.target_climatology is None
            and normalization_stats is not None
        ):
            raise ValueError(
                "target_climatology is required when TRAIN_ON_CLIMATOLOGY_ANOMALIES=True."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        h, w = self.config.IMAGE_SIZE
        target_leads = prediction_leads(self.config)
        target_times = tuple(t + int(lead) for lead in target_leads)
        target_t = t + self.config.LEAD_TIME
        max_target_t = max(target_times)
        min_history = required_input_history(self.config)
        if t < min_history or max_target_t >= self.n_timesteps:
            raise IndexError(
                f"Invalid direct-forecast sample t={t}: needs t-{min_history} >= 0 and "
                f"max target={max_target_t} < {self.n_timesteps}."
            )

        x_t_slice      = self.heat_index[:, :, t]
        x_tm1_slice    = self.heat_index[:, :, t - 1]
        x_tm2_slice    = self.heat_index[:, :, t - 2]
        x_target_slice = self.heat_index[:, :, target_t]

        raw_x_t = torch.from_numpy(self.heat_index[:, :, t].copy())
        land_mask = (torch.isfinite(raw_x_t) & (raw_x_t != 0.0)).float().unsqueeze(0)

        def normalize_hi(data_slice):
            tensor = torch.from_numpy(data_slice.copy()).float()
            valid = torch.isfinite(tensor) & (tensor != 0.0)

            safe = tensor.clone()
            safe[~valid] = self.hi_mean

            normed = (safe - self.hi_mean) / (self.hi_std + 1e-8)
            normed = normed.unsqueeze(0)

            valid_mask = valid.float().unsqueeze(0)
            return normed * valid_mask + Config.OCEAN_FILL * (1 - valid_mask)

        x_t   = normalize_hi(x_t_slice)
        x_tm1 = normalize_hi(x_tm1_slice)
        x_tm2 = normalize_hi(x_tm2_slice)
        if self.config.MULTI_LEAD_TUBE:
            y = torch.stack(
                [normalize_hi(self.heat_index[:, :, target_time]).squeeze(0)
                 for target_time in target_times],
                dim=0,
            )
        else:
            y = normalize_hi(x_target_slice)

        if self.config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
            if self.config.MULTI_LEAD_TUBE:
                target_doys = [int(self.doy_values[target_time]) for target_time in target_times]
                climo_np = np.stack(
                    [np.array(self.target_climatology[doy], dtype=np.float32, copy=True)
                     for doy in target_doys],
                    axis=0,
                )
                climo = torch.from_numpy(climo_np)
            else:
                target_doy = int(self.doy_values[target_t])
                climo_np = np.array(self.target_climatology[target_doy], dtype=np.float32, copy=True)
                climo = torch.from_numpy(climo_np).unsqueeze(0)
            y = (y - climo) * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        cond_slice = self.cond_train[:, t]
        vec_c = (torch.from_numpy(cond_slice.copy()) - self.cond_mean) / (self.cond_std + 1e-8)

        gp_t   = torch.from_numpy(self.geopotential[:, :, 0, t].copy())
        sm_t   = torch.from_numpy(self.soil_moisture[:, :, t].copy())
        slp_t  = torch.from_numpy(self.slp[:, :, t].copy())
        t2m_t  = torch.from_numpy(self.temperature_2m[:, :, t].copy())
        q850_t = torch.from_numpy(self.specific_humidity_850[:, :, t].copy())
        t850_t = torch.from_numpy(self.temperature_850[:, :, t].copy())
        u850_t = torch.from_numpy(self.u_wind_850[:, :, t].copy())
        v850_t = torch.from_numpy(self.v_wind_850[:, :, t].copy())
        z300_t = torch.from_numpy(self.geopotential_300[:, :, t].copy())

        physics = torch.stack([gp_t, sm_t, slp_t, t2m_t, q850_t, t850_t, u850_t, v850_t, z300_t], dim=0)
        physics = torch.nan_to_num(physics, nan=0.0)
        physics = (physics - self.stats_mean) / (self.stats_std + 1e-8)
        physics = physics * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        local_lag_channels = []
        for lag in self.config.LOCAL_LAG_DAYS:
            src_t = t - int(lag)
            if src_t < 0:
                raise IndexError(f"Local lag {lag} for t={t} points before the record start.")
            for var_name in self.config.LOCAL_LAG_VARIABLES:
                if var_name == "t2max":
                    local_lag_channels.append(normalize_hi(self.heat_index[:, :, src_t]))
                elif var_name == "soil_moisture":
                    sm_lag = torch.from_numpy(self.soil_moisture[:, :, src_t].copy()).float()
                    sm_lag = torch.nan_to_num(sm_lag, nan=0.0)
                    sm_lag = (sm_lag - self.stats_mean[1, 0, 0]) / (self.stats_std[1, 0, 0] + 1e-8)
                    local_lag_channels.append(sm_lag.unsqueeze(0) * land_mask + Config.OCEAN_FILL * (1 - land_mask))
                elif var_name == "temperature_2m":
                    t2m_lag = torch.from_numpy(self.temperature_2m[:, :, src_t].copy()).float()
                    t2m_lag = torch.nan_to_num(t2m_lag, nan=0.0)
                    t2m_lag = (t2m_lag - self.stats_mean[3, 0, 0]) / (self.stats_std[3, 0, 0] + 1e-8)
                    local_lag_channels.append(t2m_lag.unsqueeze(0) * land_mask + Config.OCEAN_FILL * (1 - land_mask))
                else:
                    raise ValueError(f"Unknown LOCAL_LAG_VARIABLES entry: {var_name!r}")

        if local_lag_channels:
            physics = torch.cat([physics, torch.cat(local_lag_channels, dim=0)], dim=0)

        topo = torch.from_numpy(self.topography.copy())
        if topo.shape != (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]):
            topo = topo.T
        topo = ((topo - self.topo_mean) / (self.topo_std + 1e-8)).unsqueeze(0)
        topo = topo * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        lat_c = self.lat_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)
        lon_c = self.lon_grid.clone() * land_mask + Config.OCEAN_FILL * (1 - land_mask)

        # Computed variables: doy sin/cos, TOA insolation, land-sea mask
        doy_sin_ch = torch.full((1, h, w), self.doy_sin_arr[t], dtype=torch.float32)
        doy_cos_ch = torch.full((1, h, w), self.doy_cos_arr[t], dtype=torch.float32)

        toa_1d = compute_toa_insolation(self.lat_1d_deg, self.doy_values[t])
        toa_2d = torch.from_numpy(
            np.broadcast_to(toa_1d[:, None], (h, w)).copy()
        ).float().unsqueeze(0)
        toa_2d = (toa_2d - self.toa_mean) / (self.toa_std + 1e-8)

        land_mask_ch = land_mask  # (1, H, W), values 0/1

        physics = torch.cat([physics, topo, lat_c, lon_c,
                             doy_sin_ch, doy_cos_ch, toa_2d, land_mask_ch], dim=0)
        expected_spatial_channels = Config.NUM_SPATIAL_CONDITIONS - 3
        if physics.shape[0] != expected_spatial_channels:
            raise RuntimeError(
                f"Built {physics.shape[0]} spatial condition channels, "
                f"expected {expected_spatial_channels}."
            )

        if self.global_data is not None and self.global_mean is not None:
            lagged_global_fields = []
            for lag in self.config.GLOBAL_LAG_DAYS:
                src_t = t - int(lag)
                if src_t < 0:
                    raise IndexError(f"Global lag {lag} for t={t} points before the record start.")
                current_channels = []
                low_channels = []
                residual_channels = []
                for var_name, var_data in self.global_data.items():
                    current_np = np.asarray(var_data[:, :, src_t], dtype=np.float32)
                    if self.config.GLOBAL_DECOMPOSE_LOW_RESIDUAL:
                        win = int(self.config.GLOBAL_LOWPASS_WINDOW_DAYS)
                        lo = src_t - win + 1
                        if lo < 0:
                            raise IndexError(
                                f"Global lowpass window t-{win - 1}:t for t={src_t} "
                                "points before the record start."
                            )
                        window_np = np.asarray(var_data[:, :, lo:src_t + 1], dtype=np.float32)
                        with np.errstate(invalid="ignore"):
                            low_np = np.nanmean(window_np, axis=2).astype(np.float32)
                        residual_np = current_np - low_np
                        low_channels.append(torch.from_numpy(low_np))
                        residual_channels.append(torch.from_numpy(residual_np))
                    else:
                        current_channels.append(torch.from_numpy(current_np))

                if self.config.GLOBAL_DECOMPOSE_LOW_RESIDUAL:
                    low_fields = torch.nan_to_num(torch.stack(low_channels, dim=0), nan=0.0)
                    residual_fields = torch.nan_to_num(torch.stack(residual_channels, dim=0), nan=0.0)
                    low_fields = (low_fields - self.global_mean) / (self.global_std + 1e-8)
                    residual_fields = residual_fields / (self.global_std + 1e-8)
                    lag_fields = torch.cat([low_fields, residual_fields], dim=0)
                else:
                    lag_fields = torch.nan_to_num(torch.stack(current_channels, dim=0), nan=0.0)
                    lag_fields = (lag_fields - self.global_mean) / (self.global_std + 1e-8)
                lagged_global_fields.append(lag_fields)
            global_fields = torch.cat(lagged_global_fields, dim=0)
            if global_fields.shape[0] != Config.NUM_GLOBAL_CHANNELS:
                raise RuntimeError(
                    f"Built {global_fields.shape[0]} lagged global channels, "
                    f"expected {Config.NUM_GLOBAL_CHANNELS}."
                )
        else:
            global_fields = torch.zeros(Config.NUM_GLOBAL_CHANNELS, *Config.GLOBAL_SIZE)

        mask = land_mask

        return y, x_t, x_tm1, x_tm2, physics, vec_c, global_fields, t, mask


def get_normalization_stats(dataset):
    stats = {
        'hi_mean':    dataset.hi_mean,
        'hi_std':     dataset.hi_std,
        'stats_mean': dataset.stats_mean,
        'stats_std':  dataset.stats_std,
        'cond_mean':  dataset.cond_mean,
        'cond_std':   dataset.cond_std,
        'topo_mean':  dataset.topo_mean,
        'topo_std':   dataset.topo_std,
        'toa_mean':   dataset.toa_mean,
        'toa_std':    dataset.toa_std,
        'shared_data': {
            'heat_index':            dataset.heat_index,
            'geopotential':          dataset.geopotential,
            'soil_moisture':         dataset.soil_moisture,
            'slp':                   dataset.slp,
            'cond_train':            dataset.cond_train,
            'topography':            dataset.topography,
            'temperature_2m':        dataset.temperature_2m,
            'specific_humidity_850': dataset.specific_humidity_850,
            'temperature_850':       dataset.temperature_850,
            'u_wind_850':            dataset.u_wind_850,
            'v_wind_850':            dataset.v_wind_850,
            'geopotential_300':      dataset.geopotential_300,
            'time_values':           dataset.time_values,
        },
    }
    if dataset.global_data is not None:
        stats['shared_data']['global_data'] = dataset.global_data
    stats['global_mean'] = dataset.global_mean
    stats['global_std'] = dataset.global_std
    return stats

# ======================================================================================
# SSIM
# ======================================================================================
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

def masked_ssim_01(pred, target, mask, fill=0.5):
    mask = mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    pred_m = pred * mask + fill * (1 - mask)
    targ_m = target * mask + fill * (1 - mask)
    return structural_similarity_index_measure(pred_m, targ_m, data_range=1.0)


@torch.inference_mode()
def masked_ssim_01_chunked_cpu(pred, target, mask, fill=0.5, chunk_size=8):
    """Memory-safe validation SSIM. TAC/R2 still use the full validation set."""
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    mask = mask.detach().cpu().to(dtype=pred.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)

    vals = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, pred.shape[0], chunk_size):
        end = min(start + chunk_size, pred.shape[0])
        pred_chunk = pred[start:end]
        targ_chunk = target[start:end]
        mask_chunk = mask.expand_as(pred_chunk)
        pred_m = pred_chunk * mask_chunk + fill * (1 - mask_chunk)
        targ_m = targ_chunk * mask_chunk + fill * (1 - mask_chunk)
        vals.append(structural_similarity_index_measure(pred_m, targ_m, data_range=1.0).detach().cpu())
    return torch.stack(vals).mean() if vals else torch.tensor(float("nan"))

# ======================================================================================
# OCEAN MASKING UTILITIES
# ======================================================================================
def load_conus_mask(config):
    with NetCDFDataset(config.TRAINING_DATA_PATH, 'r') as nc:
        target_name, hi = get_target_variable(nc, config)
        if hi.ndim == 4:
            hi_sample = hi[:, :, 0, 0]
        elif hi.ndim == 3:
            hi_sample = hi[:, :, 0]
        else:
            raise ValueError(f"Unexpected {target_name} ndim: {hi.ndim}")
        mask = (np.abs(hi_sample) > 0.01).astype(np.float32)
    land_fraction = mask.sum() / mask.size
    print(
        f"  Mask loaded from {target_name}: {land_fraction*100:.1f}% land, "
        f"{(1-land_fraction)*100:.1f}% ocean"
    )
    return torch.from_numpy(mask)

def apply_mask_to_predictions(predictions, mask, fill_value=0.0):
    if isinstance(predictions, torch.Tensor):
        device = predictions.device
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device)
        mask = mask.to(device)
        if predictions.ndim == 4:
            mask_broadcast = mask[None, None, :, :].expand_as(predictions)
        elif predictions.ndim == 3:
            mask_broadcast = mask[None, :, :].expand_as(predictions)
        else:
            mask_broadcast = mask
        return predictions * mask_broadcast + fill_value * (1 - mask_broadcast)
    else:
        predictions = np.array(predictions)
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if predictions.ndim == 4:
            mask_broadcast = np.broadcast_to(mask[None, None, :, :], predictions.shape)
        elif predictions.ndim == 3:
            mask_broadcast = np.broadcast_to(mask[None, :, :], predictions.shape)
        else:
            mask_broadcast = mask
        return predictions * mask_broadcast + fill_value * (1 - mask_broadcast)

# ======================================================================================
# DIRECT PREDICTION HELPER
# ======================================================================================
@torch.inference_mode()
def predict_direct(model, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, mask, device):
    """
    Single forward pass: predict the day-15 field directly.
    All inputs should be batched (B, C, H, W) and on device.
    Returns prediction tensor (B, 1, H, W).

    Model signature: model(x_input, t, vec_c, global_fields=...)
    where x_input = cat([x_t, x_tm1, x_tm2, spatial_c]) for deterministic.
    """
    h, w = Config.IMAGE_SIZE
    if Config.DETERMINISTIC:
        from mode_dispatch import (
            _deterministic_input,
            _set_exceedance_logits_from_prediction,
            split_distributional_prediction,
        )
        x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
        dummy_t = torch.full((x_t.shape[0],), 0.5, device=device)
        raw_pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)
        raw_model = model.module if hasattr(model, "module") else model
        if getattr(raw_model, "distributional_head", False):
            pred, sigma = split_distributional_prediction(model, raw_pred, x_t)
            raw_model.last_sigma = sigma.detach()
        elif getattr(raw_model, "predict_persistence_residual", False):
            pred = x_t + raw_pred
        else:
            pred = raw_pred
        _set_exceedance_logits_from_prediction(model, pred)
        if pred.ndim == 4 and pred.shape[1] != mask.shape[1]:
            mask_for_pred = mask.expand_as(pred)
        elif pred.ndim == 5:
            mask_for_pred = mask.unsqueeze(1).expand_as(pred)
        else:
            mask_for_pred = mask
        pred = pred * mask_for_pred + Config.OCEAN_FILL * (1 - mask_for_pred)
        pred = pred.clamp(-4.0, 4.0)
    else:
        # CFM mode: generate_sample signature is
        # (model, spatial_c, vec_c, x_t, x_tm1, x_tm2, global_fields, device, h, w, mask, ...)
        pred_np = generate_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, mask,
            deterministic=False, n_steps=Config.CFM_SAMPLING_STEPS
        )
        pred = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).to(device)
        pred = pred * mask
    return pred


@torch.inference_mode()
def predict_direct_ensemble(models, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, mask, device):
    if not isinstance(models, (list, tuple)):
        return predict_direct(models, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, mask, device)
    preds = [
        predict_direct(model, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, mask, device).float()
        for model in models
    ]
    return torch.stack(preds, dim=0).mean(dim=0)


# ======================================================================================
# VALIDATION (Direct single-pass prediction)
# ======================================================================================
@torch.inference_mode()
def calculate_validation_metrics_cfm(model, val_dataset, device, mask,
                                      n_samples=100, rank=0, world_size=1, ddp=False,
                                      tac_climatology=None,
                                      exceedance_q95=None,
                                      exceedance_base_rate=None):
    """
    Validation via direct single-step prediction at LEAD_TIME=15.
    No autoregressive rollout. Each sample is one forward pass.
    """
    import time
    validation_start = time.time()

    model.eval()
    if len(val_dataset) == 0:
        return 0.0, 0.0, 0.0, {}

    n_samples = min(n_samples, len(val_dataset))
    per_rank = max(1, n_samples // world_size)
    actual_n_samples = per_rank * world_size

    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, len(val_dataset))
    rank_sample_count = end_idx - start_idx

    if is_main_process():
        print(f"\n  Generating {actual_n_samples} direct 15-day predictions "
              f"across {world_size} GPUs ({per_rank} per GPU)...")

    h, w = Config.IMAGE_SIZE
    if isinstance(mask, torch.Tensor):
        mask_4d = mask[:h, :w].unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask_4d = torch.from_numpy(mask[:h, :w]).unsqueeze(0).unsqueeze(0).to(device)
    tube_mode = bool(Config.MULTI_LEAD_TUBE)
    tube_leads = prediction_leads(Config)
    tube_center_idx = center_lead_index(Config)
    if tube_mode and ddp:
        has_climo = torch.tensor(
            [1.0 if tac_climatology is not None else 0.0],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(has_climo, op=dist.ReduceOp.MIN)
        if has_climo.item() < 0.5:
            raise RuntimeError(
                "DDP tube validation requires tac_climatology on every rank "
                "because weekly7/per-lead statistics are streamed per rank."
            )
    mask_2d = mask_4d[0, 0, :h, :w].detach().cpu()
    mask_np = mask_2d.numpy()
    stream_tube_stats = tube_mode and tac_climatology is not None
    tube_weekly7_stats = _empty_tac_stats(h, w) if stream_tube_stats else None
    tube_lead_stats = (
        {int(lead): _empty_tac_stats(h, w) for lead in tube_leads}
        if stream_tube_stats else {}
    )
    tube_weekly7_samples = 0
    exceedance_enabled = (
        bool(getattr(model, "enable_exceedance_head", False))
        and exceedance_q95 is not None
        and exceedance_base_rate is not None
    )
    exceedance_sums = {
        "model_brier": 0.0,
        "climo_brier": 0.0,
        "count": 0.0,
        "truth_pos": 0.0,
        "pred_sum": 0.0,
    }
    month_values = compute_month_array(val_dataset.time_values) if exceedance_enabled else None

    all_preds = []
    all_sigmas = []
    all_truth = []
    all_persist = []
    all_target_doys = []
    all_target_time_indices = []
    all_init_time_indices = []

    for sample_i in range(rank_sample_count):
        dataset_idx = start_idx + sample_i
        batch = val_dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        # Move to device, add batch dimension
        y_dev = y.unsqueeze(0).to(device)
        x_t_dev = x_t.unsqueeze(0).to(device)
        x_tm1_dev = x_tm1.unsqueeze(0).to(device)
        x_tm2_dev = x_tm2.unsqueeze(0).to(device)
        spatial_c_dev = spatial_c.unsqueeze(0).to(device)
        vec_c_dev = vec_c.unsqueeze(0).to(device)
        global_fields_dev = global_fields.unsqueeze(0).to(device)
        mask_dev = batch_mask.unsqueeze(0).to(device)

        # Single forward pass
        pred = predict_direct(model, x_t_dev, x_tm1_dev, x_tm2_dev,
                              spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device)
        raw_model = model.module if hasattr(model, "module") else model
        sigma_pred = getattr(raw_model, "last_sigma", None)

        if tube_mode:
            all_preds.append(pred[0, tube_center_idx, :h, :w].cpu())
            if sigma_pred is not None:
                all_sigmas.append(sigma_pred[0, tube_center_idx, :h, :w].cpu())
            all_truth.append(y[tube_center_idx, :h, :w].cpu())
            center_truth_for_exceedance = y[tube_center_idx, :h, :w]
            if stream_tube_stats:
                pred_tube_np = torch.nan_to_num(
                    pred[0, :, :h, :w].float(), nan=0.0, posinf=0.0, neginf=0.0
                ).cpu().numpy()
                truth_tube_np = torch.nan_to_num(
                    y[:, :h, :w].float(), nan=0.0, posinf=0.0, neginf=0.0
                ).cpu().numpy()
                persist_np = torch.nan_to_num(
                    x_t[0, :h, :w].float(), nan=0.0, posinf=0.0, neginf=0.0
                ).cpu().numpy()
                lead_doys = [
                    int(val_dataset.doy_values[int(t_idx) + int(lead)])
                    for lead in tube_leads
                ]
                pred_mean = pred_tube_np.mean(axis=0, dtype=np.float32)
                truth_mean = truth_tube_np.mean(axis=0, dtype=np.float32)
                climo_mean = np.mean(
                    [np.asarray(tac_climatology[doy], dtype=np.float32) for doy in lead_doys],
                    axis=0,
                    dtype=np.float32,
                )
                _accumulate_tac_stats_with_climo(
                    tube_weekly7_stats,
                    pred_mean,
                    truth_mean,
                    persist_np,
                    climo_mean,
                    mask_np,
                )
                for lead_pos, lead in enumerate(tube_leads):
                    _accumulate_tac_stats(
                        tube_lead_stats[int(lead)],
                        pred_tube_np[lead_pos],
                        truth_tube_np[lead_pos],
                        persist_np,
                        lead_doys[lead_pos],
                        tac_climatology,
                        mask_np,
                    )
                tube_weekly7_samples += 1
        else:
            all_preds.append(pred[0, 0, :h, :w].cpu())
            if sigma_pred is not None:
                all_sigmas.append(sigma_pred[0, 0, :h, :w].cpu())
            all_truth.append(y[0, :h, :w].cpu())
            center_truth_for_exceedance = y[0, :h, :w]
        if exceedance_enabled:
            logits = getattr(model, "last_exceedance_logits", None)
            if logits is None:
                raise RuntimeError("Exceedance validation requested, but last_exceedance_logits is missing.")
            if tube_mode:
                center_logits = logits[0, tube_center_idx, :h, :w]
            else:
                center_logits = logits[0, 0, :h, :w]
            target_month = int(month_values[int(t_idx) + int(Config.LEAD_TIME)])
            q = torch.from_numpy(np.asarray(exceedance_q95[target_month], dtype=np.float32)).to(device=device)
            base = torch.from_numpy(np.asarray(exceedance_base_rate[target_month], dtype=np.float32)).to(device=device)
            label = (center_truth_for_exceedance.to(device) > q).float()
            prob = torch.sigmoid(center_logits.float())
            valid = (mask_4d[0, 0] > 0.5) & torch.isfinite(q) & torch.isfinite(base)
            if valid.any():
                p = prob[valid].clamp(0.0, 1.0)
                y_evt = label[valid]
                b = base[valid].clamp(0.0, 1.0)
                exceedance_sums["model_brier"] += float(((p - y_evt) ** 2).sum().item())
                exceedance_sums["climo_brier"] += float(((b - y_evt) ** 2).sum().item())
                exceedance_sums["count"] += float(p.numel())
                exceedance_sums["truth_pos"] += float(y_evt.sum().item())
                exceedance_sums["pred_sum"] += float(p.sum().item())
        all_persist.append(x_t[0, :h, :w].cpu())
        target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
        all_init_time_indices.append(int(t_idx))
        all_target_time_indices.append(target_time_idx)
        all_target_doys.append(int(val_dataset.doy_values[target_time_idx]))

    local_preds = torch.stack(all_preds)
    local_sigmas = torch.stack(all_sigmas) if all_sigmas else None
    local_truth = torch.stack(all_truth)
    local_persist = torch.stack(all_persist)
    local_target_doys = torch.tensor(all_target_doys, dtype=torch.long)
    local_target_time_indices = torch.tensor(all_target_time_indices, dtype=torch.long)
    local_init_time_indices = torch.tensor(all_init_time_indices, dtype=torch.long)

    if ddp:
        local_preds = local_preds.to(device)
        if local_sigmas is not None:
            local_sigmas = local_sigmas.to(device)
        local_truth = local_truth.to(device)
        local_persist = local_persist.to(device)
        local_target_doys = local_target_doys.to(device)
        local_target_time_indices = local_target_time_indices.to(device)
        local_init_time_indices = local_init_time_indices.to(device)

        gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
        gathered_sigmas = [torch.zeros_like(local_sigmas) for _ in range(world_size)] if local_sigmas is not None else None
        gathered_truth = [torch.zeros_like(local_truth) for _ in range(world_size)]
        gathered_persist = [torch.zeros_like(local_persist) for _ in range(world_size)]
        gathered_target_doys = [torch.zeros_like(local_target_doys) for _ in range(world_size)]
        gathered_target_time_indices = [
            torch.zeros_like(local_target_time_indices) for _ in range(world_size)
        ]
        gathered_init_time_indices = [
            torch.zeros_like(local_init_time_indices) for _ in range(world_size)
        ]

        dist.all_gather(gathered_preds, local_preds)
        if local_sigmas is not None:
            dist.all_gather(gathered_sigmas, local_sigmas)
        dist.all_gather(gathered_truth, local_truth)
        dist.all_gather(gathered_persist, local_persist)
        dist.all_gather(gathered_target_doys, local_target_doys)
        dist.all_gather(gathered_target_time_indices, local_target_time_indices)
        dist.all_gather(gathered_init_time_indices, local_init_time_indices)

        all_preds = torch.cat(gathered_preds, dim=0).cpu()
        all_sigmas = torch.cat(gathered_sigmas, dim=0).cpu() if gathered_sigmas is not None else None
        all_truth = torch.cat(gathered_truth, dim=0).cpu()
        all_persist = torch.cat(gathered_persist, dim=0).cpu()
        all_target_doys = torch.cat(gathered_target_doys, dim=0).cpu().numpy()
        all_target_time_indices = torch.cat(gathered_target_time_indices, dim=0).cpu().numpy()
        all_init_time_indices = torch.cat(gathered_init_time_indices, dim=0).cpu().numpy()
    else:
        all_preds = local_preds.cpu()
        all_sigmas = local_sigmas.cpu() if local_sigmas is not None else None
        all_truth = local_truth.cpu()
        all_persist = local_persist.cpu()
        all_target_doys = local_target_doys.cpu().numpy()
        all_target_time_indices = local_target_time_indices.cpu().numpy()
        all_init_time_indices = local_init_time_indices.cpu().numpy()

    if stream_tube_stats and ddp:
        _ddp_reduce_tac_stats_in_place(tube_weekly7_stats, device=device, dst=0)
        for lead in tube_leads:
            _ddp_reduce_tac_stats_in_place(tube_lead_stats[int(lead)], device=device, dst=0)
        sample_count_tensor = torch.tensor([float(tube_weekly7_samples)], device=device, dtype=torch.float64)
        dist.reduce(sample_count_tensor, dst=0, op=dist.ReduceOp.SUM)
        if is_main_process():
            tube_weekly7_samples = int(round(float(sample_count_tensor.item())))
    if exceedance_enabled and ddp:
        ex_tensor = torch.tensor(
            [
                exceedance_sums["model_brier"],
                exceedance_sums["climo_brier"],
                exceedance_sums["count"],
                exceedance_sums["truth_pos"],
                exceedance_sums["pred_sum"],
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.reduce(ex_tensor, dst=0, op=dist.ReduceOp.SUM)
        if is_main_process():
            exceedance_sums = {
                "model_brier": float(ex_tensor[0].item()),
                "climo_brier": float(ex_tensor[1].item()),
                "count": float(ex_tensor[2].item()),
                "truth_pos": float(ex_tensor[3].item()),
                "pred_sum": float(ex_tensor[4].item()),
            }

    if not is_main_process():
        return 0.0, 0.0, 0.0, {}

    all_preds = all_preds.unsqueeze(1)
    if all_sigmas is not None:
        all_sigmas = all_sigmas.unsqueeze(1)
    all_truth = all_truth.unsqueeze(1)
    all_persist = all_persist.unsqueeze(1)
    all_preds = torch.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
    if all_sigmas is not None:
        all_sigmas = torch.nan_to_num(all_sigmas, nan=float(Config.SIGMA_FLOOR), posinf=float(Config.SIGMA_FLOOR), neginf=float(Config.SIGMA_FLOOR))
    all_truth = torch.nan_to_num(all_truth, nan=0.0, posinf=0.0, neginf=0.0)
    all_persist = torch.nan_to_num(all_persist, nan=0.0, posinf=0.0, neginf=0.0)

    all_preds_eval = restore_full_field_from_anomaly(
        all_preds, all_target_doys, tac_climatology, mask_2d
    )
    all_truth_eval = restore_full_field_from_anomaly(
        all_truth, all_target_doys, tac_climatology, mask_2d
    )

    preds_np = all_preds_eval.numpy()
    truth_np = all_truth_eval.numpy()
    m = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(all_preds_eval).numpy()

    err2 = (preds_np - truth_np) ** 2
    mse_per_sample = (err2 * m).sum(axis=(1,2,3)) / (m.sum(axis=(1,2,3)) + 1e-8)
    avg_mse  = float(mse_per_sample.mean())
    avg_rmse = float(np.sqrt(avg_mse))

    def to_ssim_space(x):
        return ((x + 3.0) / 6.0).clamp(0.0, 1.0)

    ssim_count = min(int(Config.SSIM_VALIDATION_SAMPLES), int(all_preds_eval.shape[0]))
    if ssim_count < all_preds_eval.shape[0]:
        ssim_idx = torch.linspace(0, all_preds_eval.shape[0] - 1, steps=ssim_count).long()
    else:
        ssim_idx = torch.arange(all_preds_eval.shape[0], dtype=torch.long)
    preds_ssim = to_ssim_space(all_preds_eval[ssim_idx])
    truth_ssim = to_ssim_space(all_truth_eval[ssim_idx])
    avg_ssim = masked_ssim_01_chunked_cpu(
        preds_ssim, truth_ssim, mask_2d, fill=0.5, chunk_size=Config.SSIM_CHUNK_SIZE
    ).item()

    improved = calculate_improved_metrics(all_preds_eval, all_truth_eval, mask=mask_2d)
    if all_sigmas is not None:
        sigma_eval = torch.clamp(all_sigmas.to(dtype=all_preds_eval.dtype), min=float(Config.SIGMA_FLOOR))
        mask_eval = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(all_preds_eval)
        w_crps = (all_truth_eval - all_preds_eval) / sigma_eval
        phi = torch.exp(-0.5 * w_crps.square()) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(w_crps / math.sqrt(2.0)))
        crps_map = sigma_eval * (w_crps * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
        improved["crps"] = float((crps_map * mask_eval).sum().item() / mask_eval.sum().clamp_min(1.0).item())
        improved["sigma_min"] = float(sigma_eval.min().item())
    else:
        improved["sigma_min"] = float("nan")
    persistence = calculate_improved_metrics(all_persist, all_truth_eval, mask=mask_2d)
    zero_baseline = calculate_improved_metrics(torch.zeros_like(all_truth_eval), all_truth_eval, mask=mask_2d)
    persistence_mse = masked_mse_value(all_persist, all_truth_eval, mask=mask_2d)
    mse_skill_vs_persistence = 1.0 - avg_mse / (persistence_mse + 1e-8)
    spatial_anom_r2 = calculate_spatial_anomaly_r2(all_preds_eval, all_truth_eval, mask=mask_2d)
    persistence_spatial_anom_r2 = calculate_spatial_anomaly_r2(all_persist, all_truth_eval, mask=mask_2d)
    tac = float("nan")
    persistence_tac = float("nan")
    weekly7_tac = float("nan")
    weekly7_persistence_tac = float("nan")
    weekly7_mse = float("nan")
    weekly7_persistence_mse = float("nan")
    weekly7_samples = 0
    per_lead_mse = {}
    per_lead_tac = {}
    per_lead_persistence_tac = {}
    exceedance_brier = float("nan")
    exceedance_climo_brier = float("nan")
    exceedance_bss = float("nan")
    exceedance_base_rate = float("nan")
    exceedance_pred_rate = float("nan")
    if exceedance_enabled:
        n_ex = max(exceedance_sums["count"], 1.0)
        exceedance_brier = exceedance_sums["model_brier"] / n_ex
        exceedance_climo_brier = exceedance_sums["climo_brier"] / n_ex
        exceedance_bss = 1.0 - exceedance_brier / (exceedance_climo_brier + 1e-8)
        exceedance_base_rate = exceedance_sums["truth_pos"] / n_ex
        exceedance_pred_rate = exceedance_sums["pred_sum"] / n_ex
    if tac_climatology is not None:
        tac = temporal_anomaly_correlation_from_climo(
            all_preds_eval, all_truth_eval, all_target_doys, tac_climatology, mask_2d
        )
        persistence_tac = temporal_anomaly_correlation_from_climo(
            all_persist, all_truth_eval, all_target_doys, tac_climatology, mask_2d
        )
        if tube_mode:
            weekly7_stats = tube_weekly7_stats
            lead_stats = tube_lead_stats
            weekly7_samples = int(tube_weekly7_samples)
            for lead in tube_leads:
                lm, lp, _, _ = summarize_tac_stats(lead_stats[int(lead)], mask_np)
                per_lead_tac[int(lead)] = lm
                per_lead_persistence_tac[int(lead)] = lp
                per_lead_mse[int(lead)] = mse_from_tac_stats(
                    lead_stats[int(lead)], mask_np, persistence=False
                )
        else:
            pred_nhw = all_preds_eval[:, 0].numpy()
            truth_nhw = all_truth_eval[:, 0].numpy()
            persist_nhw = all_persist[:, 0].numpy()
            weekly_records = [
                {
                    "target_time_idx": int(all_target_time_indices[i]),
                    "pred": pred_nhw[i],
                    "truth": truth_nhw[i],
                    "persist": persist_nhw[i],
                }
                for i in range(len(all_target_time_indices))
            ]
            weekly7_stats, weekly7_samples, _ = _weekly7_stats_from_daily_records(
                weekly_records,
                np.asarray(val_dataset.time_values),
                tac_climatology,
                mask_2d.numpy(),
            )
        weekly7_tac, weekly7_persistence_tac, _, _ = summarize_tac_stats(
            weekly7_stats, mask_2d.numpy()
        )
        weekly7_mse = mse_from_tac_stats(weekly7_stats, mask_2d.numpy(), persistence=False)
        weekly7_persistence_mse = mse_from_tac_stats(
            weekly7_stats, mask_2d.numpy(), persistence=True
        )

    validation_total = time.time() - validation_start
    print(f"\n  Validation time: {validation_total:.2f}s "
          f"({actual_n_samples/max(validation_total, 0.01):.1f} samples/sec)")
    print(f"  Metrics (direct 15-day): MSE={avg_mse:.6f}, SSIM={avg_ssim:.4f}, "
          f"R2={improved['r2']:.4f}, CRPS={improved['crps']:.4f}")
    print(f"  Leakage diagnostics: persistence_R2={persistence['r2']:.4f}, "
          f"zero_R2={zero_baseline['r2']:.4f}, "
          f"MSE_skill_vs_persistence={mse_skill_vs_persistence:.4f}")
    print(f"  Spatial-anomaly R2: model={spatial_anom_r2:.4f}, "
          f"persistence={persistence_spatial_anom_r2:.4f}")
    if tac_climatology is not None:
        print(f"  Temporal anomaly correlation (TAC): model={tac:.4f}, "
              f"persistence={persistence_tac:.4f}")
        print(
            f"  {'Tube same-init ' + tube_mean_display_label(Config) if tube_mode else 'True 7-day mean'} "
            f"TAC: model={weekly7_tac:.4f}, "
            f"persistence={weekly7_persistence_tac:.4f}, n={weekly7_samples}"
        )
        if tube_mode and per_lead_tac:
            lead_text = ", ".join(
                f"+{lead}:TAC={per_lead_tac[lead]:.3f}/MSE={per_lead_mse[lead]:.3f}"
                for lead in tube_leads
            )
            print(f"  Per-lead diagnostics: {lead_text}")
    if exceedance_enabled:
        print(
            f"  Exceedance BSS: model={exceedance_bss:+.4f}, "
            f"Brier={exceedance_brier:.5f}, climo={exceedance_climo_brier:.5f}, "
            f"event_rate={exceedance_base_rate:.4f}, pred_rate={exceedance_pred_rate:.4f}"
        )

    return avg_mse, avg_rmse, avg_ssim, {
        'variance_ratio':    improved['variance_ratio'],
        'gradient_ratio':    improved['gradient_ratio'],
        'extreme_bias':      improved['extreme_bias'],
        'correlation':       improved['correlation'],
        'r2':                improved['r2'],
        'mae':               improved['mae'],
        'crps':              improved['crps'],
        'sigma_min':         improved.get('sigma_min', float("nan")),
        'pred_spatial_std':  improved['pred_spatial_std'],
        'truth_spatial_std': improved['truth_spatial_std'],
        'persistence_r2':    persistence['r2'],
        'persistence_mae':   persistence['mae'],
        'persistence_mse':   persistence_mse,
        'zero_r2':           zero_baseline['r2'],
        'mse_skill_vs_persistence': mse_skill_vs_persistence,
        'spatial_anom_r2':   spatial_anom_r2,
        'persistence_spatial_anom_r2': persistence_spatial_anom_r2,
        'tac':               tac,
        'persistence_tac':   persistence_tac,
        'weekly7_tac':       weekly7_tac,
        'weekly7_persistence_tac': weekly7_persistence_tac,
        'weekly7_n_samples': weekly7_samples,
        'weekly7_mse':       weekly7_mse,
        'weekly7_persistence_mse': weekly7_persistence_mse,
        'tube_weekly7_tac': weekly7_tac if tube_mode else float("nan"),
        'tube_weekly7_persistence_tac': weekly7_persistence_tac if tube_mode else float("nan"),
        'tube_weekly7_n_samples': weekly7_samples if tube_mode else 0,
        'tube_weekly7_mse': weekly7_mse if tube_mode else float("nan"),
        'tube_weekly7_persistence_mse': weekly7_persistence_mse if tube_mode else float("nan"),
        'exceedance_bss': exceedance_bss,
        'exceedance_brier': exceedance_brier,
        'exceedance_climo_brier': exceedance_climo_brier,
        'exceedance_base_rate': exceedance_base_rate,
        'exceedance_pred_rate': exceedance_pred_rate,
        **{f"lead{int(lead)}_mse": per_lead_mse.get(int(lead), float("nan")) for lead in (12, 13, 14, 15, 16, 17, 18)},
        **{f"lead{int(lead)}_tac": per_lead_tac.get(int(lead), float("nan")) for lead in (12, 13, 14, 15, 16, 17, 18)},
    }


STAT_KEYS = (
    "pred_sum",
    "truth_sum",
    "pred_sq_sum",
    "truth_sq_sum",
    "pred_truth_sum",
    "persist_sum",
    "persist_sq_sum",
    "persist_truth_sum",
    "count",
)


def _empty_tac_stats(h, w):
    return {
        "pred_sum": np.zeros((h, w), dtype=np.float64),
        "truth_sum": np.zeros((h, w), dtype=np.float64),
        "pred_sq_sum": np.zeros((h, w), dtype=np.float64),
        "truth_sq_sum": np.zeros((h, w), dtype=np.float64),
        "pred_truth_sum": np.zeros((h, w), dtype=np.float64),
        "persist_sum": np.zeros((h, w), dtype=np.float64),
        "persist_sq_sum": np.zeros((h, w), dtype=np.float64),
        "persist_truth_sum": np.zeros((h, w), dtype=np.float64),
        "count": np.zeros((h, w), dtype=np.float64),
    }


def _ddp_reduce_tac_stats_in_place(stats, device, dst=0):
    """Sum compact TAC sufficient-stat arrays across DDP ranks without gathering maps."""
    for key in STAT_KEYS:
        tensor = torch.from_numpy(stats[key]).to(device=device)
        dist.reduce(tensor, dst=dst, op=dist.ReduceOp.SUM)
        if is_main_process():
            stats[key][...] = tensor.cpu().numpy()


def _accumulate_tac_stats_with_climo(stats, pred, truth, persist, climo, mask_np):
    climo = np.asarray(climo, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    persist = np.asarray(persist, dtype=np.float32)
    valid = (
        (mask_np > 0.5)
        & np.isfinite(pred)
        & np.isfinite(truth)
        & np.isfinite(persist)
        & np.isfinite(climo)
    )
    if not np.any(valid):
        return

    pred_anom = np.zeros_like(pred, dtype=np.float32)
    truth_anom = np.zeros_like(truth, dtype=np.float32)
    persist_anom = np.zeros_like(persist, dtype=np.float32)
    pred_anom[valid] = pred[valid] - climo[valid]
    truth_anom[valid] = truth[valid] - climo[valid]
    persist_anom[valid] = persist[valid] - climo[valid]
    valid_f = valid.astype(np.float64)

    pa = pred_anom.astype(np.float64, copy=False)
    ta = truth_anom.astype(np.float64, copy=False)
    xa = persist_anom.astype(np.float64, copy=False)

    stats["pred_sum"] += pa
    stats["truth_sum"] += ta
    stats["pred_sq_sum"] += pa * pa
    stats["truth_sq_sum"] += ta * ta
    stats["pred_truth_sum"] += pa * ta
    stats["persist_sum"] += xa
    stats["persist_sq_sum"] += xa * xa
    stats["persist_truth_sum"] += xa * ta
    stats["count"] += valid_f


def _accumulate_tac_stats(stats, pred, truth, persist, target_doy, climo_by_doy, mask_np):
    climo = climo_by_doy[int(target_doy)].astype(np.float32)
    _accumulate_tac_stats_with_climo(stats, pred, truth, persist, climo, mask_np)


def _corr_from_tac_sums(x_sum, y_sum, x_sq_sum, y_sq_sum, xy_sum, count, mask_np):
    valid = (mask_np > 0.5) & (count >= 2)
    cov = xy_sum - (x_sum * y_sum) / np.maximum(count, 1.0)
    x_var = x_sq_sum - (x_sum * x_sum) / np.maximum(count, 1.0)
    y_var = y_sq_sum - (y_sum * y_sum) / np.maximum(count, 1.0)
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid &= denom > 1e-12
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    return corr, float(np.nanmean(corr[valid])) if np.any(valid) else float("nan")


def summarize_tac_stats(stats, mask_np):
    model_corr, model_tac = _corr_from_tac_sums(
        stats["pred_sum"], stats["truth_sum"],
        stats["pred_sq_sum"], stats["truth_sq_sum"],
        stats["pred_truth_sum"], stats["count"], mask_np,
    )
    persistence_corr, persistence_tac = _corr_from_tac_sums(
        stats["persist_sum"], stats["truth_sum"],
        stats["persist_sq_sum"], stats["truth_sq_sum"],
        stats["persist_truth_sum"], stats["count"], mask_np,
    )
    return model_tac, persistence_tac, model_corr, persistence_corr


def mse_from_tac_stats(stats, mask_np, persistence=False):
    prefix_sq = "persist_sq_sum" if persistence else "pred_sq_sum"
    prefix_xy = "persist_truth_sum" if persistence else "pred_truth_sum"
    count = stats["count"]
    valid = (mask_np > 0.5) & (count > 0)
    sqerr_sum = stats[prefix_sq] + stats["truth_sq_sum"] - 2.0 * stats[prefix_xy]
    per_pixel = np.full(count.shape, np.nan, dtype=np.float64)
    per_pixel[valid] = sqerr_sum[valid] / np.maximum(count[valid], 1.0)
    return float(np.nanmean(per_pixel[valid])) if np.any(valid) else float("nan")


def _target_doy_from_time_value(time_value):
    return int((datetime(1981, 5, 1) + timedelta(days=float(time_value))).timetuple().tm_yday)


def _weekly7_stats_from_daily_records(records, time_values, climo_by_doy, mask_np, half_window=3):
    """
    Compute true centered 7-day mean TAC stats from daily forecast records.

    Each weekly sample averages seven daily model predictions, seven matching
    truths, and seven matching day-15 persistence forecasts before anomaly
    correlation is accumulated. This is the operational-style weekly metric,
    unlike the old diagnostic that averaged truth only.
    """
    if Config.MULTI_LEAD_TUBE:
        raise RuntimeError(
            "_weekly7_stats_from_daily_records is a neighboring-init daily-mode diagnostic "
            "and must not be called in multi-lead tube mode."
        )
    h, w = mask_np.shape
    stats = _empty_tac_stats(h, w)
    monthly_stats = {}
    if not records:
        return stats, 0, monthly_stats

    by_time = {int(rec["target_time_idx"]): rec for rec in records}
    target_times = sorted(by_time)
    samples = 0

    for center_time_idx in target_times:
        window_time_indices = list(
            range(center_time_idx - int(half_window), center_time_idx + int(half_window) + 1)
        )
        window_records = [by_time.get(t) for t in window_time_indices]
        if any(rec is None for rec in window_records):
            continue
        window_values = np.asarray(time_values[window_time_indices], dtype=np.float64)
        if len(window_values) != 2 * int(half_window) + 1:
            continue
        if not np.allclose(np.diff(window_values), 1.0, atol=1e-6):
            continue

        pred_mean = np.mean([rec["pred"] for rec in window_records], axis=0, dtype=np.float32)
        truth_mean = np.mean([rec["truth"] for rec in window_records], axis=0, dtype=np.float32)
        persist_mean = np.mean([rec["persist"] for rec in window_records], axis=0, dtype=np.float32)
        window_doys = [_target_doy_from_time_value(v) for v in window_values]
        climo_mean = np.mean(
            [np.asarray(climo_by_doy[int(doy)], dtype=np.float32) for doy in window_doys],
            axis=0,
            dtype=np.float32,
        )

        _accumulate_tac_stats_with_climo(
            stats, pred_mean, truth_mean, persist_mean, climo_mean, mask_np
        )
        center_dt = datetime(1981, 5, 1) + timedelta(days=float(time_values[center_time_idx]))
        month = int(center_dt.month)
        if month not in monthly_stats:
            monthly_stats[month] = _empty_tac_stats(h, w)
        _accumulate_tac_stats_with_climo(
            monthly_stats[month], pred_mean, truth_mean, persist_mean, climo_mean, mask_np
        )
        samples += 1

    return stats, samples, monthly_stats


def _weekly_mean_target_z(dataset, center_time_idx, half_window=3, mask_np=None):
    """Return a normalized 7-day mean truth field centered on center_time_idx."""
    center = int(center_time_idx)
    lo = center - int(half_window)
    hi = center + int(half_window)
    if lo < 0 or hi >= dataset.n_timesteps:
        return None

    time_values = np.asarray(dataset.time_values)
    window_times = time_values[lo:hi + 1]
    expected_len = 2 * int(half_window) + 1
    if len(window_times) != expected_len or not np.all(np.diff(window_times) == 1):
        return None

    stack = np.asarray(dataset.heat_index[:, :, lo:hi + 1], dtype=np.float32)
    valid_stack = np.isfinite(stack) & (stack != 0.0)
    count = valid_stack.sum(axis=2)
    total = np.where(valid_stack, stack, 0.0).sum(axis=2)
    valid = count > 0
    if mask_np is not None:
        valid &= np.asarray(mask_np) > 0.5
    if not np.any(valid):
        return None

    hi_mean = float(dataset.hi_mean.detach().cpu().item() if isinstance(dataset.hi_mean, torch.Tensor) else dataset.hi_mean)
    hi_std = float(dataset.hi_std.detach().cpu().item() if isinstance(dataset.hi_std, torch.Tensor) else dataset.hi_std)
    mean_field = np.full(total.shape, hi_mean, dtype=np.float32)
    mean_field[valid] = total[valid] / np.maximum(count[valid], 1)
    normed = (mean_field - hi_mean) / (hi_std + 1e-8)
    normed[~valid] = Config.OCEAN_FILL
    return normed.astype(np.float32)


def sample_skill_summary(pred, truth, persist, mask_np):
    valid = (mask_np > 0.5) & np.isfinite(pred) & np.isfinite(truth) & np.isfinite(persist)
    if not np.any(valid):
        return {
            "mae": np.nan, "mse": np.nan, "r2": np.nan, "corr": np.nan,
            "persist_mae": np.nan, "persist_mse": np.nan,
            "truth_mean": np.nan, "pred_mean": np.nan, "persist_mean": np.nan,
            "truth_p95": np.nan, "pred_p95": np.nan, "heat_area_frac": np.nan,
        }
    p = pred[valid].astype(np.float64)
    t = truth[valid].astype(np.float64)
    x = persist[valid].astype(np.float64)
    err = p - t
    perr = x - t
    ss_tot = np.sum((t - np.mean(t)) ** 2) + 1e-8
    corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 1e-8 and np.std(t) > 1e-8 else np.nan
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot),
        "corr": float(corr),
        "persist_mae": float(np.mean(np.abs(perr))),
        "persist_mse": float(np.mean(perr ** 2)),
        "truth_mean": float(np.mean(t)),
        "pred_mean": float(np.mean(p)),
        "persist_mean": float(np.mean(x)),
        "truth_p95": float(np.percentile(t, 95)),
        "pred_p95": float(np.percentile(p, 95)),
        "heat_area_frac": float(np.mean(t > 1.5)),
    }


def _target_datetime(dataset, time_index):
    base = datetime(1981, 5, 1)
    return base + timedelta(days=float(dataset.time_values[int(time_index)]))


def _paper_map_record(dataset_idx, t_idx, target_time_idx, target_doy, pred_np, truth_np,
                      persist_np, summary, reason, climo_np=None,
                      pred_tube=None, truth_tube=None):
    init_dt = _target_datetime_obj_from_time_values(t_idx)
    target_dt = _target_datetime_obj_from_time_values(target_time_idx)
    record = {
        "dataset_idx": int(dataset_idx),
        "t_idx": int(t_idx),
        "target_time_idx": int(target_time_idx),
        "target_doy": int(target_doy),
        "init_date": init_dt.strftime("%Y-%m-%d"),
        "target_date": target_dt.strftime("%Y-%m-%d"),
        "year": int(target_dt.year),
        "month": int(target_dt.month),
        "reason": str(reason),
        "summary": summary,
        "pred": pred_np.astype(np.float16),
        "truth": truth_np.astype(np.float16),
        "persist": persist_np.astype(np.float16),
    }
    if climo_np is not None:
        record["climo"] = np.asarray(climo_np, dtype=np.float32).astype(np.float16)
    if pred_tube is not None and truth_tube is not None:
        record["pred_tube"] = np.asarray(pred_tube, dtype=np.float32).astype(np.float16)
        record["truth_tube"] = np.asarray(truth_tube, dtype=np.float32).astype(np.float16)
    return record


def _target_datetime_obj_from_time_values(time_index):
    return datetime(1981, 5, 1) + timedelta(days=float(_ACTIVE_EXPORT_TIME_VALUES[int(time_index)]))


_ACTIVE_EXPORT_TIME_VALUES = None


def _reservoir_update(bucket, seen_count, cap, record, rng):
    if cap <= 0:
        return
    if len(bucket) < cap:
        bucket.append(record)
        return
    j = int(rng.integers(0, seen_count))
    if j < cap:
        bucket[j] = record


def _save_paper_hindcast_outputs(paper_dir, run_name, split_name, sample_rows,
                                 selected_records, monthly_stats, mask_np,
                                 weekly7_monthly_stats=None):
    os.makedirs(paper_dir, exist_ok=True)
    prefix = f"{run_name or cv_split_tag(Config)}_{split_name}"

    if sample_rows:
        keys = sorted(sample_rows[0].keys())
        summary = {key: np.array([row[key] for row in sample_rows]) for key in keys}
    else:
        summary = {}

    summary_path = os.path.join(paper_dir, f"hindcast_sample_summary_{prefix}.npz")
    np.savez_compressed(summary_path, **summary)

    unique = {}
    for rec in selected_records:
        unique.setdefault((rec["dataset_idx"], rec["target_date"]), rec)
    records = list(unique.values())
    maps_path = os.path.join(paper_dir, f"hindcast_selected_maps_{prefix}.npz")
    if records:
        map_summary_keys = sorted(records[0]["summary"].keys())
        np.savez_compressed(
            maps_path,
            pred=np.stack([rec["pred"] for rec in records]),
            truth=np.stack([rec["truth"] for rec in records]),
            persist=np.stack([rec["persist"] for rec in records]),
            **(
                {"climo": np.stack([rec["climo"] for rec in records])}
                if all("climo" in rec for rec in records) else {}
            ),
            **(
                {
                    "pred_tube": np.stack([rec["pred_tube"] for rec in records]),
                    "truth_tube": np.stack([rec["truth_tube"] for rec in records]),
                    "prediction_leads": np.array(prediction_leads(Config), dtype=np.int16),
                }
                if all("pred_tube" in rec and "truth_tube" in rec for rec in records) else {}
            ),
            dataset_idx=np.array([rec["dataset_idx"] for rec in records], dtype=np.int32),
            t_idx=np.array([rec["t_idx"] for rec in records], dtype=np.int32),
            target_time_idx=np.array([rec["target_time_idx"] for rec in records], dtype=np.int32),
            target_doys=np.array([rec["target_doy"] for rec in records], dtype=np.int16),
            init_dates=np.array([rec["init_date"] for rec in records]),
            target_dates=np.array([rec["target_date"] for rec in records]),
            years=np.array([rec["year"] for rec in records], dtype=np.int16),
            months=np.array([rec["month"] for rec in records], dtype=np.int8),
            fold_ids=np.full(len(records), run_fold_id(run_name), dtype=np.int8),
            reasons=np.array([rec["reason"] for rec in records]),
            mask=mask_np.astype(np.uint8),
            **{
                key: np.array([rec["summary"][key] for rec in records], dtype=np.float32)
                for key in map_summary_keys
            },
        )
    else:
        np.savez_compressed(maps_path, mask=mask_np.astype(np.uint8))

    weekly7_monthly_stats = weekly7_monthly_stats or {}
    months = sorted(monthly_stats.keys())
    monthly_path = os.path.join(paper_dir, f"hindcast_monthly_stats_{prefix}.npz")
    if months:
        payload = dict(
            months=np.array(months, dtype=np.int8),
            mask=mask_np.astype(np.uint8),
            **{
                f"{key}_by_month": np.stack([monthly_stats[m][key] for m in months])
                for key in STAT_KEYS
            },
        )
        weekly7_months = sorted(weekly7_monthly_stats.keys())
        if weekly7_months:
            payload.update(
                weekly7_months=np.array(weekly7_months, dtype=np.int8),
                **{
                    f"weekly7_{key}_by_month": np.stack(
                        [weekly7_monthly_stats[m][key] for m in weekly7_months]
                    )
                    for key in STAT_KEYS
                },
            )
        np.savez_compressed(monthly_path, **payload)
    else:
        np.savez_compressed(monthly_path, mask=mask_np.astype(np.uint8))

    print(f"  Saved paper sample summary to {summary_path}")
    print(f"  Saved selected paper maps to {maps_path}")
    print(f"  Saved monthly paper stats to {monthly_path}")


def norm_stats_match_npz(s, config=Config):
    stats_target_half_window = int(s["target_half_window"]) if "target_half_window" in s else 0
    stats_lags = (
        tuple(np.atleast_1d(s["global_lag_days"]).astype(int).tolist())
        if "global_lag_days" in s else ()
    )
    stats_global_decompose = (
        bool(int(s["global_decompose_low_residual"]))
        if "global_decompose_low_residual" in s else False
    )
    stats_global_lowpass_window = (
        int(s["global_lowpass_window_days"])
        if "global_lowpass_window_days" in s else 0
    )
    stats_local_lags = (
        tuple(np.atleast_1d(s["local_lag_days"]).astype(int).tolist())
        if "local_lag_days" in s else ()
    )
    stats_local_vars = (
        tuple(str(x) for x in np.atleast_1d(s["local_lag_variables"]).tolist())
        if "local_lag_variables" in s else ()
    )
    stats_anomaly_target = (
        bool(int(s["train_on_climatology_anomalies"]))
        if "train_on_climatology_anomalies" in s else False
    )
    stats_multi_lead_tube = (
        bool(int(s["multi_lead_tube"]))
        if "multi_lead_tube" in s else False
    )
    stats_prediction_leads = (
        tuple(np.atleast_1d(s["prediction_leads"]).astype(int).tolist())
        if "prediction_leads" in s else (int(s["lead_time"]),) if "lead_time" in s else ()
    )
    return (
        "global_mean" in s
        and int(s["global_mean"].shape[0]) == global_base_channel_count(config)
        and "lead_time" in s
        and int(s["lead_time"]) == config.LEAD_TIME
        and stats_multi_lead_tube == bool(config.MULTI_LEAD_TUBE)
        and stats_prediction_leads == prediction_leads(config)
        and stats_target_half_window == 0
        and stats_lags == tuple(int(x) for x in config.GLOBAL_LAG_DAYS)
        and stats_global_decompose == bool(config.GLOBAL_DECOMPOSE_LOW_RESIDUAL)
        and stats_global_lowpass_window == (
            int(config.GLOBAL_LOWPASS_WINDOW_DAYS) if config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 0
        )
        and stats_local_lags == tuple(int(x) for x in config.LOCAL_LAG_DAYS)
        and stats_local_vars == tuple(str(x) for x in config.LOCAL_LAG_VARIABLES)
        and stats_anomaly_target == bool(config.TRAIN_ON_CLIMATOLOGY_ANOMALIES)
        and "cv_stride" in s
        and int(s["cv_stride"]) == config.CV_STRIDE
        and "cv_val_offsets" in s
        and tuple(np.atleast_1d(s["cv_val_offsets"]).astype(int).tolist()) == config.CV_VAL_OFFSETS
        and "cv_test_offsets" in s
        and tuple(np.atleast_1d(s["cv_test_offsets"]).astype(int).tolist()) == config.CV_TEST_OFFSETS
    )


def load_norm_stats_npz(stats_path):
    s = np.load(stats_path)
    if not norm_stats_match_npz(s, Config):
        raise RuntimeError(
            f"Normalization stats at {stats_path} do not match the active "
            f"{global_base_channel_count(Config)}-variable/"
            f"{Config.NUM_GLOBAL_CHANNELS}-lagged-channel global stack, "
            f"global_decompose={Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL}, "
            f"global_lowpass_window={Config.GLOBAL_LOWPASS_WINDOW_DAYS}, "
            f"local_lags={Config.LOCAL_LAG_DAYS}/{Config.LOCAL_LAG_VARIABLES}, "
            f"anomaly_target={Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES}, "
            f"and {cv_split_tag(Config)}. "
            "Re-run training setup for this fold or delete stale norm stats."
        )
    norm_stats = {
        'hi_mean': torch.tensor(float(s['hi_mean'])), 'hi_std': torch.tensor(float(s['hi_std'])),
        'stats_mean': torch.from_numpy(s['stats_mean']), 'stats_std': torch.from_numpy(s['stats_std']),
        'cond_mean': torch.from_numpy(s['cond_mean']), 'cond_std': torch.from_numpy(s['cond_std']),
        'topo_mean': torch.tensor(float(s['topo_mean'])), 'topo_std': torch.tensor(float(s['topo_std'])),
        'toa_mean': torch.tensor(float(s['toa_mean'])), 'toa_std': torch.tensor(float(s['toa_std'])),
    }
    if 'global_mean' in s:
        norm_stats['global_mean'] = torch.from_numpy(s['global_mean'])
        norm_stats['global_std'] = torch.from_numpy(s['global_std'])
    else:
        norm_stats['global_mean'] = None
        norm_stats['global_std'] = None
    s.close()
    return norm_stats


@torch.inference_mode()
def export_hindcast_tac_stats(model, dataset, split_name, output_path, device, mask,
                              tac_climatology, rank=0, world_size=1, ddp=False,
                              max_samples=None, save_paper_data=None,
                              paper_output_dir=None, maps_per_month=None,
                              heat_maps_per_fold=None):
    global _ACTIVE_EXPORT_TIME_VALUES
    _ACTIVE_EXPORT_TIME_VALUES = dataset.time_values
    export_models = model if isinstance(model, (list, tuple)) else [model]
    for export_model in export_models:
        export_model.eval()
    h, w = Config.IMAGE_SIZE
    mask_np = mask[:h, :w].detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask[:h, :w])
    mask_4d = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
    stats = _empty_tac_stats(h, w)
    weekly7_stats = _empty_tac_stats(h, w)
    weekly7_monthly_stats = {}
    weekly7_samples = 0
    tube_mode = bool(Config.MULTI_LEAD_TUBE)
    tube_leads = prediction_leads(Config)
    tube_center_idx = center_lead_index(Config)
    weekly_records = [] if (not ddp and not tube_mode) else None
    valid_mask = mask_np > 0.5

    save_paper_data = Config.SAVE_HINDCAST_PAPER_DATA if save_paper_data is None else bool(save_paper_data)
    if ddp and save_paper_data and is_main_process():
        print("  Paper-data export is skipped in DDP export mode; run export_hindcast with nproc_per_node=1.")
    save_paper_data = save_paper_data and not ddp
    paper_output_dir = paper_output_dir or Config.HINDCAST_PAPER_DIR
    maps_per_month = Config.PAPER_MAPS_PER_MONTH_PER_FOLD if maps_per_month is None else int(maps_per_month)
    heat_maps_per_fold = Config.PAPER_HEAT_MAPS_PER_FOLD if heat_maps_per_fold is None else int(heat_maps_per_fold)
    rng_seed = 1729 + sum(ord(ch) for ch in f"{Config.RUN_NAME}_{split_name}")
    rng = np.random.default_rng(rng_seed)
    sample_rows = []
    monthly_stats = {}
    month_seen = {}
    month_buckets = {}
    heat_records = []

    total = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    rank_indices = list(range(rank, total, world_size))
    iterator = tqdm(rank_indices, desc=f"Export hindcast {split_name}", disable=not is_main_process())

    for dataset_idx in iterator:
        batch = dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        pred = predict_direct_ensemble(
            export_models,
            x_t.unsqueeze(0).to(device),
            x_tm1.unsqueeze(0).to(device),
            x_tm2.unsqueeze(0).to(device),
            spatial_c.unsqueeze(0).to(device),
            vec_c.unsqueeze(0).to(device),
            global_fields.unsqueeze(0).to(device),
            batch_mask.unsqueeze(0).to(device),
            device,
        )

        target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
        target_doy = int(dataset.doy_values[target_time_idx])
        if tube_mode:
            pred_tube_np = pred[0, :, :h, :w].detach().cpu().numpy()
            truth_tube_np = y[:, :h, :w].detach().cpu().numpy()
            pred_np = pred_tube_np[tube_center_idx]
            truth_np = truth_tube_np[tube_center_idx]
        else:
            pred_tube_np = None
            truth_tube_np = None
            pred_np = pred[0, 0, :h, :w].detach().cpu().numpy()
            truth_np = y[0, :h, :w].detach().cpu().numpy()
        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
            climo_np = np.asarray(tac_climatology[target_doy], dtype=np.float32)
            if tube_mode:
                for lead_pos, lead in enumerate(tube_leads):
                    lead_doy = int(dataset.doy_values[int(t_idx) + int(lead)])
                    lead_climo = np.asarray(tac_climatology[lead_doy], dtype=np.float32)
                    pred_tube_np[lead_pos] = pred_tube_np[lead_pos] + lead_climo
                    truth_tube_np[lead_pos] = truth_tube_np[lead_pos] + lead_climo
                pred_np = pred_tube_np[tube_center_idx]
                truth_np = truth_tube_np[tube_center_idx]
            else:
                pred_np = pred_np + climo_np
                truth_np = truth_np + climo_np
        persist_np = x_t[0, :h, :w].detach().cpu().numpy()
        _accumulate_tac_stats(
            stats,
            pred_np,
            truth_np,
            persist_np,
            target_doy,
            tac_climatology,
            mask_np,
        )
        if tube_mode and not ddp:
            lead_doys = [
                int(dataset.doy_values[int(t_idx) + int(lead)])
                for lead in tube_leads
            ]
            climo_mean = np.mean(
                [np.asarray(tac_climatology[doy], dtype=np.float32) for doy in lead_doys],
                axis=0,
                dtype=np.float32,
            )
            _accumulate_tac_stats_with_climo(
                weekly7_stats,
                pred_tube_np.mean(axis=0, dtype=np.float32),
                truth_tube_np.mean(axis=0, dtype=np.float32),
                persist_np,
                climo_mean,
                mask_np,
            )
            weekly7_samples += 1
            center_dt = _target_datetime_obj_from_time_values(target_time_idx)
            month = int(center_dt.month)
            if month not in weekly7_monthly_stats:
                weekly7_monthly_stats[month] = _empty_tac_stats(h, w)
            _accumulate_tac_stats_with_climo(
                weekly7_monthly_stats[month],
                pred_tube_np.mean(axis=0, dtype=np.float32),
                truth_tube_np.mean(axis=0, dtype=np.float32),
                persist_np,
                climo_mean,
                mask_np,
            )
        elif weekly_records is not None:
            weekly_records.append({
                "target_time_idx": int(target_time_idx),
                "pred": pred_np.astype(np.float32, copy=True),
                "truth": truth_np.astype(np.float32, copy=True),
                "persist": persist_np.astype(np.float32, copy=True),
            })

        if save_paper_data:
            init_dt = _target_datetime_obj_from_time_values(int(t_idx))
            target_dt = _target_datetime_obj_from_time_values(target_time_idx)
            month = int(target_dt.month)
            climo_np = np.asarray(tac_climatology[target_doy], dtype=np.float32)
            summary = sample_skill_summary(pred_np, truth_np, persist_np, valid_mask)
            sample_rows.append({
                "dataset_idx": int(dataset_idx),
                "t_idx": int(t_idx),
                "target_time_idx": int(target_time_idx),
                "target_doy": int(target_doy),
                "fold": run_fold_id(Config.RUN_NAME),
                "year": int(target_dt.year),
                "month": month,
                "init_date": init_dt.strftime("%Y-%m-%d"),
                "target_date": target_dt.strftime("%Y-%m-%d"),
                **summary,
            })
            if month not in monthly_stats:
                monthly_stats[month] = _empty_tac_stats(h, w)
            _accumulate_tac_stats(
                monthly_stats[month], pred_np, truth_np, persist_np,
                target_doy, tac_climatology, mask_np,
            )

            rec = _paper_map_record(
                dataset_idx, int(t_idx), target_time_idx, target_doy,
                pred_np, truth_np, persist_np, summary, reason=f"month_{month:02d}",
                climo_np=climo_np, pred_tube=pred_tube_np, truth_tube=truth_tube_np,
            )
            month_seen[month] = month_seen.get(month, 0) + 1
            month_buckets.setdefault(month, [])
            _reservoir_update(month_buckets[month], month_seen[month], maps_per_month, rec, rng)

            heat_rec = dict(rec)
            heat_rec["reason"] = "heat_area_top"
            heat_records.append(heat_rec)
            heat_records.sort(key=lambda r: r["summary"]["heat_area_frac"], reverse=True)
            if len(heat_records) > heat_maps_per_fold:
                heat_records.pop()

    if ddp:
        for key, value in list(stats.items()):
            tensor = torch.from_numpy(value).to(device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            stats[key] = tensor.cpu().numpy()

    if not is_main_process():
        return

    if weekly_records is not None:
        weekly7_stats, weekly7_samples, weekly7_monthly_stats = _weekly7_stats_from_daily_records(
            weekly_records,
            np.asarray(dataset.time_values),
            tac_climatology,
            mask_np,
        )
    elif ddp:
        print("  True weekly7 export stats are skipped in DDP export mode; use nproc_per_node=1.")

    model_tac, persistence_tac, _, _ = summarize_tac_stats(stats, mask_np)
    weekly7_model_tac, weekly7_persistence_tac, weekly7_model_corr, weekly7_persistence_corr = (
        summarize_tac_stats(weekly7_stats, mask_np)
    )
    weekly7_model_mse = mse_from_tac_stats(weekly7_stats, mask_np, persistence=False)
    weekly7_persistence_mse = mse_from_tac_stats(weekly7_stats, mask_np, persistence=True)
    years = sorted({
        (datetime(1981, 5, 1) + timedelta(days=float(dataset.time_values[int(t)]))).year
        for t in dataset.indices[:total]
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        **stats,
        mask=mask_np.astype(np.uint8),
        split_name=np.array(split_name),
        run_name=np.array(Config.RUN_NAME),
        cv_split=np.array(cv_split_tag(Config)),
        years=np.array(years, dtype=np.int16),
        n_samples=np.array(total, dtype=np.int32),
        model_tac=np.array(model_tac, dtype=np.float32),
        persistence_tac=np.array(persistence_tac, dtype=np.float32),
        weekly7_n_samples=np.array(weekly7_samples, dtype=np.int32),
        weekly7_model_tac=np.array(weekly7_model_tac, dtype=np.float32),
        weekly7_persistence_tac=np.array(weekly7_persistence_tac, dtype=np.float32),
        weekly7_model_mse=np.array(weekly7_model_mse, dtype=np.float32),
        weekly7_persistence_mse=np.array(weekly7_persistence_mse, dtype=np.float32),
        weekly7_model_corr_map=weekly7_model_corr,
        weekly7_persistence_corr_map=weekly7_persistence_corr,
        **{f"weekly7_{key}": value for key, value in weekly7_stats.items()},
        tube_weekly7_n_samples=np.array(weekly7_samples if tube_mode else 0, dtype=np.int32),
        tube_weekly7_model_tac=np.array(weekly7_model_tac if tube_mode else np.nan, dtype=np.float32),
        tube_weekly7_persistence_tac=np.array(
            weekly7_persistence_tac if tube_mode else np.nan, dtype=np.float32
        ),
        tube_weekly7_model_mse=np.array(weekly7_model_mse if tube_mode else np.nan, dtype=np.float32),
        tube_weekly7_persistence_mse=np.array(
            weekly7_persistence_mse if tube_mode else np.nan, dtype=np.float32
        ),
        prediction_leads=np.array(tube_leads, dtype=np.int16),
        multi_lead_tube=np.array(int(tube_mode), dtype=np.int8),
        **({f"tube_weekly7_{key}": value for key, value in weekly7_stats.items()} if tube_mode else {}),
    )
    print(
        f"  Saved hindcast TAC stats for {split_name} to {output_path}\n"
        f"  {split_name}: n={total}, years={years}, TAC={model_tac:.4f}, "
        f"persistence_TAC={persistence_tac:.4f}\n"
        f"  {split_name} "
        f"{'tube same-init ' + tube_mean_display_label(Config) if tube_mode else 'true weekly7'}: "
        f"n={weekly7_samples}, "
        f"TAC={weekly7_model_tac:.4f}, persistence_TAC={weekly7_persistence_tac:.4f}, "
        f"MSE={weekly7_model_mse:.4f}, persistence_MSE={weekly7_persistence_mse:.4f}"
    )

    if save_paper_data:
        selected_records = []
        for bucket in month_buckets.values():
            selected_records.extend(bucket)
        selected_records.extend(heat_records)
        _save_paper_hindcast_outputs(
            paper_output_dir,
            Config.RUN_NAME,
            split_name,
            sample_rows,
            selected_records,
            monthly_stats,
            mask_np,
            weekly7_monthly_stats=weekly7_monthly_stats,
        )


def compute_ensemble_statistics(ensemble, mask=None):
    if mask is not None:
        mask2 = mask
        if isinstance(mask2, torch.Tensor):
            mask2 = mask2.cpu().numpy()
        mask2 = np.array(mask2)
        ensemble_masked = np.where(mask2[None, :, :] > 0.5, ensemble, np.nan)
        return {
            'mean': np.nanmean(ensemble_masked, axis=0),
            'std': np.nanstd(ensemble_masked, axis=0),
            'p10': np.nanpercentile(ensemble_masked, 10, axis=0),
            'p90': np.nanpercentile(ensemble_masked, 90, axis=0),
        }
    else:
        return {
            'mean': np.mean(ensemble, axis=0),
            'std': np.std(ensemble, axis=0),
            'p10': np.percentile(ensemble, 10, axis=0),
            'p90': np.percentile(ensemble, 90, axis=0),
        }

# ======================================================================================
# VISUALIZATION
# ======================================================================================
def save_validation_plots(model, val_dataset, device, mask, epoch, save_dir, n_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    h, w = Config.IMAGE_SIZE

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    model.eval()

    collected = 0
    if len(val_dataset) == 0:
        print(f"  Saved {collected} validation plots to: {save_dir}")
        return

    n_plot = min(n_samples, len(val_dataset))
    if n_plot == 1:
        plot_indices = np.array([len(val_dataset) // 2], dtype=np.int64)
    else:
        # Spread quick-look plots across the validation period. Using the first
        # N samples repeatedly can make one season/year look like model behavior.
        plot_indices = np.linspace(0, len(val_dataset) - 1, n_plot, dtype=np.int64)

    for dataset_idx in plot_indices:
        batch = val_dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch
        target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
        init_label = "init unknown"
        target_label = "target unknown"
        if hasattr(val_dataset, "time_values"):
            init_dt = datetime(1981, 5, 1) + timedelta(days=float(val_dataset.time_values[int(t_idx)]))
            target_dt = datetime(1981, 5, 1) + timedelta(days=float(val_dataset.time_values[target_time_idx]))
            init_label = init_dt.strftime("%Y-%m-%d")
            target_label = target_dt.strftime("%Y-%m-%d")

        # Single forward pass
        x_t_dev = x_t.unsqueeze(0).to(device)
        x_tm1_dev = x_tm1.unsqueeze(0).to(device)
        x_tm2_dev = x_tm2.unsqueeze(0).to(device)
        spatial_c_dev = spatial_c.unsqueeze(0).to(device)
        vec_c_dev = vec_c.unsqueeze(0).to(device)
        global_fields_dev = global_fields.unsqueeze(0).to(device)
        mask_dev = batch_mask.unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_tensor = predict_direct(model, x_t_dev, x_tm1_dev, x_tm2_dev,
                                         spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device)

        if Config.MULTI_LEAD_TUBE:
            center_idx = center_lead_index(Config)
            pred = pred_tensor[0, center_idx, :h, :w].cpu().numpy()
            truth = y[center_idx, :h, :w].numpy()
        else:
            pred = pred_tensor[0, 0, :h, :w].cpu().numpy()
            truth = y[0, :h, :w].numpy()
        target_doy = int(val_dataset.doy_values[target_time_idx])
        climo = None
        if val_dataset.target_climatology is not None:
            climo = np.asarray(val_dataset.target_climatology[target_doy], dtype=np.float32)
        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES and val_dataset.target_climatology is not None:
            pred = pred + climo
            truth = truth + climo

        plot_pred = pred
        plot_truth = truth
        plot_label = "Z-score"
        plot_cmap = "RdYlBu_r"
        vmin, vmax = -3.0, 3.0
        if climo is not None:
            plot_pred = pred - climo
            plot_truth = truth - climo
            plot_label = "Daily-climatology anomaly (z)"
            plot_cmap = "RdBu_r"
            vmin, vmax = -2.5, 2.5

        pred_display = np.where(mask_np[:h, :w] > 0.5, plot_pred, np.nan)
        truth_display = np.where(mask_np[:h, :w] > 0.5, plot_truth, np.nan)

        valid_mask = mask_np[:h, :w] > 0.5
        p_v = plot_pred[valid_mask]
        t_v = plot_truth[valid_mask]
        r2 = 1 - np.sum((t_v - p_v)**2) / (np.sum((t_v - t_v.mean())**2) + 1e-8)
        corr = 0.0
        if t_v.std() > 1e-6 and p_v.std() > 1e-6:
            corr = np.corrcoef(t_v, p_v)[0, 1]
        mae = np.mean(np.abs(t_v - p_v))

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        im = axes[0].imshow(truth_display, cmap=plot_cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[0].set_title('Ground Truth (t+15)', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[0], label=plot_label)
        im = axes[1].imshow(pred_display, cmap=plot_cmap, vmin=vmin, vmax=vmax, aspect='auto')
        title_mode = 'Tube center-day prediction' if Config.MULTI_LEAD_TUBE else 'Direct 15-day Prediction'
        axes[1].set_title(f'{title_mode}\nMAE={mae:.3f}, anom_r={corr:.3f}, anom_R2={r2:.3f}',
                          fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[1], label=plot_label)
        diff_display = np.where(mask_np[:h, :w] > 0.5, plot_pred - plot_truth, np.nan)
        im = axes[2].imshow(diff_display, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
        axes[2].set_title('Prediction - Truth', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[2], label='Difference (Z-score)')
        fig.suptitle(
            f'Epoch {epoch} - Val index {int(dataset_idx)} | Init {init_label} -> Target {target_label}',
            fontsize=15,
            fontweight='bold',
        )
        plt.tight_layout()
        fname = os.path.join(
            save_dir,
            f'val_epoch{epoch:04d}_idx{int(dataset_idx):05d}_{target_label}.png'
        )
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        collected += 1
    print(f"  Saved {collected} validation plots to: {save_dir}")


def save_mask_plot(mask, out_dir, fname="conus_mask.png"):
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)
    if m.ndim == 3: m = m[0]
    land_frac = float(np.mean(m))
    plt.figure(figsize=(12, 5))
    plt.imshow(m, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    plt.title(f"Land/Ocean Mask (land fraction = {land_frac:.3f})")
    plt.colorbar(label="1=land, 0=ocean")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()


# ======================================================================================
# DATA CACHE
# ======================================================================================
def prepare_shared_data(config, rank, world_size, ddp):
    cache_dir = os.path.join(config.OUTPUT_DIR, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    target_meta_path = os.path.join(cache_dir, "_target_cache_meta.npz")

    TOPO_PATH = "/blue/nessie/mostafarezaali/Teleconnection/CONUS_topography_ETOPO2022_60s_on_model_grid.nc"

    paths = {
        "heat_index":   os.path.join(cache_dir, "heat_index.npy"),
        "geopotential": os.path.join(cache_dir, "geopotential.npy"),
        "soil_moisture":os.path.join(cache_dir, "soil_moisture.npy"),
        "slp":          os.path.join(cache_dir, "slp.npy"),
        "cond_train":   os.path.join(cache_dir, "cond_train.npy"),
        "topography":   os.path.join(cache_dir, "topography.npy"),
        "temperature_2m":        os.path.join(cache_dir, "temperature_2m.npy"),
        "specific_humidity_850":os.path.join(cache_dir, "specific_humidity_850.npy"),
        "temperature_850":       os.path.join(cache_dir, "temperature_850.npy"),
        "u_wind_850":            os.path.join(cache_dir, "u_wind_850.npy"),
        "v_wind_850":            os.path.join(cache_dir, "v_wind_850.npy"),
        "geopotential_300":      os.path.join(cache_dir, "geopotential_300.npy"),
        "time_values":           os.path.join(cache_dir, "time_values.npy"),
    }

    global_cache_dir = os.path.join(cache_dir, "global")
    os.makedirs(global_cache_dir, exist_ok=True)
    global_meta_path = os.path.join(global_cache_dir, "_global_cache_meta.npz")
    global_paths = {}
    for var_name in config.GLOBAL_VARIABLES:
        global_paths[var_name] = os.path.join(global_cache_dir, f"{var_name}.npy")

    if is_main_process():
        cache_ok = all(os.path.exists(p) for p in paths.values())
        if cache_ok:
            try:
                nc_mtime = max(os.path.getmtime(config.TRAINING_DATA_PATH), os.path.getmtime(TOPO_PATH))
                cache_mtime = min(os.path.getmtime(p) for p in paths.values())
                if nc_mtime > cache_mtime:
                    cache_ok = False
                else:
                    with NetCDFDataset(config.TRAINING_DATA_PATH, "r") as nc_probe:
                        source_target_name, _ = get_target_variable(nc_probe, config)
                    meta = np.load(target_meta_path, allow_pickle=True)
                    cached_target_name = str(meta["target_variable"].item())
                    if cached_target_name != source_target_name:
                        cache_ok = False
            except Exception:
                cache_ok = False

        if not cache_ok:
            with NetCDFDataset(config.TRAINING_DATA_PATH, "r") as nc:
                target_name, target_var = get_target_variable(nc, config)
                print(f"Rank 0: caching target variable {target_name} with shape {target_var.shape}")
                hi = target_to_hw_time(target_var[:], target_name)
                np.save(paths["heat_index"], hi)
                np.save(paths["geopotential"], np.array(nc.variables["geopotential"][:], dtype=np.float32))
                np.save(paths["soil_moisture"], np.array(nc.variables["soil_moisture"][:], dtype=np.float32))
                np.save(paths["slp"], np.array(nc.variables["sea_level_pressure"][:], dtype=np.float32))
                np.save(paths["cond_train"], np.array(nc.variables["CondTrain"][:], dtype=np.float32))
                np.save(paths["temperature_2m"], np.array(nc.variables["temperature_2m"][:], dtype=np.float32))
                np.save(paths["specific_humidity_850"], np.array(nc.variables["specific_humidity_850"][:], dtype=np.float32))
                np.save(paths["temperature_850"], np.array(nc.variables["temperature_850"][:], dtype=np.float32))
                np.save(paths["u_wind_850"], np.array(nc.variables["u_wind_850"][:], dtype=np.float32))
                np.save(paths["v_wind_850"], np.array(nc.variables["v_wind_850"][:], dtype=np.float32))
                np.save(paths["geopotential_300"], np.array(nc.variables["geopotential_300"][:], dtype=np.float32))
                np.save(paths["time_values"], np.array(nc.variables["time"][:], dtype=np.float64))
                np.savez(target_meta_path, target_variable=np.array(target_name, dtype=object))

            with NetCDFDataset(TOPO_PATH, "r") as nc_topo:
                topo = np.array(nc_topo.variables["elevation"][:], dtype=np.float32)
                topo = np.flipud(topo)
            np.save(paths["topography"], topo)
            print(f"Rank 0: wrote CONUS data cache to {cache_dir}")
        else:
            print(f"Rank 0: using existing CONUS data cache in {cache_dir}")

        global_cache_ok = all(os.path.exists(p) for p in global_paths.values())
        if global_cache_ok:
            try:
                meta = np.load(global_meta_path, allow_pickle=True)
                cached_vars = [str(v) for v in meta["variables"].tolist()]
                cached_path = str(meta["global_data_path"].item())
                if cached_vars != list(config.GLOBAL_VARIABLES) or cached_path != config.GLOBAL_DATA_PATH:
                    global_cache_ok = False
            except Exception:
                global_cache_ok = False
        if not global_cache_ok:
            print(f"Rank 0: caching {global_base_channel_count(config)} base global teleconnection fields...")
            print(f"  Source: {config.GLOBAL_DATA_PATH}")
            with NetCDFDataset(config.GLOBAL_DATA_PATH, "r") as nc_g:
                missing = [name for name in config.GLOBAL_VARIABLES if name not in nc_g.variables]
                if missing:
                    raise KeyError(
                        f"Missing {len(missing)} GLOBAL_VARIABLES in {config.GLOBAL_DATA_PATH}: {missing[:20]}"
                    )
                for var_name in config.GLOBAL_VARIABLES:
                    data = np.array(nc_g.variables[var_name][:], dtype=np.float32)
                    if data.ndim == 4:
                        data = data[:, 0, :, :]
                    data = np.transpose(data, (1, 2, 0))
                    nan_count = np.isnan(data).sum()
                    if nan_count > 0:
                        data = np.nan_to_num(data, nan=0.0)
                    np.save(global_paths[var_name], data)
                    del data
            np.savez(
                global_meta_path,
                variables=np.array(config.GLOBAL_VARIABLES, dtype=object),
                global_data_path=np.array(config.GLOBAL_DATA_PATH, dtype=object),
            )
            print(f"Rank 0: wrote global data cache")
        else:
            print(
                f"Rank 0: using existing global data cache "
                f"({global_base_channel_count(config)} base fields)"
            )

    if ddp:
        dist.barrier()

    shm_dir = "/dev/shm/cfm_cache"

    if not ddp or dist.get_rank() == 0:
        os.makedirs(shm_dir, exist_ok=True)
        for key, src_path in paths.items():
            dst = os.path.join(shm_dir, os.path.basename(src_path))
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        shm_global = os.path.join(shm_dir, "global")
        os.makedirs(shm_global, exist_ok=True)
        shm_global_meta = os.path.join(shm_global, os.path.basename(global_meta_path))
        shm_global_ok = all(
            os.path.exists(os.path.join(shm_global, os.path.basename(p)))
            for p in global_paths.values()
        )
        if shm_global_ok:
            try:
                meta = np.load(shm_global_meta, allow_pickle=True)
                cached_vars = [str(v) for v in meta["variables"].tolist()]
                cached_path = str(meta["global_data_path"].item())
                if cached_vars != list(config.GLOBAL_VARIABLES) or cached_path != config.GLOBAL_DATA_PATH:
                    shm_global_ok = False
            except Exception:
                shm_global_ok = False
        for var_name, src_path in global_paths.items():
            dst = os.path.join(shm_global, os.path.basename(src_path))
            if not shm_global_ok or not os.path.exists(dst):
                shutil.copy2(src_path, dst)
        if not shm_global_ok or not os.path.exists(shm_global_meta):
            shutil.copy2(global_meta_path, shm_global_meta)
        print(f"Rank 0: copied data cache to {shm_dir}")

    if ddp:
        dist.barrier()

    shm_paths = {k: os.path.join(shm_dir, os.path.basename(v)) for k, v in paths.items()}

    shared = {
        k: np.load(shm_paths[k], mmap_mode="r")
        for k in paths
    }

    from collections import OrderedDict
    global_data = OrderedDict()
    for var_name in config.GLOBAL_VARIABLES:
        gpath = os.path.join(shm_dir, "global", f"{var_name}.npy")
        global_data[var_name] = np.load(gpath, mmap_mode="r")
    if len(global_data) != global_base_channel_count(config):
        raise RuntimeError(
            f"Loaded {len(global_data)} base global fields, expected "
            f"{global_base_channel_count(config)}."
        )
    shared['global_data'] = global_data

    return shared


torch.backends.cudnn.benchmark = True

# ======================================================================================
# TRAINING LOOP
# ======================================================================================
def train_model(rank=0, world_size=1, checkpoint_path=None):
    ddp = world_size > 1

    spike_logger = setup_spike_logger(Config.OUTPUT_DIR)

    if ddp:
        setup_ddp(rank, world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    conus_mask = load_conus_mask(Config)
    if is_main_process():
        save_mask_plot(conus_mask, Config.PLOTS_DIR)
    conus_mask = conus_mask.to(device)

    fm = FlowMatching().to(device)

    if is_main_process():
        print("\n" + "=" * 80)
        print("DIRECT 15-DAY PREDICTION + ICOSAHEDRAL MESH GNN")
        print(f"World size: {world_size}")
        print(f"Mesh level: {Config.MESH_REFINEMENT_LEVEL}, Rounds: {Config.MESH_PROCESSOR_ROUNDS}")
        if Config.MULTI_LEAD_TUBE:
            print(f"Prediction tube leads: {prediction_leads(Config)} days DIRECT (center t+{Config.LEAD_TIME})")
        else:
            print(f"Lead time: {Config.LEAD_TIME} days DIRECT (no rollout)")
        print(
            f"Global fields: {global_base_channel_count(Config)} variables x "
            f"{len(Config.GLOBAL_LAG_DAYS)} lags"
            f"{' x ' + str(len(Config.GLOBAL_COMPONENT_NAMES)) + ' components' if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else ''} "
            f"= {Config.NUM_GLOBAL_CHANNELS} channels "
            f"from {Config.GLOBAL_DATA_PATH}"
        )
        if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL:
            print(
                f"Global decomposition: trailing {Config.GLOBAL_LOWPASS_WINDOW_DAYS}-day "
                f"low-frequency field + current residual"
            )
        print(
            f"Local lag fields: {Config.LOCAL_LAG_VARIABLES} at days "
            f"{Config.LOCAL_LAG_DAYS} = {Config.NUM_LOCAL_LAG_CHANNELS} channels"
        )
        print(
            "Cross-validation: leave-k-years-out "
            f"({cv_split_tag(Config)}, run_name={Config.RUN_NAME or 'default'})"
        )
        print("=" * 80 + "\n")

    # Data
    shared_data = prepare_shared_data(Config, rank, world_size, ddp)
    n_timesteps = shared_data['heat_index'].shape[-1]

    # Read time values for season boundary detection
    time_values = np.array(shared_data['time_values'])

    runs = detect_continuous_runs(time_values)
    if is_main_process():
        print(f"Detected {len(runs)} continuous runs (expected ~43 MJJAS seasons)")

    active_max_lead = max_prediction_lead(Config)
    if active_max_lead != max(prediction_leads(Config)):
        raise RuntimeError(
            f"Valid-index max lead {active_max_lead} does not match prediction leads "
            f"{prediction_leads(Config)}."
        )
    # Build valid indices: train/val/test must keep every requested target lead inside a continuous season.
    all_valid = build_valid_indices(
        runs,
        lead_time=active_max_lead,
        min_history=required_input_history(Config),
    )

    if is_main_process():
        print(f"Total valid indices (max_lead={active_max_lead}): {len(all_valid)}")

    # Cross-validation split: leave-k-years-out
    train_indices, val_indices, test_indices, train_years, val_years, test_years = \
        build_crossval_split(all_valid, time_values)

    if is_main_process():
        print(f"\nLeave-k-Years-Out Cross-Validation Split:")
        print(f"  Train: {len(train_indices)} samples ({len(train_years)} years)")
        print(f"  Val:   {len(val_indices)} samples ({len(val_years)} years: {sorted(val_years)})")
        print(f"  Test:  {len(test_indices)} samples ({len(test_years)} years: {sorted(test_years)})")
        print()

    # Normalization stats are fold-specific so k-fold hindcast runs cannot leak.
    stats_path = get_norm_stats_path(Config)

    if is_main_process():
        stats_ok = os.path.exists(stats_path)
        if stats_ok:
            try:
                stats_probe = np.load(stats_path)
                stats_ok = norm_stats_match_npz(stats_probe, Config)
                stats_probe.close()
                if not stats_ok:
                    print(
                        "  Existing normalization stats do not match the active "
                        f"{Config.NUM_GLOBAL_CHANNELS}-channel lagged global stack "
                        "or local-lag/anomaly-target setting; recomputing."
                    )
            except Exception:
                stats_ok = False
        if not stats_ok:
            print("  Rank 0: Calculating Z-Score statistics (cross-val train set)...")
            tmp_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                         shared_data=shared_data)
            norm_stats = get_normalization_stats(tmp_dataset)
            save_dict = {
                'hi_mean': float(norm_stats['hi_mean']), 'hi_std': float(norm_stats['hi_std']),
                'stats_mean': norm_stats['stats_mean'].numpy(), 'stats_std': norm_stats['stats_std'].numpy(),
                'cond_mean': norm_stats['cond_mean'].numpy(), 'cond_std': norm_stats['cond_std'].numpy(),
                'topo_mean': float(norm_stats['topo_mean']), 'topo_std': float(norm_stats['topo_std']),
                'toa_mean': float(norm_stats['toa_mean']), 'toa_std': float(norm_stats['toa_std']),
                'lead_time': int(Config.LEAD_TIME),
                'multi_lead_tube': int(Config.MULTI_LEAD_TUBE),
                'prediction_leads': np.array(prediction_leads(Config), dtype=np.int16),
                'target_half_window': 0,
                'global_lag_days': np.array(Config.GLOBAL_LAG_DAYS, dtype=np.int16),
                'global_decompose_low_residual': int(Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL),
                'global_lowpass_window_days': (
                    int(Config.GLOBAL_LOWPASS_WINDOW_DAYS)
                    if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 0
                ),
                'local_lag_days': np.array(Config.LOCAL_LAG_DAYS, dtype=np.int16),
                'local_lag_variables': np.array(Config.LOCAL_LAG_VARIABLES),
                'train_on_climatology_anomalies': int(Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES),
                'cv_stride': int(Config.CV_STRIDE),
                'cv_val_offsets': np.array(Config.CV_VAL_OFFSETS, dtype=np.int16),
                'cv_test_offsets': np.array(Config.CV_TEST_OFFSETS, dtype=np.int16),
            }
            if norm_stats['global_mean'] is not None:
                save_dict['global_mean'] = norm_stats['global_mean'].numpy()
                save_dict['global_std'] = norm_stats['global_std'].numpy()
            np.savez(stats_path, **save_dict)
            del tmp_dataset
            gc.collect()

    if ddp:
        dist.barrier()

    s = np.load(stats_path)
    norm_stats = {
        'hi_mean': torch.tensor(float(s['hi_mean'])), 'hi_std': torch.tensor(float(s['hi_std'])),
        'stats_mean': torch.from_numpy(s['stats_mean']), 'stats_std': torch.from_numpy(s['stats_std']),
        'cond_mean': torch.from_numpy(s['cond_mean']), 'cond_std': torch.from_numpy(s['cond_std']),
        'topo_mean': torch.tensor(float(s['topo_mean'])), 'topo_std': torch.tensor(float(s['topo_std'])),
        'toa_mean': torch.tensor(float(s['toa_mean'])), 'toa_std': torch.tensor(float(s['toa_std'])),
    }
    if 'global_mean' in s:
        norm_stats['global_mean'] = torch.from_numpy(s['global_mean'])
        norm_stats['global_std'] = torch.from_numpy(s['global_std'])
    else:
        norm_stats['global_mean'] = None
        norm_stats['global_std'] = None

    tac_climatology = load_or_build_train_climatology(
        shared_data, train_indices, norm_stats, Config, ddp=ddp
    )
    exceedance_q95 = None
    exceedance_base_rate = None
    if Config.ENABLE_EXCEEDANCE_HEAD:
        exceedance_q95, exceedance_base_rate = load_or_build_exceedance_stats(
            shared_data, train_indices, norm_stats, Config, ddp=ddp
        )

    train_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                   normalization_stats=norm_stats, shared_data=shared_data,
                                   target_climatology=tac_climatology)
    val_dataset = ClimateDataset(Config, mode="val", val_indices=val_indices,
                                 normalization_stats=norm_stats, shared_data=shared_data,
                                 target_climatology=tac_climatology)
    test_dataset = ClimateDataset(Config, mode="test", test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=shared_data,
                                  target_climatology=tac_climatology)

    if is_main_process() and len(train_dataset) > 0:
        sample_t = int(train_dataset.indices[0])
        print(
            "Direct lead check: model inputs use t, t-1, t-2; "
            f"local lag fields use lags {Config.LOCAL_LAG_DAYS}; "
            f"global fields use lags {Config.GLOBAL_LAG_DAYS}; "
            f"global lowpass needs {Config.GLOBAL_LOWPASS_WINDOW_DAYS if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 1} days; "
            f"target uses leads {prediction_leads(Config)} "
            f"(first train sample t={sample_t}, center_target_t={sample_t + Config.LEAD_TIME})."
        )

    if is_main_process() and checkpoint_path is None:
        baseline_samples = min(Config.NUM_VALIDATION_SAMPLES, len(val_dataset))
        print(f"Computing persistence baseline (day-0 as day-15 forecast, n={baseline_samples})...")
        persistence_metrics = compute_persistence_baseline(val_dataset, conus_mask, n_samples=baseline_samples)
        print(f"  Persistence baseline: R2={persistence_metrics['r2']:.4f}, "
              f"corr={persistence_metrics['correlation']:.4f}, "
              f"CRPS={persistence_metrics['crps']:.4f}")

    # Build mesh + model
    mesh = build_mesh_once(Config, conus_mask, device, ddp=ddp)

    model = build_meshflow_model(Config, mesh, device)
    if Config.ENABLE_EXCEEDANCE_HEAD:
        model.exceedance_region_masks = build_exceedance_region_mask_tensor(Config, conus_mask).to(device)

    if is_main_process():
        count_parameters(model)
        print(f"  Mode: {'DETERMINISTIC (direct 15-day)' if Config.DETERMINISTIC else 'PROBABILISTIC (CFM)'}")

    # Sampler + DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=0, pin_memory=False,
        persistent_workers=False, prefetch_factor=None,
    )

    effective_batch_size = Config.BATCH_SIZE * world_size
    if is_main_process():
        print(f"Per-GPU batch: {Config.BATCH_SIZE}, Effective batch: {effective_batch_size}")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=Config.WARMUP_EPOCHS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3000, eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[Config.WARMUP_EPOCHS])

    start_epoch = 0
    best_ssim = 0.0
    best_r2 = -999.0
    best_tac = -999.0
    early_stop_best = -float("inf")
    early_stop_failures = 0

    # Checkpoint loading
    if checkpoint_path is not None:
        if is_main_process():
            print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if is_main_process() and missing:
            print(f"  New parameters (not in checkpoint): {len(missing)} keys")
        if Config.MULTI_LEAD_TUBE and any(
            key.startswith(("lead_", "tube_")) for key in missing
        ):
            raise RuntimeError(
                "This checkpoint does not contain the tube temporal head. "
                "Start tube experiments from scratch or use a tube checkpoint."
            )
        start_epoch = checkpoint.get("epoch", 0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        best_r2 = checkpoint.get("best_r2", best_r2)
        best_tac = checkpoint.get("best_tac", best_tac)
        if checkpoint.get("early_stop_metric", Config.EARLY_STOP_METRIC) == Config.EARLY_STOP_METRIC:
            early_stop_best = checkpoint.get("early_stop_best", early_stop_best)
            early_stop_failures = checkpoint.get("early_stop_failures", early_stop_failures)

    # DDP wrapping
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ema = EMA(model.module if ddp else model, decay=0.999)
    ema.ema.set_mesh(mesh)

    if checkpoint_path is not None and "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        ema_cleaned = {k.replace("module.", ""): v for k, v in ema_state.items()}
        ema.ema.load_state_dict(ema_cleaned, strict=False)
        ema.ema.set_mesh(mesh)
        if is_main_process():
            print("  Restored EMA state from checkpoint")

    if start_epoch > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()

    # Training loop
    for epoch in range(start_epoch, Config.MAX_EPOCHS):
        stop_training = False
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_extreme_loss = 0.0
        epoch_base_mse_loss = 0.0
        epoch_anom_corr_loss = 0.0
        epoch_gradient_loss = 0.0
        epoch_exceedance_bce_loss = 0.0
        epoch_exceedance_count_loss = 0.0
        epoch_sigma_min = float("inf")
        n_good_batches = 0
        consecutive_skips = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process(),
                    mininterval=10.0)

        for batch_idx, batch in enumerate(pbar):
            (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_indices, mask) = batch

            y = y.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            x_tm1 = x_tm1.to(device, non_blocking=True)
            x_tm2 = x_tm2.to(device, non_blocking=True)
            spatial_c = spatial_c.to(device, non_blocking=True)
            vec_c = vec_c.to(device, non_blocking=True)
            global_fields = global_fields.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if epoch == 0 and batch_idx == 0:
                deterministic_channels = (
                    x_t.shape[1] + x_tm1.shape[1] + x_tm2.shape[1] + spatial_c.shape[1]
                )
                if Config.DETERMINISTIC and deterministic_channels != Config.NUM_SPATIAL_CONDITIONS:
                    raise RuntimeError(
                        "Deterministic input channel mismatch: "
                        f"got {deterministic_channels}, expected {Config.NUM_SPATIAL_CONDITIONS}."
                    )
                if global_fields.shape[1] != Config.NUM_GLOBAL_CHANNELS:
                    raise RuntimeError(
                        f"Global channel mismatch: got {global_fields.shape[1]}, "
                        f"expected {Config.NUM_GLOBAL_CHANNELS}."
                    )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                exceedance_thresholds = (
                    batch_exceedance_thresholds(t_indices, train_dataset, exceedance_q95, device)
                    if Config.ENABLE_EXCEEDANCE_HEAD else None
                )
                loss, components = compute_loss(
                    model, fm, y, x_t, x_tm1, x_tm2,
                    spatial_c, vec_c, global_fields, mask,
                    deterministic=Config.DETERMINISTIC,
                    exceedance_thresholds=exceedance_thresholds)

                base_mse_val = 0.0
                anom_corr_val = 0.0
                gradient_loss_val = 0.0
                exceedance_bce_val = 0.0
                exceedance_count_val = 0.0
                grad_loss_for_loss = components.get('gradient_loss', None)
                if grad_loss_for_loss is not None:
                    gradient_loss_val = grad_loss_for_loss.item()
                if components.get('exceedance_bce_loss', None) is not None:
                    exceedance_bce_val = components['exceedance_bce_loss'].item()
                if components.get('exceedance_count_loss', None) is not None:
                    exceedance_count_val = components['exceedance_count_loss'].item()
                if components.get('sigma_min', None) is not None:
                    try:
                        epoch_sigma_min = min(epoch_sigma_min, float(components['sigma_min'].item()))
                    except Exception:
                        pass
                if 'recon_mse' in components:
                    base_mse_val = components['recon_mse'].item()
                if Config.MULTI_LEAD_TUBE and 'tube_daily_mse' in components:
                    base_mse_val = components['tube_daily_mse'].item()
                if Config.DISTRIBUTIONAL_HEAD and 'tube_daily_crps' in components:
                    base_mse_val = components['tube_daily_crps'].item()
                if Config.USE_ANOMALY_CORR_LOSS:
                    pred_for_corr = components.get('pred', None)
                    climo = batch_target_climatology(t_indices, train_dataset, device)
                    if pred_for_corr is None or climo is None:
                        raise RuntimeError(
                            "Anomaly-correlation loss requires deterministic predictions "
                            "and train-year-only target climatology."
                        )
                    with torch.amp.autocast('cuda', enabled=False):
                        base_mse = masked_mse_loss(pred_for_corr.float(), y.float(), mask.float())
                        anom_loss = anomaly_corr_loss(
                            pred_for_corr.float(), y.float(), climo, mask.float()
                        )
                        base_train_loss = base_mse
                        if Config.GRADIENT_LOSS_WEIGHT > 0.0 and grad_loss_for_loss is not None:
                            g = float(Config.GRADIENT_LOSS_WEIGHT)
                            base_train_loss = (1.0 - g) * base_mse + g * grad_loss_for_loss.float()
                        w = float(Config.ANOMALY_CORR_LOSS_WEIGHT)
                        loss = (1.0 - w) * base_train_loss + w * anom_loss
                    base_mse_val = base_mse.item()
                    anom_corr_val = anom_loss.item()

                # Extreme loss fine-tuning (Lopez-Gomez et al., 2023)
                ext_loss_val = 0.0
                if Config.USE_EXTREME_LOSS:
                    # Use prediction already computed by compute_loss (no second forward pass)
                    pred_for_ext = components.get('pred', None)

                    if pred_for_ext is not None:
                        ext_loss = exponential_extreme_loss(
                            pred_for_ext, y, mask,
                            a=Config.EXTREME_LOSS_HEAT_BIAS,
                            b=Config.EXTREME_LOSS_COLD_BIAS,
                        )
                        w = Config.EXTREME_LOSS_WEIGHT
                        loss = (1 - w) * loss + w * ext_loss
                        ext_loss_val = ext_loss.item()

            is_fatal = torch.isnan(loss) or torch.isinf(loss)
            if ddp:
                fatal_flag = torch.tensor(float(is_fatal), device=device)
                dist.all_reduce(fatal_flag, op=dist.ReduceOp.MAX)
                is_fatal = fatal_flag.item() > 0.5

            if is_fatal:
                consecutive_skips += 1
                if is_main_process() and consecutive_skips <= 3:
                    spike_logger.warning(f"NaN/Inf at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # Gradient diagnostic (first batch of first epoch only)
            if epoch == 0 and batch_idx == 0 and is_main_process():
                diag_path = os.path.join(Config.OUTPUT_DIR, "gradient_diagnostic.txt")
                with open(diag_path, "w") as f:
                    f.write("========================================================\n")
                    f.write("Gradient Diagnostic - Epoch 0, Batch 0\n")
                    f.write(f"Direct 15-day prediction mode\n")
                    f.write("========================================================\n")
                    raw = model.module if ddp else model
                    dead_layers = 0
                    for name, p in raw.named_parameters():
                        gn = p.grad.norm().item() if p.grad is not None else 0.0
                        pn = p.norm().item()
                        f.write(f"{name:60s} grad={gn:10.6f}  param={pn:10.4f}\n")
                        if gn == 0.0:
                            dead_layers += 1
                    f.write(f"\nSummary:\n")
                    f.write(f"Total dead layers (grad=0.0): {dead_layers}\n")
                print(f"  Saved gradient diagnostic to: {diag_path}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            ema.update(model.module if ddp else model)
            consecutive_skips = 0

            epoch_loss += loss.item()
            epoch_base_mse_loss += base_mse_val
            epoch_anom_corr_loss += anom_corr_val
            epoch_gradient_loss += gradient_loss_val
            epoch_exceedance_bce_loss += exceedance_bce_val
            epoch_exceedance_count_loss += exceedance_count_val
            epoch_extreme_loss += ext_loss_val
            n_good_batches += 1

            if is_main_process():
                postfix = {"loss": f"{loss.item():.4f}"}
                if Config.USE_ANOMALY_CORR_LOSS:
                    postfix["mse"] = f"{base_mse_val:.4f}"
                    postfix["ac"] = f"{anom_corr_val:.4f}"
                if Config.GRADIENT_LOSS_WEIGHT > 0.0:
                    postfix["grad"] = f"{gradient_loss_val:.4f}"
                if Config.ENABLE_EXCEEDANCE_HEAD:
                    postfix["bce"] = f"{exceedance_bce_val:.4f}"
                if Config.DISTRIBUTIONAL_HEAD and np.isfinite(epoch_sigma_min):
                    postfix["sigmin"] = f"{epoch_sigma_min:.3f}"
                if Config.USE_EXTREME_LOSS:
                    postfix["ext"] = f"{ext_loss_val:.4f}"
                pbar.set_postfix(postfix)

            if Config.MAX_TRAIN_BATCHES is not None and (batch_idx + 1) >= Config.MAX_TRAIN_BATCHES:
                if is_main_process():
                    print(f"  Stopping epoch early after {Config.MAX_TRAIN_BATCHES} batches.")
                break

        if n_good_batches == 0 and is_main_process():
            print(f"\n  *** ALERT: Epoch {epoch+1} - ALL batches had NaN/Inf loss! ***")

        if Config.DISTRIBUTIONAL_HEAD and ddp:
            sigma_tensor = torch.tensor([epoch_sigma_min], device=device, dtype=torch.float32)
            dist.all_reduce(sigma_tensor, op=dist.ReduceOp.MIN)
            epoch_sigma_min = float(sigma_tensor.item())

        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
            torch.cuda.empty_cache()
            gc.collect()

            if ddp:
                for param in ema.ema.parameters():
                    dist.broadcast(param.data, src=0)

            val_mse, val_rmse, val_ssim, improved_metrics = calculate_validation_metrics_cfm(
                ema.ema, val_dataset, device, conus_mask,
                n_samples=Config.NUM_VALIDATION_SAMPLES,
                rank=rank, world_size=world_size, ddp=ddp,
                tac_climatology=(
                    tac_climatology
                    if (is_main_process() or Config.MULTI_LEAD_TUBE)
                    else None
                ),
                exceedance_q95=exceedance_q95 if Config.ENABLE_EXCEEDANCE_HEAD else None,
                exceedance_base_rate=exceedance_base_rate if Config.ENABLE_EXCEEDANCE_HEAD else None,
            )

            current_r2 = improved_metrics.get('r2', -999.0) if improved_metrics else -999.0

            if is_main_process():
                val_plot_dir = os.path.join(Config.PLOTS_DIR, "validation")
                save_validation_plots(ema.ema, val_dataset, device, conus_mask, epoch + 1, val_plot_dir, n_samples=10)
                avg_loss = epoch_loss / max(n_good_batches, 1)
                torch.cuda.empty_cache()
                gc.collect()

                print(f"\n  === EPOCH {epoch + 1} METRICS ===")
                print(f"  Training Loss:     {avg_loss:.6f}")
                avg_base_mse = epoch_base_mse_loss / max(n_good_batches, 1)
                avg_anom_corr = epoch_anom_corr_loss / max(n_good_batches, 1)
                avg_gradient = epoch_gradient_loss / max(n_good_batches, 1)
                avg_exceedance_bce = epoch_exceedance_bce_loss / max(n_good_batches, 1)
                avg_exceedance_count = epoch_exceedance_count_loss / max(n_good_batches, 1)
                if Config.USE_ANOMALY_CORR_LOSS:
                    print(f"  Train Base MSE:    {avg_base_mse:.6f}")
                    print(f"  Anom Corr Loss:    {avg_anom_corr:.6f}")
                elif Config.MULTI_LEAD_TUBE:
                    train_metric_name = "CRPS" if Config.DISTRIBUTIONAL_HEAD and Config.CRPS_LOSS else "MSE"
                    print(f"  Train Daily {train_metric_name}:   {avg_base_mse:.6f}")
                else:
                    train_metric_name = "CRPS" if Config.DISTRIBUTIONAL_HEAD and Config.CRPS_LOSS else "MSE"
                    print(f"  Train Recon {train_metric_name}:   {avg_base_mse:.6f}")
                    avg_anom_corr = ""
                if Config.DISTRIBUTIONAL_HEAD:
                    print(f"  Min Sigma:         {epoch_sigma_min:.6f}")
                if Config.GRADIENT_LOSS_WEIGHT > 0.0:
                    print(f"  Gradient Loss:     {avg_gradient:.6f}")
                else:
                    avg_gradient = ""
                if Config.ENABLE_EXCEEDANCE_HEAD:
                    print(f"  Exceedance BCE:    {avg_exceedance_bce:.6f}")
                    print(f"  Exceed Count Loss: {avg_exceedance_count:.6f}")
                else:
                    avg_exceedance_bce = ""
                    avg_exceedance_count = ""
                if Config.USE_EXTREME_LOSS:
                    avg_ext = epoch_extreme_loss / max(n_good_batches, 1)
                    print(f"  Extreme Loss:      {avg_ext:.6f}")
                else:
                    avg_ext = ""
                print(f"  Validation MSE:    {val_mse:.6f}")
                print(f"  Validation SSIM:   {val_ssim:.4f}")
                print(f"  Variance Ratio:    {improved_metrics['variance_ratio']:.4f}")
                print(f"  Gradient Ratio:    {improved_metrics['gradient_ratio']:.4f}")
                print(f"  Extreme Bias:      {improved_metrics['extreme_bias']:.4f}")
                print(f"  Correlation:       {improved_metrics['correlation']:.4f}")
                print(f"  R2:                {improved_metrics['r2']:.4f}")
                print(f"  Persistence R2:    {improved_metrics['persistence_r2']:.4f}")
                print(f"  Zero Baseline R2:  {improved_metrics['zero_r2']:.4f}")
                print(f"  MSE Skill vs Pers: {improved_metrics['mse_skill_vs_persistence']:.4f}")
                print(f"  Spatial Anom R2:   {improved_metrics['spatial_anom_r2']:.4f}")
                print(f"  Pers Spatial R2:   {improved_metrics['persistence_spatial_anom_r2']:.4f}")
                print(f"  TAC:               {improved_metrics['tac']:.4f}")
                print(f"  Persistence TAC:   {improved_metrics['persistence_tac']:.4f}")
                if Config.MULTI_LEAD_TUBE:
                    tube_days = len(prediction_leads(Config))
                    print(f"  Tube W{tube_days} TAC:       {improved_metrics['tube_weekly7_tac']:.4f}")
                    print(f"  Tube W{tube_days} Pers TAC:  {improved_metrics['tube_weekly7_persistence_tac']:.4f}")
                    print(f"  Tube W{tube_days} samples:   {int(improved_metrics['tube_weekly7_n_samples'])}")
                else:
                    print(f"  Weekly7 TAC:       {improved_metrics['weekly7_tac']:.4f}")
                    print(f"  Weekly7 Pers TAC:  {improved_metrics['weekly7_persistence_tac']:.4f}")
                    print(f"  Weekly7 samples:   {int(improved_metrics['weekly7_n_samples'])}")
                if Config.ENABLE_EXCEEDANCE_HEAD:
                    print(f"  Exceedance BSS:    {improved_metrics['exceedance_bss']:+.4f}")
                    print(f"  Exceedance Brier:  {improved_metrics['exceedance_brier']:.5f}")
                    print(f"  Exceedance Rate:   truth={improved_metrics['exceedance_base_rate']:.4f}, pred={improved_metrics['exceedance_pred_rate']:.4f}")
                print(f"  MAE:               {improved_metrics['mae']:.4f}")
                print(f"  CRPS:              {improved_metrics['crps']:.4f}")
                if Config.DISTRIBUTIONAL_HEAD:
                    print(f"  Val Min Sigma:     {improved_metrics['sigma_min']:.6f}")
                print(f"  LR:                {scheduler.get_last_lr()[0]:.2e}")

                val_r2 = improved_metrics["r2"]
                val_tac = improved_metrics.get("tac", float("nan"))
                monitor_score = early_stop_score(
                    Config.EARLY_STOP_METRIC, improved_metrics, val_mse, val_ssim
                )
                monitor_value = early_stop_display_value(Config.EARLY_STOP_METRIC, monitor_score)
                best_monitor_value = early_stop_display_value(Config.EARLY_STOP_METRIC, early_stop_best)
                monitor_valid = np.isfinite(monitor_score)
                monitor_improved = (
                    monitor_valid
                    and monitor_score > early_stop_best + Config.EARLY_STOP_MIN_DELTA
                )
                if monitor_improved:
                    early_stop_best = monitor_score
                    early_stop_failures = 0
                    best_monitor_value = early_stop_display_value(
                        Config.EARLY_STOP_METRIC, early_stop_best
                    )
                elif (
                    Config.USE_VALIDATION_PATIENCE
                    and monitor_valid
                    and (epoch + 1) >= Config.EARLY_STOP_MIN_EPOCH
                ):
                    early_stop_failures += 1

                if Config.USE_VALIDATION_PATIENCE and monitor_valid:
                    print(
                        f"  Early-stop monitor ({Config.EARLY_STOP_METRIC}): "
                        f"value={monitor_value:.4f}, best={best_monitor_value:.4f}, "
                        f"failures={early_stop_failures}/{Config.EARLY_STOP_PATIENCE}"
                    )
                append_training_metrics({
                    "run_name": Config.RUN_NAME or cv_split_tag(Config),
                    "fold": run_fold_id(Config.RUN_NAME),
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_mse": val_mse,
                    "val_rmse": val_rmse,
                    "val_ssim": val_ssim,
                    "val_r2": improved_metrics["r2"],
                    "val_tac": val_tac,
                    "val_sigma_min": improved_metrics.get("sigma_min", ""),
                    "persistence_tac": improved_metrics["persistence_tac"],
                    "weekly7_tac": improved_metrics["weekly7_tac"],
                    "weekly7_persistence_tac": improved_metrics["weekly7_persistence_tac"],
                    "weekly7_n_samples": improved_metrics["weekly7_n_samples"],
                    "weekly7_mse": improved_metrics["weekly7_mse"],
                    "weekly7_persistence_mse": improved_metrics["weekly7_persistence_mse"],
                    "tube_weekly7_tac": improved_metrics.get("tube_weekly7_tac", ""),
                    "tube_weekly7_persistence_tac": improved_metrics.get("tube_weekly7_persistence_tac", ""),
                    "tube_weekly7_n_samples": improved_metrics.get("tube_weekly7_n_samples", ""),
                    "tube_weekly7_mse": improved_metrics.get("tube_weekly7_mse", ""),
                    "tube_weekly7_persistence_mse": improved_metrics.get("tube_weekly7_persistence_mse", ""),
                    "exceedance_bss": improved_metrics.get("exceedance_bss", ""),
                    "exceedance_brier": improved_metrics.get("exceedance_brier", ""),
                    "exceedance_climo_brier": improved_metrics.get("exceedance_climo_brier", ""),
                    "exceedance_base_rate": improved_metrics.get("exceedance_base_rate", ""),
                    "exceedance_pred_rate": improved_metrics.get("exceedance_pred_rate", ""),
                    **{f"lead{lead}_mse": improved_metrics.get(f"lead{lead}_mse", "") for lead in (12, 13, 14, 15, 16, 17, 18)},
                    **{f"lead{lead}_tac": improved_metrics.get(f"lead{lead}_tac", "") for lead in (12, 13, 14, 15, 16, 17, 18)},
                    "spatial_anom_r2": improved_metrics["spatial_anom_r2"],
                    "persistence_spatial_anom_r2": improved_metrics["persistence_spatial_anom_r2"],
                    "variance_ratio": improved_metrics["variance_ratio"],
                    "gradient_ratio": improved_metrics["gradient_ratio"],
                    "extreme_bias": improved_metrics["extreme_bias"],
                    "correlation": improved_metrics["correlation"],
                    "mae": improved_metrics["mae"],
                    "crps": improved_metrics["crps"],
                    "mse_skill_vs_persistence": improved_metrics["mse_skill_vs_persistence"],
                    "persistence_r2": improved_metrics["persistence_r2"],
                    "persistence_mae": improved_metrics["persistence_mae"],
                    "zero_r2": improved_metrics["zero_r2"],
                    "train_base_mse": avg_base_mse,
                    "train_anomaly_corr_loss": avg_anom_corr,
                    "train_gradient_loss": avg_gradient,
                    "train_exceedance_bce_loss": avg_exceedance_bce,
                    "train_exceedance_count_loss": avg_exceedance_count,
                    "train_extreme_loss": avg_ext,
                    "train_sigma_min": epoch_sigma_min if Config.DISTRIBUTIONAL_HEAD else "",
                    "lr": scheduler.get_last_lr()[0],
                    "early_stop_metric": Config.EARLY_STOP_METRIC,
                    "early_stop_value": monitor_value,
                    "early_stop_best": best_monitor_value,
                    "early_stop_failures": early_stop_failures,
                })

                raw_model = model.module if ddp else model
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_ssim": val_ssim,
                    "val_r2": val_r2,
                    "val_tac": val_tac,
                    "best_ssim": max(best_ssim, val_ssim),
                    "best_r2": max(best_r2, val_r2),
                    "best_tac": max(best_tac, val_tac) if np.isfinite(val_tac) else best_tac,
                    "early_stop_metric": Config.EARLY_STOP_METRIC,
                    "early_stop_best": early_stop_best,
                    "early_stop_failures": early_stop_failures,
                    "distributional_head": bool(Config.DISTRIBUTIONAL_HEAD),
                    "crps_loss": bool(Config.CRPS_LOSS),
                    "sigma_floor": float(Config.SIGMA_FLOOR),
                    "image_channels": int(Config.IMAGE_CHANNELS),
                    "multi_lead_tube": bool(Config.MULTI_LEAD_TUBE),
                    "prediction_leads": tuple(int(x) for x in prediction_leads(Config)),
                    "tube_decode_chunk_size": int(Config.TUBE_DECODE_CHUNK_SIZE),
                }
                torch.save(ckpt, os.path.join(Config.CHECKPOINT_DIR,
                                               f"checkpoint_epoch_{epoch+1:04d}.pth"))
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    torch.save(ckpt, Config.MODEL_SAVE_PATH)
                    print(f"  New best model saved (R2={val_r2:.4f})")
                if np.isfinite(val_tac) and val_tac > best_tac:
                    best_tac = val_tac
                    torch.save(ckpt, Config.TAC_MODEL_SAVE_PATH)
                    print(f"  New best TAC model saved (TAC={val_tac:.4f})")
                if monitor_improved:
                    torch.save(ckpt, Config.MONITOR_MODEL_SAVE_PATH)
                    print(
                        f"  New best monitor model saved "
                        f"({Config.EARLY_STOP_METRIC}={monitor_value:.4f})"
                    )
                if (
                    Config.USE_VALIDATION_PATIENCE
                    and monitor_valid
                    and (epoch + 1) >= Config.EARLY_STOP_MIN_EPOCH
                    and early_stop_failures >= Config.EARLY_STOP_PATIENCE
                ):
                    stop_training = True
                    print(
                        "  Early stopping: "
                        f"{Config.EARLY_STOP_METRIC} did not improve for "
                        f"{early_stop_failures} validation checks."
                    )

        if ddp:
            stop_tensor = torch.tensor(float(stop_training), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item() > 0.5)
            dist.barrier()

        if stop_training:
            break

        scheduler.step()
        model.train()

    raw_model = model.module if ddp else model
    if ddp:
        cleanup_ddp()

    return raw_model, ema.ema, fm, train_dataset, test_dataset


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description='MeshFlowNet Direct 15-Day Prediction')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'visualize', 'resume', 'export_hindcast'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ensemble_checkpoints', type=str, default=None,
                       help='Comma-separated checkpoint paths, or "auto", to average during export_hindcast.')
    parser.add_argument('--ensemble_top_k', type=int, default=3,
                       help='When --ensemble_checkpoints auto, average the top-k checkpoints by the current early-stop metric.')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble_size', type=int, default=20)
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic mode (default is already deterministic)')
    parser.add_argument('--extreme_loss', action='store_true',
                       help='Enable exponential extreme loss for fine-tuning')
    parser.add_argument('--extreme_weight', type=float, default=0.3,
                       help='Weight for extreme loss blending')
    parser.add_argument('--anomaly_corr_weight', type=float, default=None,
                       help='Weight for spatial anomaly-correlation training loss blend.')
    parser.add_argument('--disable_anomaly_corr_loss', action='store_true',
                       help='Disable the spatial anomaly-correlation training loss term.')
    parser.add_argument('--multi_lead_tube', action='store_true',
                       help='Predict a daily multi-lead tube instead of only the center lead.')
    parser.add_argument('--prediction_leads', type=str, default=None,
                       help='Comma-separated target leads for --multi_lead_tube, e.g. 12,13,14,15,16,17,18.')
    parser.add_argument('--tube_loss_daily_weight', type=float, default=None,
                       help='Tube loss weight for mean daily MSE.')
    parser.add_argument('--tube_loss_center_weight', type=float, default=None,
                       help='Tube loss weight for center-day MSE.')
    parser.add_argument('--tube_loss_weekly_weight', type=float, default=None,
                       help='Tube loss weight for same-init weekly-mean MSE.')
    parser.add_argument('--tube_temporal_heads', type=int, default=None,
                       help='Number of attention heads for temporal mesh attention in tube mode.')
    parser.add_argument('--tube_decode_chunk_size', type=int, default=None,
                       help='Decode this many tube leads at a time; enables decoder activation checkpointing during training.')
    parser.add_argument('--gradient_loss_weight', type=float, default=None,
                       help='Blend weight for masked spatial finite-difference gradient loss.')
    parser.add_argument('--distributional_head', action='store_true',
                       help='Use a two-channel mean/sigma head for deterministic direct prediction.')
    parser.add_argument('--crps_loss', action='store_true',
                       help='Train the distributional head with closed-form Gaussian CRPS.')
    parser.add_argument('--sigma_floor', type=float, default=None,
                       help='Minimum predictive sigma in z-units for the distributional head.')
    parser.add_argument('--mse_anchor_weight', type=float, default=None,
                       help='Optional MSE anchor weight on the distributional mean; default 0.')
    parser.add_argument('--enable_exceedance_head', action='store_true',
                       help='Enable Stage 2 month-q95 exceedance logit head and exceedance validation.')
    parser.add_argument('--exceedance_bce_weight', type=float, default=None,
                       help='Loss weight for weighted/focal BCE on month-q95 exceedance labels.')
    parser.add_argument('--exceedance_count_weight', type=float, default=None,
                       help='Loss weight for regional exceedance-count fraction matching.')
    parser.add_argument('--exceedance_pos_weight', type=float, default=None,
                       help='Positive class weight for exceedance BCE.')
    parser.add_argument('--exceedance_focal_gamma', type=float, default=None,
                       help='Optional focal BCE gamma; 0 disables focal modulation.')
    parser.add_argument('--exceedance_initial_prob', type=float, default=None,
                       help='Initial exceedance-head probability bias.')
    parser.add_argument('--dry_run', action='store_true',
                       help='Run a tiny one-GPU smoke/proxy pass: 1 epoch, 2 train batches, 4 validation samples.')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max epochs.')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override per-GPU batch size.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                       help='Override validation/checkpoint frequency in epochs.')
    parser.add_argument('--num_validation_samples', type=int, default=None,
                       help='Override number of validation samples.')
    parser.add_argument('--max_train_batches', type=int, default=None,
                       help='Stop each epoch after this many train batches.')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override AdamW learning rate.')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for model initialization and stochastic layers.')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Override model dropout rate.')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                       help='Override linear warmup length in epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                       help='Stop after this many validation checks without monitor improvement.')
    parser.add_argument('--early_stop_metric', type=str, default=None,
                       choices=['tube_weekly7_tac', 'weekly7_tac', 'tac', 'r2', 'mse_skill', 'spatial_anom_r2', 'exceedance_bss', 'ssim', 'val_mse'],
                       help='Validation metric used for patience; larger is better except val_mse is internally negated.')
    parser.add_argument('--early_stop_min_epoch', type=int, default=None,
                       help='Do not count patience failures before this epoch.')
    parser.add_argument('--early_stop_min_delta', type=float, default=None,
                       help='Minimum monitor improvement required to reset patience.')
    parser.add_argument('--disable_early_stop', action='store_true',
                       help='Disable validation-patience early stopping.')
    parser.add_argument('--mesh_level', type=int, default=None,
                       help='Override icosahedral mesh refinement level.')
    parser.add_argument('--mesh_rounds', type=int, default=None,
                       help='Override mesh processor rounds.')
    parser.add_argument('--cv_stride', type=int, default=None,
                       help='Cross-validation stride for leave-k-years-out hindcasts.')
    parser.add_argument('--val_offset', type=str, default=None,
                       help='Comma-separated CV validation offsets, e.g. 3 or 1,2.')
    parser.add_argument('--test_offset', type=str, default=None,
                       help='Comma-separated CV test/hindcast offsets, e.g. 4 or 3,4.')
    parser.add_argument('--cv_fold', type=int, default=None,
                       help='Convenience fold id: test offset=fold, validation offset=(fold+1) mod stride.')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Name used to isolate checkpoints, plots, and exported hindcast stats.')
    parser.add_argument('--hindcast_splits', type=str, default='test',
                       help='Comma-separated held-out splits to export in export_hindcast mode: val,test.')
    parser.add_argument('--hindcast_output_dir', type=str, default=None,
                       help='Directory for compact hindcast TAC-stat files.')
    parser.add_argument('--hindcast_paper_dir', type=str, default=None,
                       help='Directory for per-sample paper summaries and selected map arrays.')
    parser.add_argument('--max_hindcast_samples', type=int, default=None,
                       help='Optional smoke-test cap for export_hindcast samples per split.')
    parser.add_argument('--no_paper_export', action='store_true',
                       help='Disable paper-data sidecar export during export_hindcast.')
    parser.add_argument('--paper_maps_per_month', type=int, default=None,
                       help='Selected map reservoir size per target month and fold.')
    parser.add_argument('--paper_heat_maps', type=int, default=None,
                       help='Number of largest heat-anomaly-area maps to save per fold.')

    args = parser.parse_args()

    if args.dry_run:
        Config.MAX_EPOCHS = 1
        Config.CHECKPOINT_FREQ = 1
        Config.NUM_VALIDATION_SAMPLES = 4
        Config.BATCH_SIZE = 1
        Config.MAX_TRAIN_BATCHES = 2
        Config.WARMUP_EPOCHS = 1

    if args.epochs is not None:
        Config.MAX_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.checkpoint_freq is not None:
        Config.CHECKPOINT_FREQ = args.checkpoint_freq
    if args.num_validation_samples is not None:
        Config.NUM_VALIDATION_SAMPLES = args.num_validation_samples
    if args.max_train_batches is not None:
        Config.MAX_TRAIN_BATCHES = args.max_train_batches
    if args.learning_rate is not None:
        Config.LEARNING_RATE = args.learning_rate
    if args.seed is not None:
        Config.SEED = args.seed
    if args.dropout is not None:
        Config.DROPOUT_RATE = args.dropout
    if args.warmup_epochs is not None:
        Config.WARMUP_EPOCHS = args.warmup_epochs
    if args.early_stop_patience is not None:
        Config.EARLY_STOP_PATIENCE = args.early_stop_patience
    if args.early_stop_metric is not None:
        Config.EARLY_STOP_METRIC = args.early_stop_metric
    if args.early_stop_min_epoch is not None:
        Config.EARLY_STOP_MIN_EPOCH = args.early_stop_min_epoch
    if args.early_stop_min_delta is not None:
        Config.EARLY_STOP_MIN_DELTA = args.early_stop_min_delta
    if args.disable_early_stop:
        Config.USE_VALIDATION_PATIENCE = False
    if args.anomaly_corr_weight is not None:
        Config.ANOMALY_CORR_LOSS_WEIGHT = args.anomaly_corr_weight
        Config.USE_ANOMALY_CORR_LOSS = args.anomaly_corr_weight > 0.0
    if args.disable_anomaly_corr_loss:
        Config.USE_ANOMALY_CORR_LOSS = False
    if args.multi_lead_tube:
        Config.MULTI_LEAD_TUBE = True
    if args.prediction_leads is not None:
        Config.PREDICTION_LEADS = parse_int_tuple(args.prediction_leads)
    if args.tube_loss_daily_weight is not None:
        Config.TUBE_LOSS_DAILY_WEIGHT = args.tube_loss_daily_weight
    if args.tube_loss_center_weight is not None:
        Config.TUBE_LOSS_CENTER_WEIGHT = args.tube_loss_center_weight
    if args.tube_loss_weekly_weight is not None:
        Config.TUBE_LOSS_WEEKLY_WEIGHT = args.tube_loss_weekly_weight
    if args.tube_temporal_heads is not None:
        Config.TUBE_TEMPORAL_HEADS = args.tube_temporal_heads
    if args.tube_decode_chunk_size is not None:
        if args.tube_decode_chunk_size < 0:
            raise ValueError("--tube_decode_chunk_size must be non-negative.")
        Config.TUBE_DECODE_CHUNK_SIZE = int(args.tube_decode_chunk_size)
    if args.gradient_loss_weight is not None:
        if args.gradient_loss_weight < 0.0 or args.gradient_loss_weight >= 1.0:
            raise ValueError("--gradient_loss_weight must be in [0, 1).")
        Config.GRADIENT_LOSS_WEIGHT = float(args.gradient_loss_weight)
    if args.distributional_head:
        Config.DISTRIBUTIONAL_HEAD = True
        Config.IMAGE_CHANNELS = 2
    if args.crps_loss:
        Config.CRPS_LOSS = True
    if args.sigma_floor is not None:
        if args.sigma_floor <= 0.0:
            raise ValueError("--sigma_floor must be positive.")
        Config.SIGMA_FLOOR = float(args.sigma_floor)
    if args.mse_anchor_weight is not None:
        if args.mse_anchor_weight < 0.0:
            raise ValueError("--mse_anchor_weight must be non-negative.")
        Config.MSE_ANCHOR_WEIGHT = float(args.mse_anchor_weight)
    if args.enable_exceedance_head:
        Config.ENABLE_EXCEEDANCE_HEAD = True
    if args.exceedance_bce_weight is not None:
        Config.EXCEEDANCE_BCE_WEIGHT = float(args.exceedance_bce_weight)
    if args.exceedance_count_weight is not None:
        Config.EXCEEDANCE_COUNT_WEIGHT = float(args.exceedance_count_weight)
    if args.exceedance_pos_weight is not None:
        Config.EXCEEDANCE_POS_WEIGHT = float(args.exceedance_pos_weight)
    if args.exceedance_focal_gamma is not None:
        Config.EXCEEDANCE_FOCAL_GAMMA = float(args.exceedance_focal_gamma)
    if args.exceedance_initial_prob is not None:
        Config.EXCEEDANCE_INITIAL_PROB = float(args.exceedance_initial_prob)
    if args.mesh_level is not None:
        Config.MESH_REFINEMENT_LEVEL = args.mesh_level
    if args.mesh_rounds is not None:
        Config.MESH_PROCESSOR_ROUNDS = args.mesh_rounds
    if args.cv_stride is not None:
        Config.CV_STRIDE = args.cv_stride
    if args.cv_fold is not None:
        fold = int(args.cv_fold) % int(Config.CV_STRIDE)
        Config.CV_TEST_OFFSETS = (fold,)
        Config.CV_VAL_OFFSETS = ((fold + 1) % int(Config.CV_STRIDE),)
        if args.run_name is None:
            args.run_name = f"cvfold{fold}"
    if args.val_offset is not None:
        parsed = parse_cv_offsets(args.val_offset, Config.CV_STRIDE)
        Config.CV_VAL_OFFSETS = parsed if parsed is not None else Config.CV_VAL_OFFSETS
    if args.test_offset is not None:
        parsed = parse_cv_offsets(args.test_offset, Config.CV_STRIDE)
        Config.CV_TEST_OFFSETS = parsed if parsed is not None else Config.CV_TEST_OFFSETS
    if args.hindcast_output_dir is not None:
        Config.HINDCAST_STATS_DIR = args.hindcast_output_dir
    if args.hindcast_paper_dir is not None:
        Config.HINDCAST_PAPER_DIR = args.hindcast_paper_dir
    if args.no_paper_export:
        Config.SAVE_HINDCAST_PAPER_DATA = False
    if args.paper_maps_per_month is not None:
        Config.PAPER_MAPS_PER_MONTH_PER_FOLD = args.paper_maps_per_month
    if args.paper_heat_maps is not None:
        Config.PAPER_HEAT_MAPS_PER_FOLD = args.paper_heat_maps
    apply_run_name(args.run_name)
    if args.sampling_steps is not None:
        Config.CFM_SAMPLING_STEPS = args.sampling_steps
    if args.ensemble:
        Config.ENSEMBLE_MODE = True
        Config.ENSEMBLE_SIZE = args.ensemble_size
    if args.deterministic:
        Config.DETERMINISTIC = True
    if args.extreme_loss:
        Config.USE_EXTREME_LOSS = True
        Config.EXTREME_LOSS_WEIGHT = args.extreme_weight
    if Config.MULTI_LEAD_TUBE:
        if not Config.DETERMINISTIC:
            raise RuntimeError("--multi_lead_tube is currently deterministic-only.")
        weight_sum = (
            float(Config.TUBE_LOSS_DAILY_WEIGHT)
            + float(Config.TUBE_LOSS_CENTER_WEIGHT)
            + float(Config.TUBE_LOSS_WEEKLY_WEIGHT)
        )
        if weight_sum <= 0.0:
            raise RuntimeError("Tube loss weights must sum to a positive value.")
    active_leads, active_history = validate_prediction_lead_config(Config)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            "Prediction lead preflight: PASS "
            f"(leads={active_leads}, max_lead={max(active_leads)}, "
            f"required_input_history={active_history}, "
            f"global_lowpass={Config.GLOBAL_LOWPASS_WINDOW_DAYS} days)"
        )
    if Config.ENABLE_EXCEEDANCE_HEAD:
        if not Config.DETERMINISTIC:
            raise RuntimeError("--enable_exceedance_head is deterministic-only for Stage 2.")
        if Config.EXCEEDANCE_BCE_WEIGHT < 0.0 or Config.EXCEEDANCE_COUNT_WEIGHT < 0.0:
            raise ValueError("Exceedance loss weights must be non-negative.")
        if Config.EXCEEDANCE_POS_WEIGHT <= 0.0:
            raise ValueError("--exceedance_pos_weight must be positive.")
        if Config.EXCEEDANCE_FOCAL_GAMMA < 0.0:
            raise ValueError("--exceedance_focal_gamma must be non-negative.")
        if not (0.0 < Config.EXCEEDANCE_INITIAL_PROB < 1.0):
            raise ValueError("--exceedance_initial_prob must be in (0, 1).")
    if Config.CRPS_LOSS and not Config.DISTRIBUTIONAL_HEAD:
        raise RuntimeError("--crps_loss requires --distributional_head.")
    if Config.DISTRIBUTIONAL_HEAD:
        if not Config.DETERMINISTIC:
            raise RuntimeError("--distributional_head is deterministic-direct only.")
        if Config.ENABLE_EXCEEDANCE_HEAD:
            raise RuntimeError("--distributional_head cannot be combined with --enable_exceedance_head.")
        if Config.IMAGE_CHANNELS != 2:
            raise RuntimeError("Distributional head requires Config.IMAGE_CHANNELS=2.")
        if Config.SIGMA_FLOOR <= 0.0:
            raise ValueError("SIGMA_FLOOR must be positive.")
    if str(Config.EARLY_STOP_METRIC).lower() == "exceedance_bss" and not Config.ENABLE_EXCEEDANCE_HEAD:
        raise RuntimeError("--early_stop_metric exceedance_bss requires --enable_exceedance_head.")

    set_random_seed(Config.SEED)
    apply_extended_global_fields()
    if not (0.0 <= float(Config.ANOMALY_CORR_LOSS_WEIGHT) <= 1.0):
        raise ValueError(
            "ANOMALY_CORR_LOSS_WEIGHT must be in [0, 1], got "
            f"{Config.ANOMALY_CORR_LOSS_WEIGHT}."
        )
    if Config.PREDICT_PERSISTENCE_RESIDUAL and Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
        raise RuntimeError(
            "PREDICT_PERSISTENCE_RESIDUAL requires the full daily z-score target. "
            "Set TRAIN_ON_CLIMATOLOGY_ANOMALIES=False."
        )
    if Config.DISTRIBUTIONAL_HEAD and Config.CRPS_LOSS:
        from mode_dispatch import gaussian_crps_numerical_check
        crps_err = gaussian_crps_numerical_check(num_points=16, num_samples=50000, seed=Config.SEED, device="cpu")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Gaussian CRPS numerical check abs error: {crps_err:.6f}")
        if crps_err > 1e-4:
            raise RuntimeError(f"Gaussian CRPS numerical check failed: abs error {crps_err:.6f} > 1e-4")
    print_config_banner()
    if args.dry_run and is_main_process():
        print("Dry run: 1 epoch, batch size 1, 2 train batches, 4 validation samples.")
        print("This checks wiring/finite loss/global channels; it is not a skill estimate.")

    if args.mode in ('train', 'resume'):
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        _train_worker(rank, world_size, args)
    elif args.mode == 'export_hindcast':
        _export_hindcast(args)
    elif args.mode == 'test':
        _test(args)

    if is_main_process():
        print(f"\n{'='*80}")
        print("ALL TASKS COMPLETE")
        print(f"{'='*80}")


def _train_worker(rank, world_size, args):
    model, ema_model, fm, train_dataset, test_dataset = train_model(
        rank=rank, world_size=world_size,
        checkpoint_path=args.checkpoint if args.mode == 'resume' else None
    )


def _checkpoint_list(primary_checkpoint, ensemble_checkpoints, ensemble_top_k=3):
    paths = []
    if primary_checkpoint:
        paths.append(primary_checkpoint)
    if ensemble_checkpoints:
        if str(ensemble_checkpoints).strip().lower() == "auto":
            metrics_path = os.path.join(
                Config.TRAINING_METRICS_DIR,
                f"fold_epoch_metrics_{Config.RUN_NAME or cv_split_tag(Config)}.csv",
            )
            rows = []
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            epoch = int(float(row.get("epoch", "nan")))
                            metric_value = row.get(Config.EARLY_STOP_METRIC, row.get("val_tac", "nan"))
                            score = float(metric_value)
                            if Config.EARLY_STOP_METRIC == "val_mse":
                                score = -score
                        except ValueError:
                            continue
                        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth")
                        if np.isfinite(score) and os.path.exists(ckpt_path):
                            rows.append((score, epoch, ckpt_path))
            rows.sort(key=lambda item: item[0], reverse=True)
            for _, _, ckpt_path in rows[:max(1, int(ensemble_top_k))]:
                paths.append(ckpt_path)
        else:
            for item in str(ensemble_checkpoints).replace("\n", ",").split(","):
                item = item.strip()
                if item:
                    paths.append(item)
    if not paths:
        if os.path.exists(Config.MONITOR_MODEL_SAVE_PATH):
            paths.append(Config.MONITOR_MODEL_SAVE_PATH)
        else:
            paths.append(Config.MODEL_SAVE_PATH)

    unique = []
    seen = set()
    for path in paths:
        key = os.path.abspath(os.path.expanduser(path))
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _load_meshflownet_checkpoint(checkpoint_path, mesh, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict'))
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no model_state_dict or ema_state_dict")
    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    checkpoint_tube = bool(checkpoint.get("multi_lead_tube", False))
    if not checkpoint_tube:
        checkpoint_tube = any(
            key.startswith(("lead_embedding.", "lead_time_proj.", "tube_temporal_"))
            for key in state_dict
        )
    if checkpoint_tube:
        Config.MULTI_LEAD_TUBE = True
        saved_leads = checkpoint.get("prediction_leads")
        if saved_leads is not None:
            Config.PREDICTION_LEADS = tuple(int(x) for x in saved_leads)
        lead_weight = state_dict.get("lead_embedding.weight")
        if lead_weight is not None and len(prediction_leads(Config)) != int(lead_weight.shape[0]):
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} contains {int(lead_weight.shape[0])} tube leads, "
                f"but configured prediction leads are {prediction_leads(Config)}. "
                "Pass the checkpoint's exact --prediction_leads."
            )

    if bool(checkpoint.get("distributional_head", False)) or int(checkpoint.get("image_channels", Config.IMAGE_CHANNELS)) == 2:
        Config.DISTRIBUTIONAL_HEAD = True
        Config.IMAGE_CHANNELS = 2
        Config.CRPS_LOSS = bool(checkpoint.get("crps_loss", Config.CRPS_LOSS))
        Config.SIGMA_FLOOR = float(checkpoint.get("sigma_floor", Config.SIGMA_FLOOR))
    model = build_meshflow_model(Config, mesh, device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if Config.MULTI_LEAD_TUBE and any(
        key.startswith(("lead_", "tube_")) for key in missing
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not contain the tube temporal head; "
            "use a tube checkpoint for --multi_lead_tube export."
        )
    model.eval()
    return model


def _export_hindcast(args):
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size > 1
    if ddp:
        setup_ddp(rank, world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    conus_mask = load_conus_mask(Config).to(device)
    shared_data = prepare_shared_data(Config, rank=rank, world_size=world_size, ddp=ddp)
    time_values = np.array(shared_data['time_values'])
    runs = detect_continuous_runs(time_values)
    all_valid = build_valid_indices(
        runs,
        lead_time=max_prediction_lead(Config),
        min_history=required_input_history(Config),
    )
    train_indices, val_indices, test_indices, train_years, val_years, test_years = \
        build_crossval_split(all_valid, time_values)

    if is_main_process():
        print("\n" + "=" * 80)
        print("EXPORT CROSS-VALIDATED HINDCAST TAC STATS")
        print("=" * 80)
        print(f"CV split: {cv_split_tag(Config)}")
        print(f"Train years: {len(train_years)}")
        print(f"Val years:   {sorted(val_years)}")
        print(f"Test years:  {sorted(test_years)}")

    stats_path = get_norm_stats_path(Config)
    stats_ok = os.path.exists(stats_path)
    if is_main_process() and stats_ok:
        try:
            probe = np.load(stats_path)
            stats_ok = norm_stats_match_npz(probe, Config)
            probe.close()
        except Exception:
            stats_ok = False
    if ddp:
        flag = torch.tensor(float(stats_ok), device=device)
        dist.broadcast(flag, src=0)
        stats_ok = bool(flag.item() > 0.5)

    if not stats_ok:
        if is_main_process():
            print(f"Missing/stale fold-specific normalization stats; recomputing: {stats_path}")
            tmp_dataset = ClimateDataset(Config, mode="train", train_indices=train_indices,
                                         shared_data=shared_data)
            norm_stats_tmp = get_normalization_stats(tmp_dataset)
            save_dict = {
                'hi_mean': float(norm_stats_tmp['hi_mean']),
                'hi_std': float(norm_stats_tmp['hi_std']),
                'stats_mean': norm_stats_tmp['stats_mean'].numpy(),
                'stats_std': norm_stats_tmp['stats_std'].numpy(),
                'cond_mean': norm_stats_tmp['cond_mean'].numpy(),
                'cond_std': norm_stats_tmp['cond_std'].numpy(),
                'topo_mean': float(norm_stats_tmp['topo_mean']),
                'topo_std': float(norm_stats_tmp['topo_std']),
                'toa_mean': float(norm_stats_tmp['toa_mean']),
                'toa_std': float(norm_stats_tmp['toa_std']),
                'lead_time': int(Config.LEAD_TIME),
                'multi_lead_tube': int(Config.MULTI_LEAD_TUBE),
                'prediction_leads': np.array(prediction_leads(Config), dtype=np.int16),
                'target_half_window': 0,
                'global_lag_days': np.array(Config.GLOBAL_LAG_DAYS, dtype=np.int16),
                'global_decompose_low_residual': int(Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL),
                'global_lowpass_window_days': (
                    int(Config.GLOBAL_LOWPASS_WINDOW_DAYS)
                    if Config.GLOBAL_DECOMPOSE_LOW_RESIDUAL else 0
                ),
                'local_lag_days': np.array(Config.LOCAL_LAG_DAYS, dtype=np.int16),
                'local_lag_variables': np.array(Config.LOCAL_LAG_VARIABLES),
                'train_on_climatology_anomalies': int(Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES),
                'cv_stride': int(Config.CV_STRIDE),
                'cv_val_offsets': np.array(Config.CV_VAL_OFFSETS, dtype=np.int16),
                'cv_test_offsets': np.array(Config.CV_TEST_OFFSETS, dtype=np.int16),
            }
            if norm_stats_tmp['global_mean'] is not None:
                save_dict['global_mean'] = norm_stats_tmp['global_mean'].numpy()
                save_dict['global_std'] = norm_stats_tmp['global_std'].numpy()
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            np.savez(stats_path, **save_dict)
            del tmp_dataset
            gc.collect()
    if ddp:
        dist.barrier()
    norm_stats = load_norm_stats_npz(stats_path)

    tac_climatology = load_or_build_train_climatology(
        shared_data, train_indices, norm_stats, Config, ddp=ddp
    )

    mesh = build_mesh_once(Config, conus_mask, device, ddp=ddp)
    checkpoint_paths = _checkpoint_list(args.checkpoint, args.ensemble_checkpoints, args.ensemble_top_k)
    models = []
    for checkpoint_path in checkpoint_paths:
        if is_main_process():
            print(f"Loading export checkpoint: {checkpoint_path}")
        models.append(_load_meshflownet_checkpoint(checkpoint_path, mesh, device))
    model_for_export = models if len(models) > 1 else models[0]
    if is_main_process() and len(models) > 1:
        print(f"Export checkpoint ensemble: {len(models)} members")

    datasets = {
        "val": ClimateDataset(Config, mode="val", val_indices=val_indices,
                              normalization_stats=norm_stats, shared_data=shared_data,
                              target_climatology=tac_climatology),
        "test": ClimateDataset(Config, mode="test", test_indices=test_indices,
                               normalization_stats=norm_stats, shared_data=shared_data,
                               target_climatology=tac_climatology),
    }
    selected_splits = [s.strip() for s in args.hindcast_splits.split(",") if s.strip()]
    for split_name in selected_splits:
        if split_name not in datasets:
            raise ValueError(f"Unknown hindcast split '{split_name}'. Use val,test.")
        output_name = (
            f"hindcast_tac_stats_{Config.RUN_NAME or cv_split_tag(Config)}_{split_name}.npz"
        )
        output_path = os.path.join(Config.HINDCAST_STATS_DIR, output_name)
        export_hindcast_tac_stats(
            model_for_export,
            datasets[split_name],
            split_name,
            output_path,
            device,
            conus_mask,
            tac_climatology,
            rank=rank,
            world_size=world_size,
            ddp=ddp,
            max_samples=args.max_hindcast_samples,
            save_paper_data=Config.SAVE_HINDCAST_PAPER_DATA,
            paper_output_dir=Config.HINDCAST_PAPER_DIR,
            maps_per_month=Config.PAPER_MAPS_PER_MONTH_PER_FOLD,
            heat_maps_per_fold=Config.PAPER_HEAT_MAPS_PER_FOLD,
        )

    if ddp:
        cleanup_ddp()


def _test(args):
    device = torch.device("cuda:0")
    conus_mask = load_conus_mask(Config)

    mesh = build_mesh_once(Config, conus_mask, device, ddp=False)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    if bool(checkpoint.get("distributional_head", False)) or int(checkpoint.get("image_channels", Config.IMAGE_CHANNELS)) == 2:
        Config.DISTRIBUTIONAL_HEAD = True
        Config.IMAGE_CHANNELS = 2
        Config.CRPS_LOSS = bool(checkpoint.get("crps_loss", Config.CRPS_LOSS))
        Config.SIGMA_FLOOR = float(checkpoint.get("sigma_floor", Config.SIGMA_FLOOR))

    model = build_meshflow_model(Config, mesh, device)

    state_dict = checkpoint.get('ema_state_dict', checkpoint['model_state_dict'])
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    shared_data = prepare_shared_data(Config, rank=0, world_size=1, ddp=False)

    time_values = np.array(shared_data['time_values'])
    runs = detect_continuous_runs(time_values)
    all_valid = build_valid_indices(
        runs,
        lead_time=max_prediction_lead(Config),
        min_history=required_input_history(Config),
    )

    train_indices, _, test_indices, _, _, _ = build_crossval_split(all_valid, time_values)

    stats_path = get_norm_stats_path(Config)
    s = np.load(stats_path)
    if not norm_stats_match_npz(s, Config):
        raise RuntimeError(
            f"Normalization stats at {stats_path} do not match the active "
            f"{Config.NUM_GLOBAL_CHANNELS}-channel lagged global stack or anomaly target. Re-run training setup "
            "or delete stale norm_stats_direct15_*.npz files for this fold."
        )
    norm_stats = {
        'hi_mean': torch.tensor(float(s['hi_mean'])), 'hi_std': torch.tensor(float(s['hi_std'])),
        'stats_mean': torch.from_numpy(s['stats_mean']), 'stats_std': torch.from_numpy(s['stats_std']),
        'cond_mean': torch.from_numpy(s['cond_mean']), 'cond_std': torch.from_numpy(s['cond_std']),
        'topo_mean': torch.tensor(float(s['topo_mean'])), 'topo_std': torch.tensor(float(s['topo_std'])),
        'toa_mean': torch.tensor(float(s['toa_mean'])), 'toa_std': torch.tensor(float(s['toa_std'])),
    }
    if 'global_mean' in s:
        norm_stats['global_mean'] = torch.from_numpy(s['global_mean'])
        norm_stats['global_std'] = torch.from_numpy(s['global_std'])
    else:
        norm_stats['global_mean'] = None
        norm_stats['global_std'] = None

    tac_climatology = load_or_build_train_climatology(
        shared_data, train_indices, norm_stats, Config, ddp=False
    )
    test_dataset = ClimateDataset(Config, mode="test", test_indices=test_indices,
                                  normalization_stats=norm_stats, shared_data=shared_data,
                                  target_climatology=tac_climatology)

    h, w = Config.IMAGE_SIZE

    model.eval()
    all_preds, all_truth = [], []

    for dataset_idx in tqdm(range(len(test_dataset)), desc="Test (direct 15-day)"):
        batch = test_dataset[dataset_idx]
        (y, x_t, x_tm1, x_tm2, spatial_c, vec_c, global_fields, t_idx, batch_mask) = batch

        x_t_dev = x_t.unsqueeze(0).to(device)
        x_tm1_dev = x_tm1.unsqueeze(0).to(device)
        x_tm2_dev = x_tm2.unsqueeze(0).to(device)
        spatial_c_dev = spatial_c.unsqueeze(0).to(device)
        vec_c_dev = vec_c.unsqueeze(0).to(device)
        global_fields_dev = global_fields.unsqueeze(0).to(device)
        mask_dev = batch_mask.unsqueeze(0).to(device)

        with torch.inference_mode():
            pred = predict_direct(model, x_t_dev, x_tm1_dev, x_tm2_dev,
                                  spatial_c_dev, vec_c_dev, global_fields_dev, mask_dev, device)

        if Config.MULTI_LEAD_TUBE:
            center_idx = center_lead_index(Config)
            pred_cpu = pred[0, center_idx, :h, :w].cpu()
            truth_cpu = y[center_idx, :h, :w].cpu()
        else:
            pred_cpu = pred[0, 0, :h, :w].cpu()
            truth_cpu = y[0, :h, :w].cpu()
        if Config.TRAIN_ON_CLIMATOLOGY_ANOMALIES:
            target_time_idx = int(t_idx) + int(Config.LEAD_TIME)
            target_doy = int(test_dataset.doy_values[target_time_idx])
            climo = torch.from_numpy(np.asarray(tac_climatology[target_doy], dtype=np.float32))
            pred_cpu = pred_cpu + climo
            truth_cpu = truth_cpu + climo
        all_preds.append(pred_cpu)
        all_truth.append(truth_cpu)

    predictions = torch.stack(all_preds).unsqueeze(1)
    ground_truth = torch.stack(all_truth).unsqueeze(1)

    mask_2d = conus_mask[:h, :w].cpu()
    metrics = calculate_improved_metrics(predictions, ground_truth, mask=mask_2d)

    print(f"\n{'='*80}")
    print("TEST SET METRICS (direct 15-day prediction)")
    print(f"{'='*80}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
