"""
================================================================================
Mode Dispatch: Deterministic (GraphCast) vs Probabilistic (GenCast/CFM)
================================================================================

Import the appropriate loss and sampling functions based on Config.DETERMINISTIC.

Usage in training script:
    from mode_dispatch import compute_loss, generate_sample

    # Training:
    loss, components = compute_loss(model, fm, y, x_t, x_tm1, x_tm2,
                                     spatial_c, vec_c, global_fields, mask,
                                     deterministic=Config.DETERMINISTIC)
    # components['pred'] contains the prediction tensor (deterministic mode)

    # Inference (single step):
    pred = generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                           global_fields, device, h, w, mask,
                           deterministic=Config.DETERMINISTIC,
                           n_steps=Config.CFM_SAMPLING_STEPS)
================================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np

OCEAN_FILL = 0


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def _input_mode(model):
    return getattr(_raw_model(model), "input_mode", "standard")


def _predicts_persistence_residual(model):
    return bool(getattr(_raw_model(model), "predict_persistence_residual", False))


def _multi_lead_tube(model):
    return bool(getattr(_raw_model(model), "multi_lead_tube", False))


def _tube_center_index(model):
    raw = _raw_model(model)
    leads = tuple(int(x) for x in getattr(raw, "prediction_leads", (15,)))
    center = int(getattr(raw, "center_lead", 15 if 15 in leads else leads[len(leads) // 2]))
    return leads.index(center)


def _assert_tube_lead_shape(model, pred, target):
    expected = len(tuple(int(x) for x in getattr(_raw_model(model), "prediction_leads", (15,))))
    if pred.ndim < 2 or int(pred.shape[1]) != expected:
        raise RuntimeError(
            f"Tube prediction lead dimension mismatch: got shape={tuple(pred.shape)}, "
            f"expected L={expected}."
        )
    if target.ndim < 2 or int(target.shape[1]) != expected:
        raise RuntimeError(
            f"Tube target lead dimension mismatch: got shape={tuple(target.shape)}, "
            f"expected L={expected}."
        )


def _tube_loss_weights(model):
    raw = _raw_model(model)
    return tuple(float(x) for x in getattr(raw, "tube_loss_weights", (0.80, 0.10, 0.10)))


def _gradient_loss_weight(model):
    return float(getattr(_raw_model(model), "gradient_loss_weight", 0.0))


def _distributional_head(model):
    return bool(getattr(_raw_model(model), "distributional_head", False))


def _crps_loss_enabled(model):
    return bool(getattr(_raw_model(model), "crps_loss", False))


def _sigma_floor(model):
    return float(getattr(_raw_model(model), "sigma_floor", 0.1))


def _mse_anchor_weight(model):
    return float(getattr(_raw_model(model), "mse_anchor_weight", 0.0))


def _enable_exceedance_head(model):
    return bool(getattr(_raw_model(model), "enable_exceedance_head", False))


def _exceedance_loss_weights(model):
    raw = _raw_model(model)
    return (
        float(getattr(raw, "exceedance_bce_weight", 0.0)),
        float(getattr(raw, "exceedance_count_weight", 0.0)),
        float(getattr(raw, "exceedance_pos_weight", 10.0)),
        float(getattr(raw, "exceedance_focal_gamma", 0.0)),
    )


def spatial_gradient_loss(pred, target, mask):
    """Match masked spatial finite-difference gradients."""
    if pred.shape != target.shape:
        raise RuntimeError(
            f"Gradient loss shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    mask = mask.to(device=pred.device, dtype=pred.dtype)
    mask_expanded = mask.expand_as(pred)

    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_true = target[..., 1:, :] - target[..., :-1, :]
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_true = target[..., :, 1:] - target[..., :, :-1]

    mask_y = mask_expanded[..., 1:, :] * mask_expanded[..., :-1, :]
    mask_x = mask_expanded[..., :, 1:] * mask_expanded[..., :, :-1]

    loss_y = ((dy_pred - dy_true).square() * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    loss_x = ((dx_pred - dx_true).square() * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    return loss_y + loss_x


def split_distributional_prediction(model, raw_pred, x_t):
    """Return persistence-adjusted mean and positive sigma from a two-channel raw output."""
    floor = _sigma_floor(model)
    if raw_pred.ndim == 5:
        if raw_pred.shape[2] != 2:
            raise RuntimeError(f"Distributional tube output must be (B,L,2,H,W), got {tuple(raw_pred.shape)}")
        mean_raw = raw_pred[:, :, 0]
        sigma_raw = raw_pred[:, :, 1]
        persistence = x_t if x_t.ndim == 4 else x_t.unsqueeze(1)
        mean = persistence + mean_raw if _predicts_persistence_residual(model) else mean_raw
    elif raw_pred.ndim == 4:
        if raw_pred.shape[1] != 2:
            raise RuntimeError(f"Distributional output must be (B,2,H,W), got {tuple(raw_pred.shape)}")
        mean_raw = raw_pred[:, 0:1]
        sigma_raw = raw_pred[:, 1:2]
        mean = x_t + mean_raw if _predicts_persistence_residual(model) else mean_raw
    else:
        raise RuntimeError(f"Unsupported distributional output shape: {tuple(raw_pred.shape)}")
    sigma = F.softplus(sigma_raw) + float(floor)
    raw = _raw_model(model)
    raw.last_sigma = sigma.detach()
    raw.last_sigma_min = float(sigma.detach().amin().item())
    return mean, sigma


def gaussian_crps(mean, sigma, target, mask):
    """Closed-form Gaussian CRPS, masked to land pixels."""
    sigma = sigma.clamp_min(1e-6)
    target = target.to(device=mean.device, dtype=mean.dtype)
    mask_expanded = mask.to(device=mean.device, dtype=mean.dtype).expand_as(mean)
    valid = mask_expanded > 0.5
    if not valid.any():
        return mean.sum() * 0.0
    w = (target - mean) / sigma
    phi = torch.exp(-0.5 * w.square()) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + torch.erf(w / np.sqrt(2.0)))
    crps = sigma * (w * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return (crps * mask_expanded).sum() / mask_expanded.sum().clamp_min(1.0)


def gaussian_crps_numerical_check(num_points=16, num_samples=40001, seed=123, device="cpu"):
    """Deterministic quadrature check for random scalar Gaussian CRPS."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    mean = torch.randn(int(num_points), device=device, generator=gen)
    sigma = F.softplus(torch.randn(int(num_points), device=device, generator=gen)) + 0.1
    y = torch.randn(int(num_points), device=device, generator=gen)
    analytic = gaussian_crps(
        mean.view(1, 1, -1),
        sigma.view(1, 1, -1),
        y.view(1, 1, -1),
        torch.ones(1, 1, int(num_points), device=device),
    )
    vals = []
    n = int(num_samples)
    for i in range(int(num_points)):
        lo = min(float(mean[i] - 10.0 * sigma[i]), float(y[i] - 10.0 * sigma[i]))
        hi = max(float(mean[i] + 10.0 * sigma[i]), float(y[i] + 10.0 * sigma[i]))
        grid = torch.linspace(lo, hi, n, device=device)
        cdf = 0.5 * (1.0 + torch.erf((grid - mean[i]) / (sigma[i] * np.sqrt(2.0))))
        obs = (grid >= y[i]).to(dtype=grid.dtype)
        vals.append(torch.trapz((cdf - obs).square(), grid))
    brute = torch.stack(vals).mean()
    return float((analytic - brute).abs().item())


def exceedance_losses(model, y, mask, exceedance_thresholds):
    raw = _raw_model(model)
    logits = getattr(raw, "last_exceedance_logits", None)
    if logits is None:
        raise RuntimeError("Exceedance head is enabled, but model.last_exceedance_logits is missing.")
    if exceedance_thresholds is None:
        raise RuntimeError("Exceedance head is enabled, but no month-q95 thresholds were provided.")
    if logits.shape != y.shape:
        raise RuntimeError(
            f"Exceedance logits shape mismatch: logits={tuple(logits.shape)}, target={tuple(y.shape)}"
        )

    thresholds = exceedance_thresholds.to(device=y.device, dtype=y.dtype)
    labels = (y > thresholds).to(dtype=y.dtype)
    mask_expanded = mask.to(device=y.device, dtype=y.dtype).expand_as(y)
    valid = mask_expanded > 0.5
    if not valid.any():
        zero = logits.sum() * 0.0
        return zero, zero

    bce_weight, count_weight, pos_weight, focal_gamma = _exceedance_loss_weights(model)
    pos_weight_t = torch.tensor(float(pos_weight), device=y.device, dtype=y.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits[valid],
        labels[valid],
        pos_weight=pos_weight_t,
        reduction="none",
    )
    if focal_gamma > 0.0:
        probs = torch.sigmoid(logits[valid])
        p_t = torch.where(labels[valid] > 0.5, probs, 1.0 - probs)
        bce = ((1.0 - p_t).clamp_min(1e-6) ** float(focal_gamma)) * bce
    bce_loss = bce.mean()

    probs = torch.sigmoid(logits) * mask_expanded
    labels_masked = labels * mask_expanded
    region_masks = getattr(raw, "exceedance_region_masks", None)
    if region_masks is not None:
        region_masks = region_masks.to(device=y.device, dtype=y.dtype)
        if y.ndim == 4:
            # y: (B,L,H,W), region_masks: (R,H,W)
            denom = (region_masks.unsqueeze(0).unsqueeze(0) * mask_expanded.unsqueeze(2)).sum(
                dim=(-2, -1)
            ).clamp_min(1.0)
            pred_frac = (probs.unsqueeze(2) * region_masks.unsqueeze(0).unsqueeze(0)).sum(
                dim=(-2, -1)
            ) / denom
            obs_frac = (labels_masked.unsqueeze(2) * region_masks.unsqueeze(0).unsqueeze(0)).sum(
                dim=(-2, -1)
            ) / denom
        else:
            # y: (B,1,H,W), region_masks: (R,H,W)
            denom = (region_masks.unsqueeze(0) * mask_expanded).sum(dim=(-2, -1)).clamp_min(1.0)
            pred_frac = (probs * region_masks.unsqueeze(0)).sum(dim=(-2, -1)) / denom
            obs_frac = (labels_masked * region_masks.unsqueeze(0)).sum(dim=(-2, -1)) / denom
        count_loss = (pred_frac - obs_frac).square().mean()
    else:
        denom = mask_expanded.sum(dim=tuple(range(2, mask_expanded.ndim))).clamp_min(1.0)
        pred_frac = probs.sum(dim=tuple(range(2, probs.ndim))) / denom
        obs_frac = labels_masked.sum(dim=tuple(range(2, labels_masked.ndim))) / denom
        count_loss = (pred_frac - obs_frac).square().mean()

    return bce_loss, count_loss


def _set_exceedance_logits_from_prediction(model, pred):
    raw = _raw_model(model)
    if not _enable_exceedance_head(model):
        raw.last_exceedance_logits = None
        return
    if not hasattr(raw, "exceedance_head"):
        raise RuntimeError("Exceedance head is enabled, but model.exceedance_head is missing.")

    if pred.ndim == 4 and _multi_lead_tube(model):
        b, n_leads, h, w = pred.shape
        logits = raw.exceedance_head(pred.reshape(b * n_leads, 1, h, w))
        raw.last_exceedance_logits = logits.reshape(b, n_leads, 1, h, w).squeeze(2)
    elif pred.ndim == 4:
        raw.last_exceedance_logits = raw.exceedance_head(pred)
    elif pred.ndim == 5:
        b, n_leads, channels, h, w = pred.shape
        logits = raw.exceedance_head(pred.reshape(b * n_leads, channels, h, w))
        raw.last_exceedance_logits = logits.reshape(b, n_leads, channels, h, w).squeeze(2)
    else:
        raise RuntimeError(f"Unsupported prediction shape for exceedance logits: {tuple(pred.shape)}")


def _ensure_standard_input(model):
    mode = _input_mode(model)
    if mode != "standard":
        raise ValueError(
            f"Unsupported input_mode={mode!r}. JEPA inputs are disabled; use standard MeshFlowNet inputs."
        )


def _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c):
    _ensure_standard_input(model)
    return torch.cat([x_t, x_tm1, x_tm2, spatial_c], dim=1)


def _cfm_input(model, flow_state, x_t, x_tm1, x_tm2, spatial_c):
    _ensure_standard_input(model)
    return torch.cat([flow_state, x_t, x_tm1, x_tm2, spatial_c], dim=1)


# =============================================================================
# UNIFIED DISPATCH
# =============================================================================

def compute_loss(model, fm, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
                 global_fields, mask, deterministic=False, exceedance_thresholds=None):
    if deterministic:
        return deterministic_loss(model, y, x_t, x_tm1, x_tm2,
                                  spatial_c, vec_c, global_fields, mask,
                                  exceedance_thresholds=exceedance_thresholds)
    else:
        return cfm_loss(model, fm, y, x_t, x_tm1, x_tm2,
                        spatial_c, vec_c, global_fields, mask)


def generate_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                    global_fields, device, h, w, mask,
                    deterministic=False, n_steps=50):
    """
    Single-step generation. Returns (h, w) numpy array.
    """
    if deterministic:
        return generate_deterministic_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, mask)
    else:
        return generate_cfm_sample(
            model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
            global_fields, device, h, w, mask,
            n_steps=n_steps)


# =============================================================================
# DETERMINISTIC MODE (GraphCast)
# =============================================================================

def deterministic_loss(model, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
                       global_fields, mask, exceedance_thresholds=None):
    """
    Deterministic direct prediction loss.
    Returns prediction in components['pred'] for downstream use (e.g. extreme loss).
    """
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
    dummy_t = torch.full((y.shape[0],), 0.5, device=device)

    raw_pred = model(x_input, dummy_t, vec_c, global_fields=global_fields)
    if _distributional_head(model):
        pred, sigma = split_distributional_prediction(model, raw_pred, x_t)
    else:
        pred = x_t + raw_pred if _predicts_persistence_residual(model) else raw_pred
        sigma = None
    _set_exceedance_logits_from_prediction(model, pred)
    gradient_weight = _gradient_loss_weight(model)
    use_crps = _distributional_head(model) and _crps_loss_enabled(model)
    mse_anchor_weight = _mse_anchor_weight(model)
    exceedance_enabled = _enable_exceedance_head(model)
    exceedance_bce_weight, exceedance_count_weight, _, _ = _exceedance_loss_weights(model)
    zero = torch.tensor(0.0, device=device)

    if _multi_lead_tube(model):
        if y.ndim != 4:
            raise RuntimeError(f"Tube target must have shape (B,L,H,W), got {tuple(y.shape)}")
        _assert_tube_lead_shape(model, pred, y)
        mask_expanded = mask.expand_as(pred)
        valid = mask_expanded > 0.5
        if valid.any():
            daily_mse = ((pred - y).square() * mask_expanded).sum() / mask_expanded.sum().clamp_min(1.0)
            daily_crps = gaussian_crps(pred, sigma, y, mask) if use_crps else daily_mse
            center_idx = _tube_center_index(model)
            center_pred = pred[:, center_idx:center_idx + 1]
            center_truth = y[:, center_idx:center_idx + 1]
            center_mask = mask.expand_as(center_pred)
            center_mse = ((center_pred - center_truth).square() * center_mask).sum() / center_mask.sum().clamp_min(1.0)
            center_crps = (
                gaussian_crps(center_pred, sigma[:, center_idx:center_idx + 1], center_truth, mask)
                if use_crps else center_mse
            )
            weekly_pred = pred.mean(dim=1, keepdim=True)
            weekly_truth = y.mean(dim=1, keepdim=True)
            weekly_mask = mask.expand_as(weekly_pred)
            weekly_mse = ((weekly_pred - weekly_truth).square() * weekly_mask).sum() / weekly_mask.sum().clamp_min(1.0)
            if use_crps:
                n_leads = max(int(sigma.shape[1]), 1)
                weekly_sigma = torch.sqrt((sigma.square().mean(dim=1, keepdim=True) / n_leads).clamp_min(1e-8))
                weekly_crps = gaussian_crps(weekly_pred, weekly_sigma, weekly_truth, mask)
            else:
                weekly_crps = weekly_mse
        else:
            daily_mse = pred.sum() * 0.0
            daily_crps = daily_mse
            center_mse = daily_mse
            center_crps = daily_mse
            weekly_mse = daily_mse
            weekly_crps = daily_mse
        w_daily, w_center, w_weekly = _tube_loss_weights(model)
        if use_crps:
            base_loss = w_daily * daily_crps + w_center * center_crps + w_weekly * weekly_crps
            if mse_anchor_weight > 0.0:
                base_loss = base_loss + mse_anchor_weight * daily_mse
        else:
            base_loss = w_daily * daily_mse + w_center * center_mse + w_weekly * weekly_mse
        grad_loss = spatial_gradient_loss(pred, y, mask) if gradient_weight > 0.0 else zero
        total_loss = (1.0 - gradient_weight) * base_loss + gradient_weight * grad_loss
        exceedance_bce = zero
        exceedance_count = zero
        if exceedance_enabled:
            exceedance_bce, exceedance_count = exceedance_losses(model, y, mask, exceedance_thresholds)
            total_loss = (
                total_loss
                + exceedance_bce_weight * exceedance_bce
                + exceedance_count_weight * exceedance_count
            )
        return total_loss, {
            "det_loss": total_loss.detach(),
            "recon_mse": daily_mse.detach(),
            "tube_daily_mse": daily_mse.detach(),
            "tube_center_mse": center_mse.detach(),
            "tube_weekly_mse": weekly_mse.detach(),
            "tube_daily_crps": daily_crps.detach(),
            "tube_center_crps": center_crps.detach(),
            "tube_weekly_crps": weekly_crps.detach(),
            "gradient_loss": grad_loss,
            "exceedance_bce_loss": exceedance_bce,
            "exceedance_count_loss": exceedance_count,
            "residual_abs": raw_pred.detach().abs().mean(),
            "sigma_min": sigma.detach().amin() if sigma is not None else zero,
            "loss_t<0.33": zero,
            "loss_0.33<t<0.67": zero,
            "loss_t>0.67": zero,
            "pred": pred,
            "sigma": sigma,
        }

    mask_expanded = mask.expand_as(pred)
    valid = mask_expanded > 0.5
    if valid.any():
        total_loss = gaussian_crps(pred, sigma, y, mask) if use_crps else F.huber_loss(pred[valid], y[valid], delta=2.0)
        recon_mse = F.mse_loss(pred[valid], y[valid])
        if use_crps and mse_anchor_weight > 0.0:
            total_loss = total_loss + mse_anchor_weight * recon_mse
    else:
        total_loss = pred.sum() * 0.0
        recon_mse = total_loss.detach()

    grad_loss = spatial_gradient_loss(pred, y, mask) if gradient_weight > 0.0 else zero
    total_loss = (1.0 - gradient_weight) * total_loss + gradient_weight * grad_loss
    exceedance_bce = zero
    exceedance_count = zero
    if exceedance_enabled:
        exceedance_bce, exceedance_count = exceedance_losses(model, y, mask, exceedance_thresholds)
        total_loss = (
            total_loss
            + exceedance_bce_weight * exceedance_bce
            + exceedance_count_weight * exceedance_count
        )
    return total_loss, {
        "det_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
        "gradient_loss": grad_loss,
        "exceedance_bce_loss": exceedance_bce,
        "exceedance_count_loss": exceedance_count,
        "residual_abs": raw_pred.detach().abs().mean(),
        "sigma_min": sigma.detach().amin() if sigma is not None else zero,
        "loss_t<0.33": zero,
        "loss_0.33<t<0.67": zero,
        "loss_t>0.67": zero,
        "pred": pred,  # prediction tensor for extreme loss
        "sigma": sigma,
    }


@torch.inference_mode()
def generate_deterministic_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                                  global_fields, device, h, w, mask):
    model.eval()
    mask = mask.to(device=device, dtype=x_t.dtype)
    x_input = _deterministic_input(model, x_t, x_tm1, x_tm2, spatial_c)
    dummy_t = torch.full((1,), 0.5, device=device)

    raw_hat = model(x_input, dummy_t, vec_c, global_fields=global_fields)
    if _distributional_head(model):
        y_hat, sigma = split_distributional_prediction(model, raw_hat, x_t)
    else:
        y_hat = x_t + raw_hat if _predicts_persistence_residual(model) else raw_hat
        sigma = None
    y_hat = y_hat * mask + OCEAN_FILL * (1 - mask)
    y_hat = y_hat.clamp(-4.0, 4.0)
    if sigma is not None:
        _raw_model(model).last_sigma = sigma.detach()

    if _multi_lead_tube(model):
        center_idx = _tube_center_index(model)
        return y_hat[0, center_idx, :h, :w].cpu().numpy()

    return y_hat[0, 0, :h, :w].cpu().numpy()


# =============================================================================
# PROBABILISTIC MODE (GenCast/CFM)
# =============================================================================

def sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0):
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


def cfm_loss(model, fm, y, x_t, x_tm1, x_tm2, spatial_c, vec_c,
             global_fields, mask):
    batch_size = y.shape[0]
    device = y.device
    mask = mask.to(device=device, dtype=y.dtype)

    times = sample_times_logit_normal(batch_size, device, mean=0.0, std=1.0)
    times = times.clamp(1e-5, 1.0 - 1e-5)

    x_t_flow = fm.sample_xt(x_0=x_t, x_1=y, t=times)
    x_t_flow = x_t_flow * mask + OCEAN_FILL * (1 - mask)
    v_target = fm.velocity_target(x_0=x_t, x_1=y)

    x_input = _cfm_input(model, x_t_flow, x_t, x_tm1, x_tm2, spatial_c)
    v_pred = model(x_input, times, vec_c, global_fields=global_fields)

    loss_per_pixel = (v_pred - v_target) ** 2 * mask
    loss_per_sample = loss_per_pixel.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
    total_loss = loss_per_sample.mean()

    with torch.no_grad():
        low_mask = times < 0.33
        mid_mask = (times >= 0.33) & (times < 0.67)
        high_mask = times >= 0.67
        loss_low = loss_per_sample[low_mask].mean() if low_mask.any() else torch.tensor(0.0, device=device)
        loss_mid = loss_per_sample[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=device)
        loss_high = loss_per_sample[high_mask].mean() if high_mask.any() else torch.tensor(0.0, device=device)

    t_view = times.view(-1, 1, 1, 1)
    y_recon = x_t_flow + (1 - t_view) * v_pred
    recon_mse = ((y_recon - y) ** 2 * mask).sum(dim=(1, 2, 3))
    recon_mse = (recon_mse / (mask.sum(dim=(1, 2, 3)) + 1e-8)).mean()

    return total_loss, {
        "cfm_loss": total_loss.detach(),
        "recon_mse": recon_mse.detach(),
        "loss_t<0.33": loss_low.detach(),
        "loss_0.33<t<0.67": loss_mid.detach(),
        "loss_t>0.67": loss_high.detach(),
        "pred": None,  # CFM doesn't have a single prediction during training
    }


@torch.inference_mode()
def generate_cfm_sample(model, spatial_c, vec_c, x_t, x_tm1, x_tm2,
                        global_fields, device, h, w, mask,
                        n_steps=50):
    """
    ODE integration (Heun's method) for one step.
    """
    model.eval()
    mask = mask.to(device=device, dtype=x_t.dtype)

    z = x_t.clone()
    z = z * mask + OCEAN_FILL * (1 - mask)

    dt = 1.0 / n_steps
    VAL_MIN, VAL_MAX = -4.0, 4.0

    for i in range(n_steps):
        t_i = torch.tensor([i * dt], device=device)
        t_next = torch.tensor([(i + 1) * dt], device=device).clamp(max=1.0)

        x_input = _cfm_input(model, z, x_t, x_tm1, x_tm2, spatial_c)
        v1 = model(x_input, t_i.expand(1), vec_c, global_fields=global_fields)
        v1 = v1 * mask

        z_euler = z + v1 * dt
        z_euler = z_euler.clamp(VAL_MIN, VAL_MAX)
        z_euler = z_euler * mask + OCEAN_FILL * (1 - mask)

        x_input2 = _cfm_input(model, z_euler, x_t, x_tm1, x_tm2, spatial_c)
        v2 = model(x_input2, t_next.expand(1), vec_c, global_fields=global_fields)
        v2 = v2 * mask

        z = z + (v1 + v2) * 0.5 * dt
        z = z.clamp(VAL_MIN, VAL_MAX)
        z = z * mask + OCEAN_FILL * (1 - mask)

    return z[0, 0, :h, :w].cpu().numpy()
