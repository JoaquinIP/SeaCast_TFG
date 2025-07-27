# Third-party
import torch


def get_metric(metric_name):
    """
    Get a defined metric with given name

    metric_name: str, name of the metric

    Returns:
    metric: function implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(
    metric_entry_vals, mask, grid_weights, average_grid, sum_vars
):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """

    if grid_weights is not None:
        metric_entry_vals = metric_entry_vals * grid_weights

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        metric_entry_vals = torch.sum(
            mask * metric_entry_vals, dim=-2
        ) / torch.sum(mask, dim=-2)
        # (..., d_state)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(
            metric_entry_vals, dim=-1
        )  # (..., N) or (...,)

    return metric_entry_vals


def wmse(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    Weighted Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mse_weighted = entry_mse / (pred_std**2)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mse(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    (Unweighted) Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmse(
        pred,
        target,
        torch.ones_like(pred_std),
        mask,
        grid_weights,
        average_grid,
        sum_vars,
    )


def wmae(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    Weighted Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mae_weighted = entry_mae / pred_std  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mae(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    (Unweighted) Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred,
        target,
        torch.ones_like(pred_std),
        mask,
        grid_weights,
        average_grid,
        sum_vars,
    )


def nll(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def crps_gauss(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Closed-form expression based on Gaussian predictive distribution

    (...,) is any number of batch dimensions, potentially different
            but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (..., N, d_state), mask describing which grid nodes to use in metric
    grid_weights: (N, 1), weights for each grid point
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device), torch.ones((), device=pred.device)
    )
    target_standard = (target - pred) / pred_std  # (..., N, d_state)

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_crps,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )

import torch.nn.functional as F

def huber_loss(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
    delta=1.0,
):
    """
    Huber Loss usando torch.nn.functional.huber_loss

    Esta función calcula el Huber Loss, que es cuadrático para errores pequeños (|error| <= delta)
    y lineal para errores grandes (|error| > delta). Se utiliza reduction='none' para obtener
    el tensor de pérdidas elemento a elemento, y luego se aplica una máscara y se reducen las dimensiones
    mediante mask_and_reduce_metric.

    Parámetros:
    - pred: (..., N, d_state), predicción del modelo.
    - target: (..., N, d_state), valor real.
    - pred_std: (..., N, d_state) o (d_state,), desviación estándar predicha (no utilizada en este cálculo).
    - mask: (..., N, d_state), máscara que indica qué nodos se deben considerar.
    - grid_weights: (N, 1), pesos para cada punto de la grilla.
    - average_grid: boolean, si se debe promediar la dimensión de la grilla.
    - sum_vars: boolean, si se deben sumar las variables.
    - delta: float, umbral que separa el comportamiento cuadrático del lineal.

    Retorna:
    - metric_val: Valor final de la pérdida, tras aplicar máscara y reducción.
    """
    loss = F.huber_loss(pred, target, delta=delta, reduction="none")
    
    return mask_and_reduce_metric(
        loss,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )

def huber_loss_2(
    pred,
    target,
    pred_std,
    mask=None,
    grid_weights=None,
    average_grid=True,
    sum_vars=True,
    delta=0.5986,
):
    """
    Huber Loss usando torch.nn.functional.huber_loss

    Esta función calcula el Huber Loss, que es cuadrático para errores pequeños (|error| <= delta)
    y lineal para errores grandes (|error| > delta). Se utiliza reduction='none' para obtener
    el tensor de pérdidas elemento a elemento, y luego se aplica una máscara y se reducen las dimensiones
    mediante mask_and_reduce_metric.

    Parámetros:
    - pred: (..., N, d_state), predicción del modelo.
    - target: (..., N, d_state), valor real.
    - pred_std: (..., N, d_state) o (d_state,), desviación estándar predicha (no utilizada en este cálculo).
    - mask: (..., N, d_state), máscara que indica qué nodos se deben considerar.
    - grid_weights: (N, 1), pesos para cada punto de la grilla.
    - average_grid: boolean, si se debe promediar la dimensión de la grilla.
    - sum_vars: boolean, si se deben sumar las variables.
    - delta: float, umbral que separa el comportamiento cuadrático del lineal.

    Retorna:
    - metric_val: Valor final de la pérdida, tras aplicar máscara y reducción.
    """
    loss = F.huber_loss(pred, target, delta=delta, reduction="none")
    
    return mask_and_reduce_metric(
        loss,
        mask=mask,
        grid_weights=grid_weights,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "huber": huber_loss,
    "huber_2": huber_loss_2,
}
