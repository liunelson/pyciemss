# %%[markdown]
# # Scoring with ATE and WIS
#
# Compare simulation results using ATE and WIS scores

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
df_baseline = pd.read_csv('./data/scoring_ate_wis/result_baseline.csv')
df_soft_policy = pd.read_csv('./data/scoring_ate_Wis/result_soft_policy.csv')

# %%
def average_treatment_effect(df_baseline: pd.DataFrame, df_treatment: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    num_samples_0 = len(df_baseline['sample_id'].unique())
    x0 = df_baseline[['timepoint_unknown', outcome]].groupby(['timepoint_unknown'])
    x0_mean = x0.mean()
    x0_mean_err = x0.std() / np.sqrt(num_samples_0)
    
    num_samples_1 = len(df_treatment['sample_id'].unique())
    x1 = df_treatment[['timepoint_unknown', outcome]].groupby(['timepoint_unknown'])
    x1_mean = x1.mean()
    x1_mean_err = x1.std() / np.sqrt(num_samples_1)

    ate = (x1_mean - x0_mean).reset_index()
    ate_err = (np.sqrt(x0_mean_err ** 2.0 + x1_mean_err ** 2.0)).reset_index()

    return (ate, ate_err)

# %%
outcome = 'S_state'
ate, ate_err = average_treatment_effect(df_baseline, df_soft_policy, 'S_state')

# %%
fig, ax = plt.subplots(1, 1, figsize = (8, 6))

for outcome in ('S_state', 'I_state', 'R_state'):
    ate, ate_err = average_treatment_effect(df_baseline, df_soft_policy, outcome)
    __ = ax.plot(ate['timepoint_unknown'], ate[outcome], label = outcome)
    __ = ax.fill_between(ate['timepoint_unknown'], ate[outcome] - ate_err[outcome], ate[outcome] + ate_err[outcome], alpha = 0.5)

__ = plt.setp(ax, xlabel = 'Timepoint (days)', ylabel = ('ATE'), title = 'Average Treatment Effect')
__ = ax.legend()

# %%
def compute_quantile_dict(df: pd.DataFrame, outcome: str, quantiles: list) -> dict:

    df_quantiles = df[['timepoint_unknown', outcome]].groupby('timepoint_unknown').quantile(q = quantiles).reorder_levels(order = [1, 0])
    quantile_dict = {q: df_quantiles.loc[q].values.squeeze() for q in quantiles}

    return quantile_dict

# %%
# From https://github.com/adrian-lison/interval-scoring/blob/master/scoring.py

# Interval Score
def interval_score(
    observations,
    alpha,
    q_dict=None,
    q_left=None,
    q_right=None,
    percent=False,
    check_consistency=True,
):
    """
    Compute interval scores (1) for an array of observations and predicted intervals.
    
    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be
    specified or the quantiles need to be specified via q_left and q_right.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total interval scores.
    sharpness : array_like
        Sharpness component of interval scores.
    calibration : array_like
        Calibration component of interval scores.
        
    (1) Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378.    
    """

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(alpha / 2)
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha/2}-quantile")

        q_right = q_dict.get(1 - (alpha / 2))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1-(alpha/2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    if percent:
        sharpness = sharpness / np.abs(observations)
        calibration = calibration / np.abs(observations)
    total = sharpness + calibration
    return total, sharpness, calibration

# Weighted Interval Score
def weighted_interval_score(
    observations, alphas, q_dict, weights=None, percent=False, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the WIS-score (2). A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield the double absolute percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
        
    (2) Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint arXiv:2005.12881.
    """
    if weights is None:
        weights = np.array(alphas)/2

    def weigh_scores(tuple_in, weight):
        return tuple_in[0] * weight, tuple_in[1] * weight, tuple_in[2] * weight

    interval_scores = [
        i
        for i in zip(
            *[
                weigh_scores(
                    interval_score(
                        observations,
                        alpha,
                        q_dict=q_dict,
                        percent=percent,
                        check_consistency=check_consistency,
                    ),
                    weight,
                )
                for alpha, weight in zip(alphas, weights)
            ]
        )
    ]

    total = np.sum(np.vstack(interval_scores[0]), axis=0) / sum(weights)
    sharpness = np.sum(np.vstack(interval_scores[1]), axis=0) / sum(weights)
    calibration = np.sum(np.vstack(interval_scores[2]), axis=0) / sum(weights)

    return total, sharpness, calibration

# Weighted Interval Score (Fast)
def weighted_interval_score_fast(
    observations, alphas, q_dict, weights=None, percent=False, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the WIS-score (2). A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    This is a more efficient implementation using array operations instead of repeated calls of `interval_score`.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
        
    (2) Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint arXiv:2005.12881.
    """
    if weights is None:
        weights = np.array(alphas)/2

    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        raise ValueError("Alpha values must be sorted in ascending order.")

    reversed_weights = list(reversed(weights))

    lower_quantiles = [q_dict.get(alpha / 2) for alpha in alphas]
    upper_quantiles = [q_dict.get(1 - (alpha / 2)) for alpha in reversed(alphas)]
    if any(q is None for q in lower_quantiles) or any(
        q is None for q in upper_quantiles
    ):
        raise ValueError(
            f"Quantile dictionary does not include all necessary quantiles."
        )

    lower_quantiles = np.vstack(lower_quantiles)
    upper_quantiles = np.vstack(upper_quantiles)

    # Check for consistency
    if check_consistency and np.any(
        np.diff(np.vstack((lower_quantiles, upper_quantiles)), axis=0) < 0
    ):
        raise ValueError("Quantiles are not consistent.")

    lower_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    upper_q_alphas = (2 / np.array(list(reversed(alphas)))).reshape((-1, 1))

    # compute score components for all intervals
    sharpnesses = np.flip(upper_quantiles, axis=0) - lower_quantiles

    lower_calibrations = (
        np.clip(lower_quantiles - observations, a_min=0, a_max=None) * lower_q_alphas
    )
    upper_calibrations = (
        np.clip(observations - upper_quantiles, a_min=0, a_max=None) * upper_q_alphas
    )
    calibrations = lower_calibrations + np.flip(upper_calibrations, axis=0)

    # scale to percentage absolute error
    if percent:
        sharpnesses = sharpnesses / np.abs(observations)
        calibrations = calibrations / np.abs(observations)

    totals = sharpnesses + calibrations

    # weigh scores
    weights = np.array(weights).reshape((-1, 1))

    sharpnesses_weighted = sharpnesses * weights
    calibrations_weighted = calibrations * weights
    totals_weighted = totals * weights

    # normalize and aggregate all interval scores
    weights_sum = np.sum(weights)

    sharpnesses_final = np.sum(sharpnesses_weighted, axis=0) / weights_sum
    calibrations_final = np.sum(calibrations_weighted, axis=0) / weights_sum
    totals_final = np.sum(totals_weighted, axis=0) / weights_sum

    return totals_final, sharpnesses_final, calibrations_final

# Mean Absolute Error (MAE)
def mae_score(observations, point_forecasts):
    return np.abs(observations - point_forecasts).mean()

# Mean Squared Error (MSE)
def rmse_score(observations, point_forecasts):
    return np.pow(observations - point_forecasts, 2).mean()
    
# Root Mean Squared Error (RMSE)
def rmse_score(observations, point_forecasts):
    return np.pow(np.pow(observations - point_forecasts, 2).mean(), 0.5)
    
# Mean Absolute Percentage Error (MAPE)
def mape_score(observations, point_forecasts):
    return 100 * np.abs(point_forecasts - observations) / np.abs(observations)

# Symmetric Mean Absolute Percentage Error (sMAPE)
def smape_score(observations, point_forecasts):
    return 100 * (np.abs(point_forecasts - observations) / (0.5 * (np.abs(observations) + np.abs(point_forecasts))))

# %%
# Forecast Hub required alpha quantiles
DEFAULT_ALPHA_QS = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]

# %%
observations = compute_quantile_dict(df_baseline, outcome = 'I_state', quantiles = DEFAULT_ALPHA_QS)[0.5]
alphas = DEFAULT_ALPHA_QS
q_dict = compute_quantile_dict(df_soft_policy, outcome = 'I_state', quantiles = DEFAULT_ALPHA_QS)

# %%
IS_total, IS_sharpness, IS_calibration = interval_score(
    observations, 
    alpha = 0.2,
    q_dict = q_dict, 
    percent = True
)

WIS_total, WIS_sharpness, WIS_calibration = weighted_interval_score_fast(
    observations,
    alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    weights = None, 
    q_dict = q_dict,
    percent = True
)

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
__ = ax.plot(IS_total)



# %%
fig, ax = plt.subplots(1, 1, figsize = (8, 6))

for outcome in ('S_state', 'I_state', 'R_state'):

    observations = compute_quantile_dict(df_baseline, outcome = outcome, quantiles = DEFAULT_ALPHA_QS)[0.5]
    alphas = DEFAULT_ALPHA_QS
    q_dict = compute_quantile_dict(df_soft_policy, outcome = outcome, quantiles = DEFAULT_ALPHA_QS)

    IS_total, IS_sharpness, IS_calibration = interval_score(
        observations, 
        alpha = 0.2,
        q_dict = q_dict, 
        percent = True
    )

    WIS_total, WIS_sharpness, WIS_calibration = weighted_interval_score_fast(
        observations,
        alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        weights = None, 
        q_dict = q_dict,
        percent = True
    )

    x = df_baseline['timepoint_unknown'].unique()
    y = IS_total
    z = WIS_total

    __ = ax.plot(x, y, label = outcome)
    __ = ax.plot(x, z, linestyle = '--', label = outcome)

    # __ = ax.fill_between(ate['timepoint_unknown'], ate[outcome] - ate_err[outcome], ate[outcome] + ate_err[outcome], alpha = 0.5)

__ = plt.setp(ax, xlabel = 'Timepoint (days)', ylabel = 'Interval Score', title = 'Scoring')
__ = ax.legend()


# %%