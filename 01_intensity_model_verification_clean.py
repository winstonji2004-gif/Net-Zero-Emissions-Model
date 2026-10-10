from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
VALID_START = 2010
VALID_END = 2024
DELTA_AICC_THRESHOLD = 2.0
HOLDOUT_TOLERANCE = 1.05
MAX_DECLINE_RATE = 0.5
FIGURE_DPI = 180
DATA_FILE: str | None = None
OUTPUT_DIRECTORY: str | None = None
for font_name in [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
DATA_FILENAMES = [
    "05_emissions_energy_1965_2024.csv",
    "05_emissions_energy_1965_2024(1).csv",
    "05_emissions_energy_1965_2024(2).csv",
    "02_historical_observed_data_1965_2024.csv",
    "02_historical_observed_data_1965_2024(1).csv",
    "02_historical_observed_data_1965_2024(2).csv",
    "01_calibration_input_data_1965_2024.csv",
    "01_calibration_input_data_1965_2024(1).csv",
    "01_calibration_input_data_1965_2024(2).csv",
]
SEARCH_DIRECTORIES = [
    SCRIPT_DIR,
    Path.cwd(),
    Path("/mnt/data"),
    Path(r"F:/MATH3041_NetZero_Cleaned_Data"),
    Path(r"C:/Users/winst/Desktop/groupwork"),
    Path(r"C:/Users/winst/Desktop/groupwork/data/MATH3041"),
]
def locate_data_file() -> Path:
    candidates: List[Path] = []
    if DATA_FILE:
        candidates.append(Path(DATA_FILE))
    env_path = os.getenv("MATH3041_INTENSITY_DATA")
    if env_path:
        candidates.append(Path(env_path))
    for directory in SEARCH_DIRECTORIES:
        for filename in DATA_FILENAMES:
            candidates.append(directory / filename)
    checked: List[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate.resolve()
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find a CSV containing the intensity data. Place the CSV in the same directory as this script, "
        "set DATA_FILE, or define the MATH3041_INTENSITY_DATA environment variable.\n"
        f"Checked paths:\n{checked_text}"
    )
def make_output_directory() -> Path:
    configured = OUTPUT_DIRECTORY or os.getenv("MATH3041_INTENSITY_OUTPUT")
    output_dir = (
        Path(configured)
        if configured
        else SCRIPT_DIR / "intensity_model_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()
REQUIRED_COLUMNS = [
    "year",
    "energy_intensity_kwh_per_usd",
    "carbon_intensity_kgco2_per_kwh",
]
def read_and_validate_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).copy()
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    data = data.sort_values("year").reset_index(drop=True)
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[REQUIRED_COLUMNS].isna().any().any():
        bad_rows = data.loc[data[REQUIRED_COLUMNS].isna().any(axis=1), REQUIRED_COLUMNS]
        raise ValueError(f"Required columns contain missing or non-numeric values:\n{bad_rows}")
    years_float = data["year"].to_numpy(dtype=float)
    if not np.allclose(years_float, np.round(years_float)):
        raise ValueError("The year column must contain integer years.")
    data["year"] = np.round(years_float).astype(int)
    if data["year"].duplicated().any():
        duplicates = data.loc[data["year"].duplicated(keep=False), "year"].tolist()
        raise ValueError(f"Duplicate years: {duplicates}")
    year_differences = np.diff(data["year"].to_numpy())
    if np.any(year_differences <= 0):
        raise ValueError("Years must be strictly increasing.")
    if np.any(year_differences != 1):
        missing_years = []
        all_years = set(range(int(data["year"].min()), int(data["year"].max()) + 1))
        missing_years = sorted(all_years - set(data["year"].tolist()))
        warnings.warn(f"The annual series is incomplete; missing years: {missing_years}")
    for column in [
        "energy_intensity_kwh_per_usd",
        "carbon_intensity_kgco2_per_kwh",
    ]:
        values = data[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} contains infinite values.")
        if np.any(values <= 0):
            raise ValueError(f"{column} must contain only positive values.")
    return data
def verify_intensity_construction(data: pd.DataFrame) -> None:
    print("\n===== Data Consistency Check =====")
    print(
        f"Year range: {int(data['year'].min())}--{int(data['year'].max())}; observations: {len(data)}"
    )
    print("Missing values in required intensity columns: 0")
    if {"primary_energy_twh", "gdp_constant_2015_usd"}.issubset(data.columns):
        calculated_ie = (
            data["primary_energy_twh"].to_numpy(dtype=float)
            * 1e9
            / data["gdp_constant_2015_usd"].to_numpy(dtype=float)
        )
        observed_ie = data["energy_intensity_kwh_per_usd"].to_numpy(dtype=float)
        relative_error = np.abs(calculated_ie - observed_ie) / np.maximum(
            np.abs(observed_ie), 1e-15
        )
        print(
            "Maximum relative difference from reconstructing the energy-intensity definition: "
            f"{100 * np.max(relative_error):.8f}%"
        )
    else:
        print("Both primary_energy_twh and gdp_constant_2015_usd were not found; skipping the I_E reconstruction check.")
    if {"fossil_co2_mtco2", "primary_energy_twh"}.issubset(data.columns):
        calculated_ic = (
            data["fossil_co2_mtco2"].to_numpy(dtype=float)
            / data["primary_energy_twh"].to_numpy(dtype=float)
        )
        observed_ic = data["carbon_intensity_kgco2_per_kwh"].to_numpy(dtype=float)
        relative_error = np.abs(calculated_ic - observed_ic) / np.maximum(
            np.abs(observed_ic), 1e-15
        )
        print(
            "Maximum relative difference from reconstructing the carbon-intensity definition: "
            f"{100 * np.max(relative_error):.8f}%"
        )
    else:
        print("Both fossil_co2_mtco2 and primary_energy_twh were not found; skipping the I_C reconstruction check.")
def exp_nofloor(relative_year: np.ndarray, I0: float, r: float) -> np.ndarray:
    return I0 * np.exp(-r * relative_year)
def exp_floor(
    relative_year: np.ndarray,
    Imin: float,
    amplitude: float,
    r: float,
) -> np.ndarray:
    return Imin + amplitude * np.exp(-r * relative_year)
def predict(model_key: str, relative_year: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    if model_key == "nofloor":
        return exp_nofloor(relative_year, *parameters)
    if model_key == "floor":
        return exp_floor(relative_year, *parameters)
    raise ValueError(f"Unknown model: {model_key}")
def _initial_guesses(model_key: str, y: np.ndarray) -> Iterable[List[float]]:
    y0 = float(y[0])
    ymin = float(np.min(y))
    y_range = float(np.ptp(y))
    scale = max(abs(y0), abs(ymin), y_range, 1e-9)
    rate_guesses = [0.002, 0.005, 0.01, 0.03, 0.08]
    if model_key == "nofloor":
        for rate in rate_guesses:
            yield [max(y0, 1e-9), rate]
        return
    floor_guesses = [
        0.0,
        max(0.0, 0.25 * ymin),
        max(0.0, 0.50 * ymin),
        max(0.0, 0.80 * ymin),
        max(0.0, 0.95 * ymin),
    ]
    for floor_guess in floor_guesses:
        amplitude_guess = max(y0 - floor_guess, 0.05 * scale, 1e-9)
        for rate in rate_guesses:
            yield [floor_guess, amplitude_guess, rate]
def fit_parameters(
    relative_year: np.ndarray,
    y: np.ndarray,
    model_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    relative_year = np.asarray(relative_year, dtype=float)
    y = np.asarray(y, dtype=float)
    if model_key == "nofloor":
        function = exp_nofloor
        lower_bounds = [0.0, 0.0]
        upper_bounds = [np.inf, MAX_DECLINE_RATE]
    elif model_key == "floor":
        function = exp_floor
        lower_bounds = [0.0, 0.0, 0.0]
        upper_bounds = [np.inf, np.inf, MAX_DECLINE_RATE]
    else:
        raise ValueError(f"Unknown model: {model_key}")
    best_parameters: np.ndarray | None = None
    best_covariance: np.ndarray | None = None
    best_sse = np.inf
    errors: List[str] = []
    for guess in _initial_guesses(model_key, y):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                parameters, covariance = curve_fit(
                    function,
                    relative_year,
                    y,
                    p0=guess,
                    bounds=(lower_bounds, upper_bounds),
                    method="trf",
                    x_scale="jac",
                    maxfev=100000,
                )
            fitted = function(relative_year, *parameters)
            sse = float(np.sum((y - fitted) ** 2))
            if np.isfinite(sse) and sse < best_sse:
                best_sse = sse
                best_parameters = np.asarray(parameters, dtype=float)
                best_covariance = np.asarray(covariance, dtype=float)
        except Exception as exc:
            errors.append(str(exc))
    if best_parameters is None or best_covariance is None:
        error_text = " | ".join(errors[-5:])
        raise RuntimeError(f"{model_key} model fitting failed. Most recent error: {error_text}")
    return best_parameters, best_covariance
def calculate_metrics(
    y: np.ndarray,
    fitted: np.ndarray,
    parameter_count: int,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    n = len(y)
    residual = y - fitted
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    variance_mle = max(sse / n, np.finfo(float).tiny)
    rmse = float(np.sqrt(sse / n))
    mae = float(np.mean(np.abs(residual)))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    aic = float(n * np.log(variance_mle) + 2 * parameter_count)
    if n <= parameter_count + 1:
        aicc = np.inf
    else:
        aicc = float(
            aic
            + 2 * parameter_count * (parameter_count + 1)
            / (n - parameter_count - 1)
        )
    bic = float(n * np.log(variance_mle) + parameter_count * np.log(n))
    return {
        "n": float(n),
        "SSE": sse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "AICc": aicc,
        "BIC": bic,
    }
def lag1_autocorrelation(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    if len(residual) < 3:
        return np.nan
    x = residual[:-1]
    y = residual[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])
def durbin_watson(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    denominator = float(np.sum(residual ** 2))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.diff(residual) ** 2) / denominator)
def standard_errors(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance).astype(float)
    diagonal = np.where(diagonal >= 0, diagonal, np.nan)
    return np.sqrt(diagonal)
def transformed_parameter_values(result: Dict[str, object]) -> Dict[str, float]:
    parameters = np.asarray(result["parameters"], dtype=float)
    covariance = np.asarray(result["covariance"], dtype=float)
    if result["model_key"] == "nofloor":
        errors = standard_errors(covariance)
        return {
            "I0": float(parameters[0]),
            "SE_I0": float(errors[0]),
            "r": float(parameters[1]),
            "SE_r": float(errors[1]),
        }
    Imin, amplitude, rate = parameters
    errors = standard_errors(covariance)
    variance_I0 = float(
        covariance[0, 0]
        + covariance[1, 1]
        + 2.0 * covariance[0, 1]
    )
    se_I0 = np.sqrt(variance_I0) if variance_I0 >= 0 else np.nan
    return {
        "I0": float(Imin + amplitude),
        "SE_I0": float(se_I0),
        "Imin": float(Imin),
        "SE_Imin": float(errors[0]),
        "A": float(amplitude),
        "SE_A": float(errors[1]),
        "r": float(rate),
        "SE_r": float(errors[2]),
    }
def parameter_string(result: Dict[str, object]) -> str:
    p = transformed_parameter_values(result)
    if result["model_key"] == "nofloor":
        return (
            f"I0={p['I0']:.5f}±{p['SE_I0']:.5f}, "
            f"r={p['r']:.5f}±{p['SE_r']:.5f}"
        )
    return (
        f"I0={p['I0']:.5f}±{p['SE_I0']:.5f}, "
        f"Imin={p['Imin']:.5f}±{p['SE_Imin']:.5f}, "
        f"r={p['r']:.5f}±{p['SE_r']:.5f}"
    )
def floor_identifiability(
    floor_result: Dict[str, object],
    y: np.ndarray,
) -> Dict[str, object]:
    p = transformed_parameter_values(floor_result)
    Imin = p["Imin"]
    se = p["SE_Imin"]
    scale = max(float(np.mean(np.abs(y))), float(np.ptp(y)), 1e-12)
    near_zero_boundary = Imin <= 1e-6 * scale
    if np.isfinite(se):
        ci_low = Imin - 1.96 * se
        ci_high = Imin + 1.96 * se
        relative_se = se / max(abs(Imin), 1e-15)
    else:
        ci_low = np.nan
        ci_high = np.nan
        relative_se = np.inf
    identifiable = bool(
        np.isfinite(se)
        and not near_zero_boundary
        and ci_low > 0
        and relative_se <= 1.0
    )
    reasons: List[str] = []
    if not np.isfinite(se):
        reasons.append("Imin standard error is unavailable")
    if near_zero_boundary:
        reasons.append("Imin is close to the zero boundary")
    if np.isfinite(ci_low) and ci_low <= 0:
        reasons.append("The approximate 95% interval for Imin includes 0")
    if relative_se > 1.0:
        reasons.append("The relative standard error of Imin exceeds 100%")
    if not reasons:
        reasons.append("Imin is identifiable under this diagnostic")
    return {
        "identifiable": identifiable,
        "Imin": Imin,
        "SE_Imin": se,
        "CI95_low": ci_low,
        "CI95_high": ci_high,
        "relative_SE": relative_se,
        "reason": "; ".join(reasons),
    }
def holdout_rmse(
    years: np.ndarray,
    y: np.ndarray,
    start_year: int,
    model_key: str,
) -> float:
    years = np.asarray(years, dtype=int)
    y = np.asarray(y, dtype=float)
    train_mask = (years >= start_year) & (years < VALID_START)
    test_mask = (years >= VALID_START) & (years <= VALID_END)
    if np.sum(train_mask) <= (3 if model_key == "floor" else 2):
        return np.nan
    if np.sum(test_mask) == 0:
        return np.nan
    train_relative_year = years[train_mask].astype(float) - float(start_year)
    test_relative_year = years[test_mask].astype(float) - float(start_year)
    parameters, _ = fit_parameters(
        train_relative_year,
        y[train_mask],
        model_key,
    )
    test_prediction = predict(model_key, test_relative_year, parameters)
    return float(np.sqrt(np.mean((y[test_mask] - test_prediction) ** 2)))
def fit_result(
    years: np.ndarray,
    y: np.ndarray,
    start_year: int,
    model_key: str,
    label: str,
) -> Dict[str, object]:
    years = np.asarray(years, dtype=int)
    y = np.asarray(y, dtype=float)
    mask = years >= start_year
    fitted_years = years[mask]
    fitted_y = y[mask]
    relative_year = fitted_years.astype(float) - float(start_year)
    parameters, covariance = fit_parameters(relative_year, fitted_y, model_key)
    fitted_values = predict(model_key, relative_year, parameters)
    residual = fitted_y - fitted_values
    parameter_count = 2 if model_key == "nofloor" else 3
    result_metrics = calculate_metrics(fitted_y, fitted_values, parameter_count)
    result: Dict[str, object] = {
        "label": label,
        "model_key": model_key,
        "start_year": int(start_year),
        "years": fitted_years,
        "observed": fitted_y,
        "relative_year": relative_year,
        "parameters": parameters,
        "covariance": covariance,
        "fitted": fitted_values,
        "residual": residual,
        "holdout_RMSE": holdout_rmse(
            years=years,
            y=y,
            start_year=start_year,
            model_key=model_key,
        ),
        "lag1_autocorrelation": lag1_autocorrelation(residual),
        "Durbin_Watson": durbin_watson(residual),
        **result_metrics,
    }
    return result
def result_row(result: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "Model": result["label"],
        "Fit start year": int(result["start_year"]),
        "Parameters": parameter_string(result),
        "RMSE": float(result["RMSE"]),
        "MAE": float(result["MAE"]),
        "R2": float(result["R2"]),
        "AICc": float(result["AICc"]),
        "BIC": float(result["BIC"]),
        "Holdout_RMSE_2010_2024": float(result["holdout_RMSE"]),
        "Residual_lag1_autocorrelation": float(result["lag1_autocorrelation"]),
        "Durbin_Watson": float(result["Durbin_Watson"]),
    }
    if result["model_key"] == "floor":
        diagnostics = floor_identifiability(
            result,
            np.asarray(result["observed"], dtype=float),
        )
        row["Imin_identifiable"] = diagnostics["identifiable"]
        row["Imin_95CI_lower"] = diagnostics["CI95_low"]
        row["Imin_95CI_upper"] = diagnostics["CI95_high"]
        row["Imin_diagnostic"] = diagnostics["reason"]
    else:
        row["Imin_identifiable"] = np.nan
        row["Imin_95CI_lower"] = np.nan
        row["Imin_95CI_upper"] = np.nan
        row["Imin_diagnostic"] = "Not applicable"
    return row
def compare_models(
    years: np.ndarray,
    y: np.ndarray,
    start_year: int,
    variable_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    results = {
        "nofloor": fit_result(
            years,
            y,
            start_year,
            "nofloor",
            "No floor",
        ),
        "floor": fit_result(
            years,
            y,
            start_year,
            "floor",
            "With floor",
        ),
    }
    table = pd.DataFrame([result_row(results["nofloor"]), result_row(results["floor"])])
    table = table.set_index("Model")
    print(f"\n===== {variable_name}: same-sample model comparison ({start_year}--{int(np.max(years))}) =====")
    print(table.round(5).to_string())
    return table, results
def choose_model(
    results: Dict[str, Dict[str, object]],
    y: np.ndarray,
) -> Tuple[str, str]:
    nofloor = results["nofloor"]
    floor = results["floor"]
    floor_diagnostic = floor_identifiability(floor, y)
    delta_aicc = float(nofloor["AICc"] - floor["AICc"])
    nofloor_holdout = float(nofloor["holdout_RMSE"])
    floor_holdout = float(floor["holdout_RMSE"])
    if not floor_diagnostic["identifiable"]:
        reason = (
            "Imin in the With floor model is not identifiable ("
            f"{floor_diagnostic['reason']}); therefore, the No floor model is selected according to the principle of parsimony."
        )
        return "nofloor", reason
    holdout_guard_passes = (
        not np.isfinite(nofloor_holdout)
        or not np.isfinite(floor_holdout)
        or floor_holdout <= HOLDOUT_TOLERANCE * nofloor_holdout
    )
    if delta_aicc >= DELTA_AICC_THRESHOLD and holdout_guard_passes:
        reason = (
            f"The AICc of the With floor model is lower than that of the No floor model by {delta_aicc:.2f}, and holdout performance does not deteriorate materially; therefore, the With floor model is selected."
        )
        return "floor", reason
    if delta_aicc < DELTA_AICC_THRESHOLD:
        reason = (
            f"The AICc improvement of the With floor model is only {delta_aicc:.2f}, which does not reach the preset threshold of {DELTA_AICC_THRESHOLD:.1f}; therefore, the No floor model is selected according to the principle of parsimony."
        )
    else:
        reason = (
            "Although the With floor model has a lower in-sample AICc, its holdout RMSE is substantially worse, "
            "so the No floor model is selected."
        )
    return "nofloor", reason
def equation_string(
    variable_symbol: str,
    result: Dict[str, object],
) -> str:
    start_year = int(result["start_year"])
    p = transformed_parameter_values(result)
    if result["model_key"] == "nofloor":
        return (
            f"{variable_symbol}(t) = {p['I0']:.5f} * "
            f"exp(-{p['r']:.6f} * (t-{start_year}))"
        )
    return (
        f"{variable_symbol}(t) = {p['Imin']:.5f} + {p['A']:.5f} * "
        f"exp(-{p['r']:.6f} * (t-{start_year}))"
    )
def add_validation_shading(axis: plt.Axes, maximum_year: int) -> None:
    if maximum_year >= VALID_START:
        axis.axvspan(
            VALID_START,
            min(VALID_END, maximum_year),
            alpha=0.08,
            label="Holdout period (2010–2024)",
        )
def plot_results(
    years: np.ndarray,
    energy: np.ndarray,
    carbon: np.ndarray,
    energy_peak_year: int,
    energy_full_results: Dict[str, Dict[str, object]],
    energy_peak_results: Dict[str, Dict[str, object]],
    energy_chosen_key: str,
    carbon_results: Dict[str, Dict[str, object]],
    carbon_chosen_key: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    maximum_year = int(np.max(years))
    ax = axes[0, 0]
    ax.plot(years, energy, "o-", markersize=3.5, linewidth=1.0, label="Observed data")
    for key, linestyle, alpha in [
        ("nofloor", "--", 0.45),
        ("floor", ":", 0.45),
    ]:
        result = energy_full_results[key]
        ax.plot(
            result["years"],
            result["fitted"],
            linestyle=linestyle,
            linewidth=1.6,
            alpha=alpha,
            label=f"{result['label']} (full period; context only)",
        )
    for key, linestyle in [("nofloor", "--"), ("floor", "-")]:
        result = energy_peak_results[key]
        linewidth = 2.8 if key == energy_chosen_key else 1.8
        suffix = " — selected" if key == energy_chosen_key else ""
        ax.plot(
            result["years"],
            result["fitted"],
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{result['label']} from {energy_peak_year}{suffix}",
        )
    ax.axvline(energy_peak_year, linestyle="-.", linewidth=1.1, label="Fitting start")
    add_validation_shading(ax, maximum_year)
    ax.set_title("Energy intensity $I_E$: model comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("kWh / 2015 USD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.axhline(0, linewidth=0.9)
    for key, linestyle in [("nofloor", "--"), ("floor", "-")]:
        result = energy_peak_results[key]
        linewidth = 2.3 if key == energy_chosen_key else 1.5
        suffix = " — selected" if key == energy_chosen_key else ""
        ax.plot(
            result["years"],
            result["residual"],
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{result['label']} residuals{suffix}",
        )
    add_validation_shading(ax, maximum_year)
    ax.set_title(f"Energy intensity residuals ({energy_peak_year}–{maximum_year})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Observed − fitted (kWh/USD)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(years, carbon, "o-", markersize=3.5, linewidth=1.0, label="Observed data")
    for key, linestyle in [("nofloor", "--"), ("floor", "-")]:
        result = carbon_results[key]
        linewidth = 2.8 if key == carbon_chosen_key else 1.8
        suffix = " — selected" if key == carbon_chosen_key else ""
        ax.plot(
            result["years"],
            result["fitted"],
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{result['label']}{suffix}",
        )
    add_validation_shading(ax, maximum_year)
    ax.set_title("Carbon intensity $I_C$: model comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("kg CO$_2$ / kWh")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.axhline(0, linewidth=0.9)
    for key, linestyle in [("nofloor", "--"), ("floor", "-")]:
        result = carbon_results[key]
        linewidth = 2.3 if key == carbon_chosen_key else 1.5
        suffix = " — selected" if key == carbon_chosen_key else ""
        ax.plot(
            result["years"],
            result["residual"],
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"{result['label']} residuals{suffix}",
        )
    add_validation_shading(ax, maximum_year)
    ax.set_title("Carbon intensity residuals")
    ax.set_xlabel("Year")
    ax.set_ylabel("Observed − fitted (kg CO$_2$/kWh)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
def build_prediction_table(
    years: np.ndarray,
    observed: np.ndarray,
    results: Dict[str, Dict[str, object]],
    selected_key: str,
    variable_prefix: str,
) -> pd.DataFrame:
    start_year = int(results["nofloor"]["start_year"])
    mask = years >= start_year
    output = pd.DataFrame(
        {
            "year": years[mask],
            f"{variable_prefix}_observed": observed[mask],
        }
    )
    for key in ["nofloor", "floor"]:
        result = results[key]
        output[f"{variable_prefix}_{key}_fitted"] = result["fitted"]
        output[f"{variable_prefix}_{key}_residual"] = result["residual"]
    output[f"{variable_prefix}_selected_model"] = selected_key
    output["is_holdout_period"] = output["year"].between(VALID_START, VALID_END)
    return output
def main() -> None:
    data_path = locate_data_file()
    output_dir = make_output_directory()
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")
    data = read_and_validate_data(data_path)
    verify_intensity_construction(data)
    years = data["year"].to_numpy(dtype=int)
    energy = data["energy_intensity_kwh_per_usd"].to_numpy(dtype=float)
    carbon = data["carbon_intensity_kgco2_per_kwh"].to_numpy(dtype=float)
    first_year = int(np.min(years))
    last_year = int(np.max(years))
    energy_peak_index = int(np.argmax(energy))
    energy_peak_year = int(years[energy_peak_index])
    print(
        f"\nAutomatically detected historical peak year of energy intensity = {energy_peak_year}; this year is used as the fitting start of the sustained decline period."
    )
    energy_full_table, energy_full_results = compare_models(
        years,
        energy,
        first_year,
        "Energy intensity I_E (full period; start-year sensitivity reference only)",
    )
    energy_peak_table, energy_peak_results = compare_models(
        years,
        energy,
        energy_peak_year,
        "Energy intensity I_E (formal post-peak comparison)",
    )
    energy_chosen_key, energy_reason = choose_model(
        energy_peak_results,
        energy[years >= energy_peak_year],
    )
    energy_final = energy_peak_results[energy_chosen_key]
    start_sensitivity_rows = []
    for result_name, result in [
        (f"Full-period No floor ({first_year} start)", energy_full_results["nofloor"]),
        (f"Post-peak No floor ({energy_peak_year} start)", energy_peak_results["nofloor"]),
        (f"Post-peak With floor ({energy_peak_year} start)", energy_peak_results["floor"]),
    ]:
        row = result_row(result)
        row["Specification"] = result_name
        row["Interpretation"] = (
            "Different start years produce different sample sizes, so AICc, training RMSE, and R-squared are not used for formal selection across start years; "
            "holdout RMSE is used only as evidence of start-year sensitivity."
        )
        start_sensitivity_rows.append(row)
    energy_start_table = pd.DataFrame(start_sensitivity_rows).set_index("Specification")
    carbon_table, carbon_results = compare_models(
        years,
        carbon,
        first_year,
        "Carbon intensity I_C (formal full-period comparison)",
    )
    carbon_chosen_key, carbon_reason = choose_model(carbon_results, carbon)
    carbon_final = carbon_results[carbon_chosen_key]
    print("\n===== Final Energy-Intensity Specification =====")
    print(f"Selected model: {energy_final['label']} ({energy_peak_year}--{last_year})")
    print(f"Reason: {energy_reason}")
    print(f"Parameters: {parameter_string(energy_final)}")
    print(
        f"RMSE={energy_final['RMSE']:.5f}  "
        f"MAE={energy_final['MAE']:.5f}  "
        f"R2={energy_final['R2']:.5f}  "
        f"AICc={energy_final['AICc']:.2f}  "
        f"Holdout RMSE={energy_final['holdout_RMSE']:.5f}  "
        f"lag1={energy_final['lag1_autocorrelation']:.5f}"
    )
    print("Final equation: " + equation_string("I_E", energy_final))
    print("\n===== Final Carbon-Intensity Specification =====")
    print(f"Selected model: {carbon_final['label']} ({first_year}--{last_year})")
    print(f"Reason: {carbon_reason}")
    print(f"Parameters: {parameter_string(carbon_final)}")
    print(
        f"RMSE={carbon_final['RMSE']:.5f}  "
        f"MAE={carbon_final['MAE']:.5f}  "
        f"R2={carbon_final['R2']:.5f}  "
        f"AICc={carbon_final['AICc']:.2f}  "
        f"Holdout RMSE={carbon_final['holdout_RMSE']:.5f}  "
        f"lag1={carbon_final['lag1_autocorrelation']:.5f}"
    )
    print("Final equation: " + equation_string("I_C", carbon_final))
    print("\n===== Residual Interpretation =====")
    print(
        "A clearly positive lag-1 autocorrelation and a Durbin--Watson statistic well below 2 indicate that the model mainly captures "
        "the long-run declining trend rather than persistent short-run fluctuations. Parameter standard errors should therefore be interpreted cautiously."
    )
    energy_full_table.to_csv(
        output_dir / "01_energy_full_period_comparison.csv",
        encoding="utf-8-sig",
    )
    energy_peak_table.to_csv(
        output_dir / "02_energy_peak_period_comparison.csv",
        encoding="utf-8-sig",
    )
    energy_start_table.to_csv(
        output_dir / "03_energy_start_sensitivity.csv",
        encoding="utf-8-sig",
    )
    carbon_table.to_csv(
        output_dir / "04_carbon_model_comparison.csv",
        encoding="utf-8-sig",
    )
    energy_predictions = build_prediction_table(
        years,
        energy,
        energy_peak_results,
        energy_chosen_key,
        "energy_intensity",
    )
    energy_predictions.to_csv(
        output_dir / "05_energy_predictions_and_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    carbon_predictions = build_prediction_table(
        years,
        carbon,
        carbon_results,
        carbon_chosen_key,
        "carbon_intensity",
    )
    carbon_predictions.to_csv(
        output_dir / "06_carbon_predictions_and_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection_summary = pd.DataFrame(
        [
            {
                "Variable": "Energy intensity I_E",
                "Fitting period": f"{energy_peak_year}--{last_year}",
                "Selected model": energy_final["label"],
                "Selection reason": energy_reason,
                "Parameters": parameter_string(energy_final),
                "Final equation": equation_string("I_E", energy_final),
                "RMSE": energy_final["RMSE"],
                "MAE": energy_final["MAE"],
                "R2": energy_final["R2"],
                "AICc": energy_final["AICc"],
                "Holdout_RMSE_2010_2024": energy_final["holdout_RMSE"],
                "Residual_lag1_autocorrelation": energy_final["lag1_autocorrelation"],
                "Durbin_Watson": energy_final["Durbin_Watson"],
            },
            {
                "Variable": "Carbon intensity I_C",
                "Fitting period": f"{first_year}--{last_year}",
                "Selected model": carbon_final["label"],
                "Selection reason": carbon_reason,
                "Parameters": parameter_string(carbon_final),
                "Final equation": equation_string("I_C", carbon_final),
                "RMSE": carbon_final["RMSE"],
                "MAE": carbon_final["MAE"],
                "R2": carbon_final["R2"],
                "AICc": carbon_final["AICc"],
                "Holdout_RMSE_2010_2024": carbon_final["holdout_RMSE"],
                "Residual_lag1_autocorrelation": carbon_final["lag1_autocorrelation"],
                "Durbin_Watson": carbon_final["Durbin_Watson"],
            },
        ]
    )
    selection_summary.to_csv(
        output_dir / "07_final_model_selection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = output_dir / "intensity_model_selection_corrected.png"
    plot_results(
        years=years,
        energy=energy,
        carbon=carbon,
        energy_peak_year=energy_peak_year,
        energy_full_results=energy_full_results,
        energy_peak_results=energy_peak_results,
        energy_chosen_key=energy_chosen_key,
        carbon_results=carbon_results,
        carbon_chosen_key=carbon_chosen_key,
        output_path=figure_path,
    )
    print("\n===== Saved Files =====")
    for path in sorted(output_dir.iterdir()):
        print(path)
    print("\n===== Results-Section Conclusion Template =====")
    print(
        f"1) Energy intensity: Because the series did not show a sustained decline before {energy_peak_year}, "
        f"the model is fitted from the historical peak in {energy_peak_year}. {energy_reason} "
        f"The final model is {equation_string('I_E', energy_final)}."
    )
    print(
        f"2) Carbon intensity: The two models are compared over the same sample from {first_year}--{last_year}. "
        f"{carbon_reason} The final model is {equation_string('I_C', carbon_final)}."
    )
    print(
        "3) Residuals from both models should be interpreted together with lag-1 autocorrelation: if autocorrelation is high, "
        "the model should be treated as a long-run trend model rather than a model of year-to-year fluctuations."
    )
if __name__ == "__main__":
    main()
