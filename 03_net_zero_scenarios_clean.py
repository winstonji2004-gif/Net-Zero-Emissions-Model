from __future__ import annotations
import os
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from scipy.optimize import brentq
BASE_DIR = pathlib.Path(
    os.environ.get("MATH3041_DATA_DIR", r"F:/MATH3041_NetZero_Cleaned_Data")
)
OUT_DIR = BASE_DIR / "scenario_results_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INCLUDE_NATURAL_CDR = True
FUTURE_NATURAL_CDR_MODE = "constant_2024"
FUTURE_ARTIFICIAL_CDR_MODE = "iam_median"
IE_I0 = 3.234943
IE_R = 0.010501
IE_BASE_YEAR = 1970
IC_IMIN = 0.222726
IC_A = 0.041719
IC_R = 0.040895
IC_BASE_YEAR = 1965
TCRE = {
    "low": 0.27,
    "central": 0.45,
    "high": 0.63,
}
SCENARIO1_GROWTH_END = 2040
SCENARIO1_PLATFORM_YEAR = 2050
SCENARIO2_NET_ZERO_YEAR = 2080
SCENARIO3_ANNUAL_NET_ZERO_YEAR = 2050
SCENARIO3_INTEGRATED_NET_ZERO_YEAR = 2100
SEA_LEVEL_SMOOTHING_WINDOW = 5
for font_name in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
def find_col(df: pd.DataFrame, text: str) -> str:
    for col in df.columns:
        if text.lower() in str(col).lower():
            return str(col)
    raise KeyError(f"Cannot find a column containing: {text!r}")
def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path
def annual_numeric_series(
    df: pd.DataFrame,
    year_col: str,
    value_col: str,
    start_year: int,
    end_year: int,
) -> pd.Series:
    temp = pd.DataFrame(
        {
            "year": pd.to_numeric(df[year_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna()
    temp["year"] = temp["year"].astype(int)
    series = temp.groupby("year")["value"].mean().sort_index()
    full_index = pd.Index(np.arange(start_year, end_year + 1), name="year")
    return series.reindex(full_index).interpolate(limit_direction="both")
def metric_at_year(values: np.ndarray, years: np.ndarray, year: int) -> float:
    idx = np.where(years == year)[0]
    if len(idx) != 1:
        raise ValueError(f"Year {year} is not uniquely present in the year array.")
    return float(values[idx[0]])
def save_assumptions() -> None:
    assumptions = pd.DataFrame(
        [
            ["include_natural_cdr", INCLUDE_NATURAL_CDR, "Historical natural CDR included or excluded"],
            ["future_natural_cdr_mode", FUTURE_NATURAL_CDR_MODE, "Future natural CDR baseline assumption"],
            ["future_artificial_cdr_mode", FUTURE_ARTIFICIAL_CDR_MODE, "Future artificial/novel CDR baseline assumption"],
            ["IE_I0", IE_I0, "Energy-intensity initial value"],
            ["IE_R", IE_R, "Energy-intensity decline rate"],
            ["IC_IMIN", IC_IMIN, "Carbon-intensity asymptote"],
            ["IC_A", IC_A, "Carbon-intensity exponential amplitude"],
            ["IC_R", IC_R, "Carbon-intensity decline rate"],
            ["TCRE_low_degC_per_1000Gt", TCRE["low"], "Low TCRE sensitivity"],
            ["TCRE_central_degC_per_1000Gt", TCRE["central"], "Central TCRE sensitivity"],
            ["TCRE_high_degC_per_1000Gt", TCRE["high"], "High TCRE sensitivity"],
            ["sea_level_smoothing_window", SEA_LEVEL_SMOOTHING_WINDOW, "Causal trailing mean window in years"],
        ],
        columns=["parameter", "value", "description"],
    )
    assumptions.to_csv(OUT_DIR / "04_model_assumptions.csv", index=False, encoding="utf-8-sig")
print("Data folder:", BASE_DIR)
print("Output folder:", OUT_DIR)
cal_path = require_file(BASE_DIR / "01_calibration_input_data_1965_2024.csv")
cal = pd.read_csv(cal_path)
year_h = pd.to_numeric(cal[find_col(cal, "year")], errors="raise").to_numpy(dtype=int)
P_h = pd.to_numeric(cal[find_col(cal, "population_people")], errors="raise").to_numpy(dtype=float)
g_h = pd.to_numeric(cal[find_col(cal, "gdp_per_capita")], errors="raise").to_numpy(dtype=float)
IE_obs = pd.to_numeric(cal[find_col(cal, "energy_intensity")], errors="raise").to_numpy(dtype=float)
IC_obs = pd.to_numeric(cal[find_col(cal, "carbon_intensity")], errors="raise").to_numpy(dtype=float)
E_obs = pd.to_numeric(cal[find_col(cal, "fossil_co2")], errors="raise").to_numpy(dtype=float)
B_obs = pd.to_numeric(cal[find_col(cal, "cumulative_fossil_co2")], errors="raise").to_numpy(dtype=float)
T_obs = pd.to_numeric(cal[find_col(cal, "temperature")], errors="raise").to_numpy(dtype=float)
if year_h[-1] != 2024:
    raise ValueError(f"Expected the historical data to end in 2024, but it ends in {year_h[-1]}.")
R_nat_h = np.zeros(len(year_h), dtype=float)
R_art_h = np.zeros(len(year_h), dtype=float)
nat_path = require_file(
    BASE_DIR
    / "MATH3041_Natural_CDR_Data_20260730"
    / "MATH3041_Natural_CDR_Data_20260730"
    / "01_DATA"
    / "natural_cdr_model_input_global_2005_2024.csv"
)
nat = pd.read_csv(nat_path)
for _, row in nat.iterrows():
    yy = int(row["year"])
    mask = year_h == yy
    if np.any(mask):
        R_nat_h[mask] += float(row["natural_cdr_mtco2_per_year"])
art_hist_path = require_file(
    BASE_DIR
    / "MATH3041_Artificial_CDR_Data_20260730"
    / "01_DATA"
    / "artificial_cdr_historical_global_2017_2025.csv"
)
art_hist = pd.read_csv(art_hist_path)
for _, row in art_hist.iterrows():
    yy = int(row["year"])
    if 2017 <= yy <= 2022:
        mask = year_h == yy
        if np.any(mask):
            R_art_h[mask] += float(row["artificial_cdr_mtco2_per_year"])
art_model_path = require_file(
    BASE_DIR
    / "MATH3041_Artificial_CDR_Data_20260730"
    / "01_DATA"
    / "artificial_cdr_model_input_global_2023_2030.csv"
)
art_model = pd.read_csv(art_model_path)
for _, row in art_model.iterrows():
    yy = int(row["year"])
    if (
        str(row.get("data_status", "")) == "actual_delivered_cdr"
        and 2023 <= yy <= 2024
    ):
        mask = year_h == yy
        if np.any(mask):
            R_art_h[mask] += float(row["artificial_cdr_mtco2_per_year"])
R_h = R_art_h + (R_nat_h if INCLUDE_NATURAL_CDR else 0.0)
E_net_obs = E_obs - R_h
B_net_obs = B_obs - np.cumsum(R_h) / 1000.0
E24 = float(E_net_obs[-1])
B24 = float(B_net_obs[-1])
T24 = float(T_obs[-1])
years_f = np.arange(2025, 2101, dtype=int)
un_path = require_file(BASE_DIR / "07_un_wpp_population_projection_2025_2100.csv")
un_df = pd.read_csv(un_path)
un_year_col = find_col(un_df, "year")
un_value_col = find_col(un_df, "population")
un_series = annual_numeric_series(un_df, un_year_col, un_value_col, 2025, 2100)
P_f = P_h[-1] * un_series.loc[years_f].to_numpy(dtype=float) / float(un_series.loc[2025])
oecd_path = require_file(
    BASE_DIR / "OECD.ECO.MAD,DSD_EO_LTB@DF_EO_LTB,+W.GDPVTRD_CAP.BAU1.A.csv"
)
oecd_df = pd.read_csv(oecd_path)
oecd_df = oecd_df.assign(year=pd.to_numeric(oecd_df["TIME_PERIOD"], errors="coerce"))
oecd_series = annual_numeric_series(oecd_df, "year", "OBS_VALUE", 2024, 2100)
if 2024 in oecd_series.index and np.isfinite(oecd_series.loc[2024]):
    g_anchor = float(oecd_series.loc[2024])
else:
    g_anchor = float(oecd_series.loc[2025])
g_f = g_h[-1] * oecd_series.loc[years_f].to_numpy(dtype=float) / g_anchor
IE_f = IE_I0 * np.exp(-IE_R * (years_f - IE_BASE_YEAR))
IC_f = IC_IMIN + IC_A * np.exp(-IC_R * (years_f - IC_BASE_YEAR))
E_gross_f = P_f * g_f * IE_f * IC_f / 1e9
iam_path = require_file(
    BASE_DIR
    / "MATH3041_Artificial_CDR_Data_20260730"
    / "01_DATA"
    / "artificial_cdr_scenarios_global_2030_2100.csv"
)
iam = pd.read_csv(iam_path)
iam["year"] = pd.to_numeric(iam["year"], errors="coerce")
iam["artificial_cdr_mtco2_per_year"] = pd.to_numeric(
    iam["artificial_cdr_mtco2_per_year"], errors="coerce"
)
iam = iam.dropna(subset=["year", "artificial_cdr_mtco2_per_year"])
iam["year"] = iam["year"].astype(int)
iam_group = iam.groupby("year")["artificial_cdr_mtco2_per_year"]
iam_median_sparse = iam_group.median().sort_index()
iam_min_sparse = iam_group.min().sort_index()
iam_max_sparse = iam_group.max().sort_index()
art_2024 = float(R_art_h[-1])
def interpolate_with_2024_anchor(sparse: pd.Series, anchor_2024: float) -> np.ndarray:
    points = pd.concat(
        [pd.Series({2024: anchor_2024}, dtype=float), sparse.astype(float)]
    ).groupby(level=0).last().sort_index()
    return np.interp(
        years_f.astype(float),
        points.index.to_numpy(dtype=float),
        points.to_numpy(dtype=float),
    )
if FUTURE_ARTIFICIAL_CDR_MODE != "iam_median":
    raise ValueError("Only FUTURE_ARTIFICIAL_CDR_MODE='iam_median' is implemented.")
R_art_base_f = interpolate_with_2024_anchor(iam_median_sparse, art_2024)
R_art_iam_min_f = interpolate_with_2024_anchor(iam_min_sparse, art_2024)
R_art_iam_max_f = interpolate_with_2024_anchor(iam_max_sparse, art_2024)
if not INCLUDE_NATURAL_CDR or FUTURE_NATURAL_CDR_MODE == "zero":
    R_nat_base_f = np.zeros(len(years_f), dtype=float)
elif FUTURE_NATURAL_CDR_MODE == "constant_2024":
    R_nat_base_f = np.full(len(years_f), float(R_nat_h[-1]), dtype=float)
else:
    raise ValueError(
        "FUTURE_NATURAL_CDR_MODE must be 'constant_2024' or 'zero'."
    )
R_base_f = R_art_base_f + R_nat_base_f
E_net_base_f = E_gross_f - R_base_f
cdr_reference = pd.DataFrame(
    {
        "year": years_f,
        "baseline_natural_CDR_Mt": R_nat_base_f,
        "baseline_artificial_CDR_Mt": R_art_base_f,
        "baseline_total_CDR_Mt": R_base_f,
        "IAM_artificial_CDR_min_Mt": R_art_iam_min_f,
        "IAM_artificial_CDR_median_Mt": R_art_base_f,
        "IAM_artificial_CDR_max_Mt": R_art_iam_max_f,
        "baseline_net_emissions_Mt": E_net_base_f,
    }
)
cdr_reference.to_csv(
    OUT_DIR / "03_cdr_baseline_and_iam_reference.csv",
    index=False,
    encoding="utf-8-sig",
)
res_temp_emp = stats.linregress(B_net_obs, T_obs)
alpha_emp = float(res_temp_emp.slope)
intercept_emp = float(res_temp_emp.intercept)
sl_path = require_file(BASE_DIR / "sea-level.csv")
sl = pd.read_csv(sl_path)
sl["year"] = sl["Day"].astype(str).str[:4].astype(int)
H_col = "Average of Church and White (2011) and UHSLC"
H_ann = sl.groupby("year")[H_col].mean().sort_index()
years_sl = np.array([y for y in H_ann.index if 1965 <= y <= 2020], dtype=int)
H_obs = H_ann.loc[years_sl].to_numpy(dtype=float)
T_by_year = pd.Series(T_obs, index=year_h)
T_sl = T_by_year.loc[years_sl].to_numpy(dtype=float)
sea_cal = pd.DataFrame({"year": years_sl, "H": H_obs, "T": T_sl})
sea_cal["H_smooth"] = sea_cal["H"].rolling(
    SEA_LEVEL_SMOOTHING_WINDOW,
    min_periods=SEA_LEVEL_SMOOTHING_WINDOW,
).mean()
sea_cal["T_smooth"] = sea_cal["T"].rolling(
    SEA_LEVEL_SMOOTHING_WINDOW,
    min_periods=SEA_LEVEL_SMOOTHING_WINDOW,
).mean()
sea_cal["dH_smooth"] = sea_cal["H_smooth"].diff()
sea_fit = sea_cal.dropna(subset=["T_smooth", "dH_smooth"])
res_sl = stats.linregress(sea_fit["T_smooth"], sea_fit["dH_smooth"])
a_sl = float(res_sl.slope)
if abs(a_sl) < 1e-12:
    raise RuntimeError("Sea-level regression slope is too close to zero to define T0.")
T0_sl = float(-res_sl.intercept / a_sl)
sea_rate_r2 = float(res_sl.rvalue**2)
H_bridge_years = np.arange(int(years_sl[-1]) + 1, 2025, dtype=int)
H_bridge_values = []
H_prev = float(H_obs[-1])
for yy in H_bridge_years:
    temp_y = float(T_by_year.loc[yy])
    H_prev = H_prev + a_sl * (temp_y - T0_sl)
    H_bridge_values.append(H_prev)
H_bridge_values = np.asarray(H_bridge_values, dtype=float)
H24 = float(H_bridge_values[-1]) if len(H_bridge_values) else float(H_obs[-1])
warnings.warn(
    "The sea-level projection is exploratory. Interpret the rate-regression "
    f"R2={sea_rate_r2:.3f} and parameter sensitivity, not only the integrated path.",
    RuntimeWarning,
)
E_net_2005 = metric_at_year(E_net_obs, year_h, 2005)
if E_net_2005 <= 0 or E24 <= 0:
    raise ValueError("Scenario 1 CAGR requires positive net emissions in 2005 and 2024.")
g0 = (E24 / E_net_2005) ** (1.0 / (2024 - 2005)) - 1.0
growth_path_s1 = np.zeros(len(years_f), dtype=float)
for i, yy in enumerate(years_f):
    if yy <= SCENARIO1_GROWTH_END:
        growth_path_s1[i] = g0
    elif yy <= SCENARIO1_PLATFORM_YEAR:
        growth_path_s1[i] = g0 * (
            (SCENARIO1_PLATFORM_YEAR - yy)
            / (SCENARIO1_PLATFORM_YEAR - SCENARIO1_GROWTH_END)
        )
    else:
        growth_path_s1[i] = 0.0
E_target_s1 = np.empty(len(years_f), dtype=float)
previous = E24
for i, growth in enumerate(growth_path_s1):
    previous = previous * (1.0 + growth)
    E_target_s1[i] = previous
E_target_s2 = E24 * np.clip(
    (SCENARIO2_NET_ZERO_YEAR - years_f)
    / (SCENARIO2_NET_ZERO_YEAR - 2024),
    0.0,
    None,
)
E_target_s3_pre = E24 * np.clip(
    (SCENARIO3_ANNUAL_NET_ZERO_YEAR - years_f)
    / (SCENARIO3_ANNUAL_NET_ZERO_YEAR - 2024),
    0.0,
    None,
)
def realise_target(E_target_f: np.ndarray) -> dict[str, np.ndarray]:
    E_target_f = np.asarray(E_target_f, dtype=float)
    R_extra_f = np.maximum(E_net_base_f - E_target_f, 0.0)
    R_total_f = R_base_f + R_extra_f
    E_actual_f = E_gross_f - R_total_f
    identity_error = E_gross_f - R_total_f - E_actual_f
    if np.max(np.abs(identity_error)) > 1e-8:
        raise RuntimeError("Emissions accounting identity failed.")
    return {
        "target": E_target_f,
        "extra_cdr": R_extra_f,
        "total_cdr": R_total_f,
        "actual_net": E_actual_f,
    }
def cumulative_2100_for_s3(c_mt_per_yr2: float) -> float:
    target = np.where(
        years_f <= SCENARIO3_ANNUAL_NET_ZERO_YEAR,
        E_target_s3_pre,
        -c_mt_per_yr2 * (years_f - SCENARIO3_ANNUAL_NET_ZERO_YEAR),
    )
    realised = realise_target(target)
    return B24 + float(np.sum(realised["actual_net"]) / 1000.0)
if cumulative_2100_for_s3(0.0) <= 0.0:
    c_s3 = 0.0
else:
    c_upper = 1000.0
    while cumulative_2100_for_s3(c_upper) > 0.0 and c_upper < 1e7:
        c_upper *= 2.0
    if cumulative_2100_for_s3(c_upper) > 0.0:
        raise RuntimeError(
            "Could not bracket the Scenario 3 solution. Check the baseline and units."
        )
    c_s3 = float(
        brentq(cumulative_2100_for_s3, 0.0, c_upper, xtol=1e-10, rtol=1e-12)
    )
E_target_s3 = np.where(
    years_f <= SCENARIO3_ANNUAL_NET_ZERO_YEAR,
    E_target_s3_pre,
    -c_s3 * (years_f - SCENARIO3_ANNUAL_NET_ZERO_YEAR),
)
scenario_targets = {
    "Scenario 1: net-zero rate of change": E_target_s1,
    "Scenario 2: net-zero annual emissions": E_target_s2,
    "Scenario 3: integrated net zero": E_target_s3,
}
def integrate_future(E_actual_f: np.ndarray) -> dict[str, np.ndarray]:
    E_actual_f = np.asarray(E_actual_f, dtype=float)
    B_f = B24 + np.cumsum(E_actual_f) / 1000.0
    delta_B = B_f - B24
    T_f = {
        key: T24 + (value / 1000.0) * delta_B
        for key, value in TCRE.items()
    }
    H_f = np.empty(len(years_f), dtype=float)
    previous_H = H24
    for i in range(len(years_f)):
        previous_H = previous_H + a_sl * (T_f["central"][i] - T0_sl)
        H_f[i] = previous_H
    return {
        "B": B_f,
        "T_low": T_f["low"],
        "T_central": T_f["central"],
        "T_high": T_f["high"],
        "H": H_f,
    }
series: dict[str, np.ndarray] = {
    "year": years_f,
    "population_people": P_f,
    "gdp_per_capita": g_f,
    "energy_intensity": IE_f,
    "carbon_intensity": IC_f,
    "E_gross_Mt": E_gross_f,
    "baseline_natural_CDR_Mt": R_nat_base_f,
    "baseline_artificial_CDR_Mt": R_art_base_f,
    "baseline_total_CDR_Mt": R_base_f,
    "baseline_net_emissions_Mt": E_net_base_f,
}
summary_rows: list[dict[str, float | int | str]] = []
scenario_results: dict[str, dict[str, np.ndarray]] = {}
for scenario_name, target_path in scenario_targets.items():
    realised = realise_target(target_path)
    climate = integrate_future(realised["actual_net"])
    scenario_results[scenario_name] = {**realised, **climate}
    prefix = scenario_name.split(":")[0].replace(" ", "_").lower()
    series[f"{prefix}_target_net_Mt"] = realised["target"]
    series[f"{prefix}_actual_net_Mt"] = realised["actual_net"]
    series[f"{prefix}_additional_CDR_Mt"] = realised["extra_cdr"]
    series[f"{prefix}_total_CDR_Mt"] = realised["total_cdr"]
    series[f"{prefix}_B_Gt"] = climate["B"]
    series[f"{prefix}_T_low_C"] = climate["T_low"]
    series[f"{prefix}_T_central_C"] = climate["T_central"]
    series[f"{prefix}_T_high_C"] = climate["T_high"]
    series[f"{prefix}_H_exploratory_mm"] = climate["H"]
    peak_idx = int(np.argmax(climate["T_central"]))
    identity_rmse = float(
        np.sqrt(
            np.mean(
                (
                    E_gross_f
                    - realised["total_cdr"]
                    - realised["actual_net"]
                )
                ** 2
            )
        )
    )
    summary_rows.append(
        {
            "scenario": scenario_name,
            "target_Enet_2050_Gt_per_yr": metric_at_year(realised["target"], years_f, 2050) / 1000.0,
            "actual_Enet_2050_Gt_per_yr": metric_at_year(realised["actual_net"], years_f, 2050) / 1000.0,
            "actual_Enet_2080_Gt_per_yr": metric_at_year(realised["actual_net"], years_f, 2080) / 1000.0,
            "actual_Enet_2100_Gt_per_yr": metric_at_year(realised["actual_net"], years_f, 2100) / 1000.0,
            "additional_CDR_2050_Gt_per_yr": metric_at_year(realised["extra_cdr"], years_f, 2050) / 1000.0,
            "additional_CDR_2080_Gt_per_yr": metric_at_year(realised["extra_cdr"], years_f, 2080) / 1000.0,
            "additional_CDR_2100_Gt_per_yr": metric_at_year(realised["extra_cdr"], years_f, 2100) / 1000.0,
            "total_CDR_2100_Gt_per_yr": metric_at_year(realised["total_cdr"], years_f, 2100) / 1000.0,
            "cumulative_additional_CDR_2025_2100_Gt": float(np.sum(realised["extra_cdr"]) / 1000.0),
            "B_2100_Gt": float(climate["B"][-1]),
            "T_2050_central_C": metric_at_year(climate["T_central"], years_f, 2050),
            "T_2100_low_C": float(climate["T_low"][-1]),
            "T_2100_central_C": float(climate["T_central"][-1]),
            "T_2100_high_C": float(climate["T_high"][-1]),
            "T_peak_central_C": float(np.max(climate["T_central"])),
            "T_peak_year": int(years_f[peak_idx]),
            "H_2050_exploratory_mm": metric_at_year(climate["H"], years_f, 2050),
            "H_2100_exploratory_mm": float(climate["H"][-1]),
            "accounting_identity_RMSE_Mt": identity_rmse,
            "scenario3_negative_slope_Gt_per_yr2": (
                c_s3 / 1000.0
                if scenario_name == "Scenario 3: integrated net zero"
                else np.nan
            ),
        }
    )
summary_df = pd.DataFrame(summary_rows)
series_df = pd.DataFrame(series)
summary_df.to_csv(
    OUT_DIR / "01_scenario_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
series_df.to_csv(
    OUT_DIR / "02_scenario_series_2025_2100.csv",
    index=False,
    encoding="utf-8-sig",
)
save_assumptions()
print("\n" + "=" * 92)
print("Historical and parameter diagnostics")
print(f"Observed 2024 net emissions: {E24 / 1000.0:.3f} GtCO2/yr")
print(f"Observed 2024 cumulative net emissions: {B24:.3f} GtCO2")
print(f"Observed 2024 temperature anomaly: {T24:.3f} degC")
print(
    "Historical empirical temperature coefficient: "
    f"{alpha_emp * 1000.0:.3f} degC/1000 GtCO2 "
    "(diagnostic only)"
)
print(
    "Future TCRE assumptions: "
    f"low={TCRE['low']:.2f}, central={TCRE['central']:.2f}, "
    f"high={TCRE['high']:.2f} degC/1000 GtCO2"
)
print(
    "Sea-level exploratory calibration: "
    f"a={a_sl:.4f} mm/yr/degC, T0={T0_sl:.4f} degC, "
    f"rate R2={sea_rate_r2:.4f}"
)
print("\n" + "=" * 92)
print("Scenario summary")
with pd.option_context("display.max_columns", None, "display.width", 240):
    print(summary_df.round(3).to_string(index=False))
print(
    "\nScenario 3 solved negative-emissions slope: "
    f"c={c_s3 / 1000.0:.4f} GtCO2/yr^2"
)
print(
    "Scenario 3 achieved cumulative net emissions in 2100: "
    f"{scenario_results['Scenario 3: integrated net zero']['B'][-1]:.8f} GtCO2"
)
for name, result in scenario_results.items():
    max_identity_error = float(
        np.max(np.abs(E_gross_f - result["total_cdr"] - result["actual_net"]))
    )
    target_not_binding = int(np.sum(result["actual_net"] < result["target"] - 1e-9))
    print(
        f"{name}: max accounting error={max_identity_error:.3e} MtCO2/yr; "
        f"baseline already below target in {target_not_binding} year(s)."
    )
colors = {
    "Scenario 1: net-zero rate of change": "#1f77b4",
    "Scenario 2: net-zero annual emissions": "#ff7f0e",
    "Scenario 3: integrated net zero": "#2ca02c",
}
fig, axes = plt.subplots(3, 2, figsize=(14, 13))
ax = axes[0, 0]
ax.plot(year_h, E_net_obs / 1000.0, "k-", lw=1.2, label="Historical net emissions")
for name, result in scenario_results.items():
    color = colors[name]
    ax.plot(years_f, result["actual_net"] / 1000.0, color=color, lw=2.1, label=name)
    ax.plot(years_f, result["target"] / 1000.0, color=color, lw=1.0, ls="--", alpha=0.65)
ax.axhline(0.0, color="black", lw=0.8)
ax.axvline(2050, color="gray", ls=":", lw=1)
ax.axvline(2080, color="gray", ls=":", lw=1)
ax.set_title("(a) Target and achieved net emissions")
ax.set_xlabel("Year")
ax.set_ylabel("GtCO$_2$ / yr")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[0, 1]
ax.plot(years_f, R_base_f / 1000.0, "k--", lw=1.5, label="Baseline total CDR")
for name, result in scenario_results.items():
    ax.plot(
        years_f,
        result["extra_cdr"] / 1000.0,
        color=colors[name],
        lw=2.1,
        label=name,
    )
ax.set_title("(b) Additional CDR required beyond baseline")
ax.set_xlabel("Year")
ax.set_ylabel("GtCO$_2$ / yr")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1, 0]
ax.plot(year_h, B_net_obs, "k-", lw=1.2, label="Historical cumulative net emissions")
for name, result in scenario_results.items():
    ax.plot(years_f, result["B"], color=colors[name], lw=2.1, label=name)
ax.axhline(0.0, color="black", lw=0.8)
ax.set_title("(c) Cumulative net emissions")
ax.set_xlabel("Year")
ax.set_ylabel("GtCO$_2$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1, 1]
ax.plot(year_h, T_obs, "k-", lw=1.2, label="Historical temperature (GISTEMP)")
for name, result in scenario_results.items():
    color = colors[name]
    lower = np.minimum(result["T_low"], result["T_high"])
    upper = np.maximum(result["T_low"], result["T_high"])
    ax.fill_between(years_f, lower, upper, color=color, alpha=0.10)
    ax.plot(years_f, result["T_central"], color=color, lw=2.1, label=name)
ax.set_title("(d) Temperature anomaly (central TCRE; shaded sensitivity)")
ax.set_xlabel("Year")
ax.set_ylabel(r"$^\circ$C")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[2, 0]
ax.plot(years_sl, H_obs, "k-", lw=1.2, label="Historical sea level")
if len(H_bridge_years):
    ax.plot(H_bridge_years, H_bridge_values, "k--", lw=1.1, label="2021-2024 bridge")
for name, result in scenario_results.items():
    ax.plot(years_f, result["H"], color=colors[name], lw=2.1, label=name)
ax.set_title(f"(e) Exploratory sea level (rate $R^2$={sea_rate_r2:.2f})")
ax.set_xlabel("Year")
ax.set_ylabel("mm")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[2, 1]
ax.plot(year_h, E_obs / 1000.0, "k-", lw=1.2, label="Historical fossil CO$_2$")
ax.plot(years_f, E_gross_f / 1000.0, "r-", lw=2.1, label="Projected gross emissions")
ax.plot(
    years_f,
    E_net_base_f / 1000.0,
    color="gray",
    lw=1.8,
    ls="--",
    label="Baseline net emissions",
)
ax.set_title("(f) Kaya gross-emissions projection")
ax.set_xlabel("Year")
ax.set_ylabel("GtCO$_2$ / yr")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
figure_path = OUT_DIR / "scenarios_analysis_corrected.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print("\nSaved outputs")
for filename in [
    "01_scenario_summary.csv",
    "02_scenario_series_2025_2100.csv",
    "03_cdr_baseline_and_iam_reference.csv",
    "04_model_assumptions.csv",
    "scenarios_analysis_corrected.png",
]:
    print(OUT_DIR / filename)
