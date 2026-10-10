import os
import numpy as np
import pandas as pd
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

# --- data location -------------------------------------------------------
# Resolves to the data bundled with this repository. Override with the
# MATH3041_DATA_DIR environment variable if your data lives elsewhere.
DATA_DIR = pathlib.Path(
    os.environ.get(
        "MATH3041_DATA_DIR",
        pathlib.Path(__file__).resolve().parent / "data" / "MATH3041_NetZero_Cleaned_Data",
    )
)
CDR_DIR = DATA_DIR.parent / "MATH3041_Carbon_Removal_Data_20260730"

for f in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
OUT_DIR = pathlib.Path(__file__).resolve().parent / "calibration_results"
OUT_DIR.mkdir(exist_ok=True)
print("Output directory:", OUT_DIR)
cal = pd.read_csv(DATA_DIR / "01_calibration_input_data_1965_2024.csv")
def fc(s):
    for c in cal.columns:
        if s.lower() in c.lower():
            return c
    raise KeyError(s)
year = cal[fc("year")].to_numpy(dtype=float)
P = cal[fc("population_people")].to_numpy(dtype=float)
g = cal[fc("gdp_per_capita")].to_numpy(dtype=float)
IE_obs = cal[fc("energy_intensity")].to_numpy(dtype=float)
IC_obs = cal[fc("carbon_intensity")].to_numpy(dtype=float)
E_obs = cal[fc("fossil_co2")].to_numpy(dtype=float)
B_obs = cal[fc("cumulative_fossil_co2")].to_numpy(dtype=float)
T_obs = cal[fc("temperature")].to_numpy(dtype=float)
R = np.zeros(len(year))
nat = pd.read_csv(CDR_DIR / "02_CLEANED_MODEL_INPUTS" / "socdr_conventional_cdr_global_2005_2024.csv")
nat["natural_cdr_mtco2_per_year"] = nat["conventional_cdr_mtco2_per_year"]  # cleaned package renamed this column
for _, r in nat.iterrows():
    R[year == int(r["year"])] += float(r["natural_cdr_mtco2_per_year"])
art = pd.read_csv(CDR_DIR / "02_CLEANED_MODEL_INPUTS" / "socdr_novel_cdr_global_2017_2025.csv")
art["artificial_cdr_mtco2_per_year"] = art["novel_cdr_mtco2_per_year"]
for _, r in art.iterrows():
    yy = int(r["year"])
    if 2017 <= yy <= 2022:
        R[year == yy] += float(r["artificial_cdr_mtco2_per_year"])
am = pd.read_csv(CDR_DIR / "02_CLEANED_MODEL_INPUTS" / "socdr_novel_cdr_pipeline_total_global_2023_2030.csv")
am["artificial_cdr_mtco2_per_year"] = am["novel_cdr_mtco2_per_year"]  # cleaned package renamed this column
for _, r in am.iterrows():
    yy = int(r["year"])
    if r["data_status"] == "actual_delivered_cdr" and 2023 <= yy <= 2024:
        R[year == yy] += float(r["artificial_cdr_mtco2_per_year"])
def IE_model(t_year):
    out = 3.23494 * np.exp(-0.010501 * (t_year - 1970.0))
    return np.where(t_year >= 1970, out, np.nan)
def IC_model(t_year):
    return 0.22273 + 0.04172 * np.exp(-0.040895 * (t_year - 1965.0))
IE_fit = IE_model(year)
IE_fit[year < 1970] = IE_obs[year < 1970]
IC_fit = IC_model(year)
def kaya(IE, IC):
    return P * g * IE * IC / 1e9
Eg = kaya(IE_fit, IC_fit)
En = Eg - R
B_model = np.empty_like(B_obs)
B_model[0] = B_obs[0]
for i in range(1, len(year)):
    B_model[i] = B_model[i-1] + En[i] / 1000.0
En_obs = E_obs - R
B_net_obs = B_obs - np.cumsum(R) / 1000.0
Eg_identity = kaya(IE_obs, IC_obs)
def metrics(obs, pred):
    n = len(obs)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    mae = float(np.mean(np.abs(obs - pred)))
    mape = float(np.mean(np.abs((obs - pred) / obs)) * 100)
    r2 = float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))
    bias = float(np.mean(pred - obs))
    return rmse, mae, mape, r2, bias
print("=" * 70)
print("[3. Historical Emissions Fit]")
for lab, obs, pred in [("E_gross", E_obs, Eg), ("E_net", En_obs, En), ("B_net(Gt)", B_net_obs, B_model)]:
    rmse, mae, mape, r2, bias = metrics(obs, pred)
    print(f"{lab}: RMSE={rmse:9.3f}  MAE={mae:8.3f}  MAPE={mape:5.2f}%  R2={r2:.5f}  Bias={bias:+8.3f}")
rmse_id, *_ = metrics(E_obs, Eg_identity)
print(f"Consistency check (using observed intensities): RMSE={rmse_id:.6f} (expected to be approximately 0)")
m_test = year > 2010
rmse_o, mae_o, mape_o, r2_o, bias_o = metrics(E_obs[m_test], Eg[m_test])
print(f"E_gross holdout validation (2011-2024): RMSE={rmse_o:.1f}  MAPE={mape_o:.2f}%  Bias={bias_o:+.1f}")
print("\n" + "=" * 70)
print("[4. Temperature Calibration: T = intercept + alpha * B]")
res = stats.linregress(B_net_obs, T_obs)
alpha, a0 = res.slope, res.intercept
T_model = a0 + alpha * B_model
rmse_t, mae_t, mape_t, r2_t, _ = metrics(T_obs, T_model)
print(f"alpha = {alpha:.6f} °C/GtCO2 (equivalent to {alpha*1000:.3f} °C/1000 GtCO2)")
print(f"Intercept = {a0:.4f} °C; regression R2 = {res.rvalue**2:.4f}")
print(f"In-sample: RMSE={rmse_t:.3f} °C, MAE={mae_t:.3f} °C, R2={r2_t:.4f}")
mask_tr = year <= 2010
res_tr = stats.linregress(B_net_obs[mask_tr], T_obs[mask_tr])
T_oos = res_tr.intercept + res_tr.slope * B_model[~mask_tr]
rmse_to, mae_to, _, r2_to, _ = metrics(T_obs[~mask_tr], T_oos)
print(f"Holdout validation (2011-2024): RMSE={rmse_to:.3f} °C, MAE={mae_to:.3f} °C, R2={r2_to:.4f}")
print("\n" + "=" * 70)
print("[5. Sea-Level Calibration: dH/dt = a*(T - T0)]")
sl = pd.read_csv(DATA_DIR / "sea-level.csv")
sl["year"] = sl["Day"].str[:4].astype(int)
H_ann = sl.groupby("year")["Average of Church and White (2011) and UHSLC"].mean()
years_h = np.array([y for y in H_ann.index if 1965 <= y <= 2020])
H = H_ann.loc[years_h].to_numpy(dtype=float)
T_h = T_obs[np.isin(year, years_h)]
def ma5(x):
    return np.convolve(x, np.ones(5) / 5, mode="valid")
Hs, Ts = ma5(H), ma5(T_h)
dH = np.diff(Hs)
Td = Ts[:-1]
res_sl = stats.linregress(Td, dH)
a_sl, b_sl = res_sl.slope, res_sl.intercept
T0_sl = -b_sl / a_sl
print(f"a = {a_sl:.4f} mm/yr/°C, T0 = {T0_sl:.3f} °C, rate-regression R2 = {res_sl.rvalue**2:.4f}")
H_model = np.empty_like(H)
H_model[0] = H[0]
for i in range(1, len(H)):
    H_model[i] = H_model[i-1] + a_sl * (T_h[i] - T0_sl)
rmse_h, mae_h, mape_h, r2_h, _ = metrics(H, H_model)
print(f"After integration: RMSE={rmse_h:.2f} mm, MAE={mae_h:.2f} mm, R2={r2_h:.4f} (historical rise approximately {H[-1]-H[0]:.0f} mm)")
m1 = (years_h >= 1970) & (years_h <= 2000)
m2 = years_h > 2000
H1s, T1s = ma5(H[m1]), ma5(T_h[m1])
res1 = stats.linregress(T1s[:-1], np.diff(H1s))
a1, T01 = res1.slope, -res1.intercept / res1.slope
H_oos = np.empty_like(H[m2])
H_oos[0] = H[m2][0]
for i in range(1, len(H[m2])):
    H_oos[i] = H_oos[i-1] + a1 * (T_h[m2][i] - T01)
rmse_ho, mae_ho, _, r2_ho, _ = metrics(H[m2], H_oos)
print(f"Holdout validation (2001-2020): RMSE={rmse_ho:.2f} mm, MAE={mae_ho:.2f} mm, R2={r2_ho:.4f}")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
ax.plot(year, E_obs, "ko-", ms=4, lw=1, label="Observed")
ax.plot(year, Eg, "r-", lw=1.8, label="Model (fitted intensities)")
ax.plot(year, Eg_identity, "g--", lw=1.5, label="Identity check (observed intensities)")
ax.set_title("Gross emissions E_gross", fontsize=12)
ax.set_xlabel("Year"); ax.set_ylabel("MtCO2 / yr"); ax.legend(); ax.grid(alpha=0.3)
ax = axes[0, 1]
ax.plot(year, En_obs, "ko-", ms=4, lw=1, label="Observed (fossil CO2 - R)")
ax.plot(year, En, "r-", lw=1.8, label="Model")
ax.set_title("Net emissions E_net", fontsize=12)
ax.set_xlabel("Year"); ax.set_ylabel("MtCO2 / yr"); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1, 0]
ax.plot(year, T_obs, "ko-", ms=4, lw=1, label="Observed (GISTEMP)")
ax.plot(year, T_model, "r-", lw=1.8, label="Model (alpha fitted)")
ax.set_title("Temperature anomaly", fontsize=12)
ax.set_xlabel("Year"); ax.set_ylabel("°C anomaly"); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1, 1]
ax.plot(years_h, H, "ko-", ms=4, lw=1, label="Observed (Church & White / UHSLC)")
ax.plot(years_h, H_model, "r-", lw=1.8, label="Model (integrated)")
ax.set_title("Sea level", fontsize=12)
ax.set_xlabel("Year"); ax.set_ylabel("mm"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
out = OUT_DIR / "historical_temperature_sealevel_calibration.png"
fig.savefig(out, dpi=150)
print("\nFigure saved:", out)
m_g = metrics(E_obs, Eg)
m_b = metrics(B_net_obs, B_model)
print("\n" + "=" * 70)
print("[Results Summary]")
print(f"E_gross: RMSE={m_g[0]:.0f} Mt, MAPE={m_g[2]:.2f}%")
print(f"B_net: RMSE={m_b[0]:.2f} Gt, MAPE={m_b[2]:.2f}%")
print(f"T: alpha={alpha*1000:.3f} °C/1000GtCO2, in-sample RMSE={rmse_t:.3f} °C, holdout RMSE={rmse_to:.3f} °C")
print(f"H: a={a_sl:.2f} mm/yr/°C, integrated RMSE={rmse_h:.1f} mm, holdout RMSE={rmse_ho:.1f} mm")
summary_rows = [
    {"Metric": "E_gross RMSE", "Value": round(m_g[0], 2), "Unit": "MtCO2/yr"},
    {"Metric": "E_gross MAPE", "Value": round(m_g[2], 2), "Unit": "%"},
    {"Metric": "E_gross R2", "Value": round(m_g[3], 5), "Unit": ""},
    {"Metric": "E_gross mean bias", "Value": round(m_g[4], 2), "Unit": "MtCO2/yr"},
    {"Metric": "E_gross holdout RMSE (2011-2024)", "Value": round(rmse_o, 2), "Unit": "MtCO2/yr"},
    {"Metric": "E_gross holdout MAPE (2011-2024)", "Value": round(mape_o, 2), "Unit": "%"},
    {"Metric": "E_net RMSE", "Value": round(metrics(En_obs, En)[0], 2), "Unit": "MtCO2/yr"},
    {"Metric": "E_net MAPE", "Value": round(metrics(En_obs, En)[2], 2), "Unit": "%"},
    {"Metric": "E_net R2", "Value": round(metrics(En_obs, En)[3], 5), "Unit": ""},
    {"Metric": "B_net RMSE", "Value": round(m_b[0], 3), "Unit": "GtCO2"},
    {"Metric": "B_net MAPE", "Value": round(m_b[2], 3), "Unit": "%"},
    {"Metric": "B_net R2", "Value": round(m_b[3], 6), "Unit": ""},
    {"Metric": "Consistency-check RMSE", "Value": round(rmse_id, 6), "Unit": "MtCO2/yr"},
    {"Metric": "Temperature alpha", "Value": round(alpha * 1000, 4), "Unit": "°C/1000 GtCO2"},
    {"Metric": "Temperature intercept", "Value": round(a0, 4), "Unit": "°C"},
    {"Metric": "Temperature regression R2", "Value": round(res.rvalue ** 2, 4), "Unit": ""},
    {"Metric": "Temperature in-sample RMSE", "Value": round(rmse_t, 3), "Unit": "°C"},
    {"Metric": "Temperature holdout RMSE (2011-2024)", "Value": round(rmse_to, 3), "Unit": "°C"},
    {"Metric": "Sea-level a", "Value": round(a_sl, 4), "Unit": "mm/yr/°C"},
    {"Metric": "Sea-level T0", "Value": round(T0_sl, 3), "Unit": "°C"},
    {"Metric": "Sea-level rate-regression R2", "Value": round(res_sl.rvalue ** 2, 4), "Unit": ""},
    {"Metric": "Sea-level integrated RMSE", "Value": round(rmse_h, 2), "Unit": "mm"},
    {"Metric": "Sea-level integrated R2", "Value": round(r2_h, 4), "Unit": ""},
    {"Metric": "Sea-level holdout RMSE (2001-2020)", "Value": round(rmse_ho, 2), "Unit": "mm"},
]
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "01_results_summary.csv", index=False, encoding="utf-8-sig")
pred_df = pd.DataFrame({
    "year": year.astype(int),
    "E_obs_mtco2": E_obs,
    "E_gross_model_mtco2": Eg,
    "E_net_obs_mtco2": En_obs,
    "E_net_model_mtco2": En,
    "B_obs_net_gtco2": B_net_obs,
    "B_model_gtco2": B_model,
    "T_obs_c": T_obs,
    "T_model_c": T_model,
})
pred_df.to_csv(OUT_DIR / "02_model_predictions.csv", index=False, encoding="utf-8-sig")
sl_df = pd.DataFrame({
    "year": years_h.astype(int),
    "H_obs_mm": H,
    "H_model_mm": H_model,
})
sl_df.to_csv(OUT_DIR / "03_sea_level_predictions.csv", index=False, encoding="utf-8-sig")
print("Result files saved to:", OUT_DIR)
for f in sorted(OUT_DIR.glob("*")):
    print("  -", f.name)
