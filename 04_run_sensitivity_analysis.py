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
OUT_DIR = pathlib.Path(__file__).resolve().parent / "sensitivity_results"
OUT_DIR.mkdir(exist_ok=True)
print("Output directory:", OUT_DIR)
cal = pd.read_csv(DATA_DIR / "01_calibration_input_data_1965_2024.csv")
def fc(s):
    for c in cal.columns:
        if s.lower() in c.lower():
            return c
    raise KeyError(s)
year_h = cal[fc("year")].to_numpy(dtype=float)
P_h = cal[fc("population_people")].to_numpy(dtype=float)
g_h = cal[fc("gdp_per_capita")].to_numpy(dtype=float)
IE_obs = cal[fc("energy_intensity")].to_numpy(dtype=float)
IC_obs = cal[fc("carbon_intensity")].to_numpy(dtype=float)
E_obs = cal[fc("fossil_co2")].to_numpy(dtype=float)
B_obs = cal[fc("cumulative_fossil_co2")].to_numpy(dtype=float)
T_obs = cal[fc("temperature")].to_numpy(dtype=float)
R_h = np.zeros(len(year_h))
nat = pd.read_csv(CDR_DIR / "02_CLEANED_MODEL_INPUTS" / "socdr_conventional_cdr_global_2005_2024.csv")
nat["natural_cdr_mtco2_per_year"] = nat["conventional_cdr_mtco2_per_year"]  # cleaned package renamed this column
for _, r in nat.iterrows():
    R_h[year_h == int(r["year"])] += float(r["natural_cdr_mtco2_per_year"])
art = pd.read_csv(CDR_DIR / "02_CLEANED_MODEL_INPUTS" / "socdr_novel_cdr_global_2017_2025.csv")
art["artificial_cdr_mtco2_per_year"] = art["novel_cdr_mtco2_per_year"]
for _, r in art.iterrows():
    yy = int(r["year"])
    if 2017 <= yy <= 2022:
        R_h[year_h == yy] += float(r["artificial_cdr_mtco2_per_year"])
am = pd.read_csv(CDR_DIR / "02_CLEANED_MODEL_INPUTS" / "socdr_novel_cdr_pipeline_total_global_2023_2030.csv")
am["artificial_cdr_mtco2_per_year"] = am["novel_cdr_mtco2_per_year"]  # cleaned package renamed this column
for _, r in am.iterrows():
    yy = int(r["year"])
    if r["data_status"] == "actual_delivered_cdr" and 2023 <= yy <= 2024:
        R_h[year_h == yy] += float(r["artificial_cdr_mtco2_per_year"])
un = pd.read_csv(DATA_DIR / "07_un_wpp_population_projection_2025_2100.csv")
un = un.set_index("year")["un_wpp_population_people"]
oecd = pd.read_csv(DATA_DIR / "OECD.ECO.MAD,DSD_EO_LTB@DF_EO_LTB,+W.GDPVTRD_CAP.BAU1.A.csv")
oecd = oecd.assign(year=oecd["TIME_PERIOD"].astype(int)).set_index("year")["OBS_VALUE"].astype(float)
oecd = oecd[oecd.index >= 2020]
years_f = np.arange(2025, 2101)
P_f = P_h[-1] * un.loc[years_f].to_numpy(dtype=float) / un.loc[2025]
if 2024 in oecd.index:
    g_f = g_h[-1] * oecd.loc[years_f].to_numpy(dtype=float) / oecd.loc[2024]
else:
    g_f = g_h[-1] * oecd.loc[years_f].to_numpy(dtype=float) / oecd.loc[2025]
R_f = np.full(len(years_f), R_h[-1])
rE0, IE0, pk = 0.010501, 3.23494, 1970
Imin0, dI0, rC0 = 0.22273, 0.04172, 0.040895
B_net_obs = B_obs - np.cumsum(R_h) / 1000.0
res_t = stats.linregress(B_net_obs, T_obs)
alpha0, a0 = res_t.slope, res_t.intercept
sl = pd.read_csv(DATA_DIR / "sea-level.csv")
sl["year"] = sl["Day"].str[:4].astype(int)
H_ann = sl.groupby("year")["Average of Church and White (2011) and UHSLC"].mean()
years_sl = np.array([y for y in H_ann.index if 1965 <= y <= 2020])
H_obs = H_ann.loc[years_sl].to_numpy(dtype=float)
T_sl = T_obs[np.isin(year_h, years_sl)]
def ma5(x):
    return np.convolve(x, np.ones(5) / 5, mode="valid")
Hs, Ts = ma5(H_obs), ma5(T_sl)
res_sl = stats.linregress(Ts[:-1], np.diff(Hs))
a0_sl, T0_sl = res_sl.slope, -res_sl.intercept / res_sl.slope
print(f"Baseline parameters: rE={rE0}, rC={rC0}, Imin={Imin0}, alpha={alpha0:.6f}, a={a0_sl:.4f}, T0={T0_sl:.3f}")
def run(rE=rE0, Imin=Imin0, rC=rC0, alpha=alpha0, a=a0_sl, T0=T0_sl):
    yrs = np.arange(1965, 2101)
    IE = np.where(yrs >= pk, IE0 * np.exp(-rE * (yrs - pk)), np.nan)
    IE[:5] = IE_obs[:5]
    IC = Imin + dI0 * np.exp(-rC * (yrs - 1965))
    P = np.concatenate([P_h, P_f])
    g = np.concatenate([g_h, g_f])
    R = np.concatenate([R_h, R_f])
    Eg = P * g * IE * IC / 1e9
    En = Eg - R
    B = np.empty_like(yrs, dtype=float)
    B[0] = B_net_obs[0]
    for i in range(1, len(yrs)):
        B[i] = B[i-1] + En[i] / 1000.0
    T = a0 + alpha * B
    H = np.empty_like(yrs, dtype=float)
    H[0] = H_obs[0]
    for i in range(1, len(yrs)):
        H[i] = H[i-1] + a * (T[i] - T0)
    return yrs, Eg, B, T, H
yrs, Eg0, B0, T0b, H0 = run()
base = {
    "T2050": float(T0b[yrs == 2050][0]),
    "T2100": float(T0b[yrs == 2100][0]),
    "H2100": float(H0[yrs == 2100][0]),
    "E2050": float(Eg0[yrs == 2050][0]),
    "B2100": float(B0[yrs == 2100][0]),
}
print(f"\nBaseline scenario: T2050={base['T2050']:.2f} °C, T2100={base['T2100']:.2f} °C, "
      f"H2100={base['H2100']:.0f} mm, E2050={base['E2050']:.0f} Mt, B2100={base['B2100']:.0f} Gt")
params = {
    "r_E": (rE0, 0.010501),
    "r_C": (rC0, 0.040895),
    "I_C_min": (Imin0, 0.22273),
    "alpha": (alpha0, alpha0),
    "a": (a0_sl, a0_sl),
}
rows = []
for name, (val, _) in params.items():
    row = {"Parameter": name, "Baseline value": val}
    for pct, tag in [(-0.2, "low"), (0.2, "high")]:
        kw = dict(rE=rE0, Imin=Imin0, rC=rC0, alpha=alpha0, a=a0_sl, T0=T0_sl)
        if name == "r_E":
            kw["rE"] = rE0 * (1 + pct)
        elif name == "r_C":
            kw["rC"] = rC0 * (1 + pct)
        elif name == "I_C_min":
            kw["Imin"] = Imin0 * (1 + pct)
        elif name == "alpha":
            kw["alpha"] = alpha0 * (1 + pct)
        elif name == "a":
            kw["a"] = a0_sl * (1 + pct)
        yrs_, Eg_, B_, T_, H_ = run(**kw)
        row[f"T2050_{tag}"] = float(T_[yrs_ == 2050][0])
        row[f"T2100_{tag}"] = float(T_[yrs_ == 2100][0])
        row[f"H2100_{tag}"] = float(H_[yrs_ == 2100][0])
        row[f"E2050_{tag}"] = float(Eg_[yrs_ == 2050][0])
    row["T2100_range"] = row["T2100_high"] - row["T2100_low"]
    row["H2100_range"] = row["H2100_high"] - row["H2100_low"]
    rows.append(row)
tab = pd.DataFrame(rows)
print("\n===== Sensitivity Summary (+/-20%) =====")
print(tab.round(3).to_string(index=False))
tab.to_csv(OUT_DIR / "01_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
names = ["r_E", "r_C", "I_C_min", "alpha", "a"]
labels = ["r_E (energy intensity rate)", "r_C (carbon intensity rate)",
          "I_C min (carbon floor)", "alpha (temp response)", "a (sea-level coeff)"]
ax = axes[0]
lo = [tab.loc[tab["Parameter"] == n, "T2100_low"].iloc[0] - base["T2100"] for n in names]
hi = [tab.loc[tab["Parameter"] == n, "T2100_high"].iloc[0] - base["T2100"] for n in names]
ypos = np.arange(len(names))[::-1]
ax.barh(ypos, lo, left=base["T2100"], color="#4C72B0", alpha=0.85, label="-20%")
ax.barh(ypos, hi, left=base["T2100"], color="#C44E52", alpha=0.85, label="+20%")
ax.axvline(base["T2100"], color="k", lw=1)
ax.set_yticks(ypos); ax.set_yticklabels(labels)
ax.set_title("Sensitivity of 2100 temperature (°C)", fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
lo = [tab.loc[tab["Parameter"] == n, "H2100_low"].iloc[0] - base["H2100"] for n in names]
hi = [tab.loc[tab["Parameter"] == n, "H2100_high"].iloc[0] - base["H2100"] for n in names]
ax.barh(ypos, lo, left=base["H2100"], color="#4C72B0", alpha=0.85, label="-20%")
ax.barh(ypos, hi, left=base["H2100"], color="#C44E52", alpha=0.85, label="+20%")
ax.axvline(base["H2100"], color="k", lw=1)
ax.set_yticks(ypos); ax.set_yticklabels(labels)
ax.set_title("Sensitivity of 2100 sea level (mm)", fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
png = OUT_DIR / "sensitivity_tornado_2100.png"
fig.savefig(png, dpi=150)
print("\nFigure saved:", png)
rank = tab.sort_values("T2100_range", ascending=False)[["Parameter", "T2100_range", "H2100_range"]]
print("\n===== Sensitivity Ranking (largest to smallest effect on T2100) =====")
print(rank.round(3).to_string(index=False))
