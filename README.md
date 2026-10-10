# Net-Zero Emissions Pathway Model

MATH3041 数学建模课程作业：从历史数据校准到净零情景与敏感性分析的完整建模流水线。

四个脚本可**直接运行**：默认读取本仓库 `data/` 下的数据，无需修改任何路径。

## 流水线

按编号顺序执行。每个脚本的输出是下一个脚本的输入。

| 文件 | 内容 | 主要产出 |
| --- | --- | --- |
| `01_fit_intensity_models.py` | 能源强度 $I_E$ 与碳强度 $I_C$ 的模型拟合与选择 | 以 `curve_fit` 做非线性最小二乘，在「无下限」与「带下限」两种指数形式间用 AICc + 留出验证选择；输出预测表与拟合图 |
| `02_calibrate_historical_kaya.py` | 历史期 Kaya 恒等式校准 | 校准排放、温度（$T = a_0 + \alpha B$）与海平面（$dH/dt = a(T-T_0)$）三组参数，含 2011–2024 留出验证 |
| `03_build_net_zero_scenarios.py` | 三种净零情景至 2100 年 | 构造并根据目标反解排放路径，换算累计排放、升温区间与海平面；输出情景对比图 |
| `04_run_sensitivity_analysis.py` | 关键参数敏感性分析 | 五个参数各 ±20%，输出对 2100 年温度与海平面的影响排序（龙卷风图） |

### 模型形式

```
I_E(t) = I_0 · exp(-r_E · (t - t_0))                  能源强度（1970 年起）
I_C(t) = I_C,min + A · exp(-r_C · (t - t_0))          碳强度（1965 年起，带下限）
E(t)   = P(t) · g(t) · I_E(t) · I_C(t) / 10^9         Kaya 恒等式
E_net  = E_gross - CDR                                 净排放
T(t)   = a_0 + α · B(t)                               温度（B 为累计净排放）
dH/dt  = a · (T - T_0)                                海平面
```

脚本 01 拟合出的参数（$I_0 = 3.23494$, $r_E = 0.010501$, $I_{C,min} = 0.22273$, $r_C = 0.040895$）
已作为常量写入脚本 02–04，因此四个脚本无需串联运行即可单独执行。

## 运行

```sh
python -m pip install -r requirements.txt
python 01_fit_intensity_models.py
python 02_calibrate_historical_kaya.py
python 03_build_net_zero_scenarios.py
python 04_run_sensitivity_analysis.py
```

Python 3.10 或更高版本。结果分别写入各脚本所在目录下的
`intensity_model_results/`、`calibration_results/`、`sensitivity_results/`，
以及 `data/MATH3041_NetZero_Cleaned_Data/scenario_results_corrected/`。

## 数据

- `data/MATH3041_NetZero_Cleaned_Data/`：历史校准数据（1965–2024）、UN WPP 人口预测（2025–2100）、
  OECD 长期 GDP 预测，以及海平面观测序列 `sea-level.csv`（Church and White (2011) / UHSLC）。
- `data/MATH3041_Carbon_Removal_Data_20260730/`：碳移除数据包，含原始官方 CSV、清洗后的模型输入
  与元数据（变量字典、来源清单、质量检查、转换日志）。来源与许可信息见该数据包
  `03_METADATA/source_manifest.csv`。

> 建模代码使用 `natural_cdr_*` / `artificial_cdr_*` 命名。数据包中对应列为
> `conventional_cdr_mtco2_per_year`（造林与森林管理）与 `novel_cdr_mtco2_per_year`（新型 CDR），
> 映射关系见 `03_METADATA/model_variable_mapping.csv`。各脚本读取时已做列名对齐。

## 路径配置

脚本默认使用仓库内的数据目录，可通过环境变量覆盖：

| 变量 | 作用 |
| --- | --- |
| `MATH3041_DATA_DIR` | 覆盖 `data/MATH3041_NetZero_Cleaned_Data` |
| `MATH3041_INTENSITY_DATA` | 仅脚本 01：直接指定强度数据 CSV |
| `MATH3041_INTENSITY_OUTPUT` | 仅脚本 01：指定输出目录 |

碳移除数据目录按 `MATH3041_DATA_DIR` 的同级目录 `MATH3041_Carbon_Removal_Data_20260730`
自动定位。
