# clean-python-codes

MATH3041 净零模型作业：Python 代码与现有数据。

## 代码

| 文件 | 内容 |
| --- | --- |
| `01_intensity_model_verification_clean.py` | 能源强度与碳强度模型验证 |
| `02_historical_calibration_clean.py` | 历史数据校准 |
| `03_net_zero_scenarios_clean.py` | 净零情景分析 |
| `04_sensitivity_analysis_clean.py` | 敏感性分析 |

四个脚本保留原始内容。

## 数据

- `data/MATH3041_NetZero_Cleaned_Data/`：7 个历史与人口预测 CSV，以及 OECD 长期 GDP 预测 CSV。
- `data/MATH3041_Carbon_Removal_Data_20260730/`：22 个碳移除 CSV，包括原始数据、清洗数据和来源、字典、质量检查等元数据。来源与许可信息保存在该数据包的元数据中。

## 依赖与路径

建议使用 Python 3.10 或更高版本，安装依赖：

```sh
python -m pip install -r requirements.txt
```

原始脚本仍使用本地数据路径，运行前需要配置或修改路径：

- 脚本 01 支持 `MATH3041_INTENSITY_DATA`，可设为本仓库的 `data/MATH3041_NetZero_Cleaned_Data/05_emissions_energy_1965_2024.csv`。
- 脚本 03 支持 `MATH3041_DATA_DIR`，可设为本仓库的 `data/MATH3041_NetZero_Cleaned_Data`。
- 脚本 02、04 的输入路径仍硬编码为 `F:/MATH3041_NetZero_Cleaned_Data`，需要按实际目录修改。
