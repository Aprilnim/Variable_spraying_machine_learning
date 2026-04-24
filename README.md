# Flow To Pulse Modeling

This project models:

- input: `avg_flow_l_min`
- output: `esc_us`
- filter: ignore rows where `avg_flow_l_min == 0`

using the dataset placed in the project folder:

- `flow_data_averages_summary.csv`

## Models Compared

- linear regression
- 2nd-order polynomial regression
- 3rd-order polynomial regression
- piecewise linear regression

## Split

- training: `0.8`
- validation: `0.2`

## Metrics

- `R^2`
- `MAE`
- `RMSE`

## Run

```powershell
cd "C:\Users\11523\Documents\New project\machine_learning"
python compare_flow_to_pulse_models.py
```

## Outputs

The script writes files into `outputs/`:

- `model_comparison_metrics.csv`
- `best_model_predictions_flow_to_pulse.csv`
- `best_model_fit_flow_to_pulse.png`
- `best_model_validation_parity.png`
- `model_comparison_report.txt`
