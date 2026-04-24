from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "flow_data_averages_summary.csv"
OUTPUT_DIR = Path.home() / "ml_outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SEGMENT_SAMPLES = 5


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_polynomial_model(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )


def compute_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def evaluate_sklearn_model(model, x_train, x_test, y_train, y_test, model_name):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    return {
        "model_name": model_name,
        "model": model,
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
    }


class PiecewiseLinearModel:
    def __init__(self, breakpoint_value, left_model, right_model):
        self.breakpoint_value = breakpoint_value
        self.left_model = left_model
        self.right_model = right_model

    def predict(self, x_values):
        x_array = np.asarray(x_values).reshape(-1)
        predictions = np.zeros_like(x_array, dtype=float)

        left_mask = x_array <= self.breakpoint_value
        right_mask = ~left_mask

        if left_mask.any():
            predictions[left_mask] = self.left_model.predict(
                x_array[left_mask].reshape(-1, 1)
            )
        if right_mask.any():
            predictions[right_mask] = self.right_model.predict(
                x_array[right_mask].reshape(-1, 1)
            )

        return predictions


def fit_piecewise_linear(x_train, y_train):
    train_df = pd.DataFrame(
        {"avg_flow_l_min": x_train.reshape(-1), "esc_us": y_train.reshape(-1)}
    ).sort_values("avg_flow_l_min")

    candidate_breaks = sorted(train_df["avg_flow_l_min"].unique())
    best = None

    for breakpoint_value in candidate_breaks:
        left_df = train_df[train_df["avg_flow_l_min"] <= breakpoint_value]
        right_df = train_df[train_df["avg_flow_l_min"] > breakpoint_value]

        if len(left_df) < MIN_SEGMENT_SAMPLES or len(right_df) < MIN_SEGMENT_SAMPLES:
            continue

        left_model = LinearRegression()
        right_model = LinearRegression()

        left_model.fit(left_df[["avg_flow_l_min"]], left_df["esc_us"])
        right_model.fit(right_df[["avg_flow_l_min"]], right_df["esc_us"])

        piecewise_model = PiecewiseLinearModel(
            breakpoint_value=breakpoint_value,
            left_model=left_model,
            right_model=right_model,
        )

        predictions = piecewise_model.predict(train_df["avg_flow_l_min"].to_numpy())
        rmse = mean_squared_error(train_df["esc_us"], predictions) ** 0.5

        if best is None or rmse < best["train_rmse"]:
            best = {
                "model": piecewise_model,
                "breakpoint_value": breakpoint_value,
                "train_rmse": rmse,
            }

    if best is None:
        raise ValueError("Could not fit piecewise linear model with the current data.")

    return best["model"], best["breakpoint_value"]


def evaluate_piecewise_model(x_train, x_test, y_train, y_test):
    model, breakpoint_value = fit_piecewise_linear(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    return {
        "model_name": f"piecewise_linear_break_{breakpoint_value:.3f}",
        "model": model,
        "breakpoint_value": breakpoint_value,
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
    }


def save_metrics(results, output_dir):
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": result["model_name"],
                "train_r2": result["train_r2"],
                "test_r2": result["test_r2"],
                "train_mae": result["train_mae"],
                "test_mae": result["test_mae"],
                "train_rmse": result["train_rmse"],
                "test_rmse": result["test_rmse"],
            }
            for result in results
        ]
    )
    path = output_dir / "model_comparison_metrics.csv"
    metrics_df.to_csv(path, index=False)
    return path


def save_predictions(best_result, modeling_df, output_dir):
    prediction_df = modeling_df.copy()
    prediction_df["predicted_esc_us"] = best_result["model"].predict(
        prediction_df[["avg_flow_l_min"]].to_numpy()
    )
    path = output_dir / "best_model_predictions_flow_to_pulse.csv"
    prediction_df.to_csv(path, index=False)
    return path


def save_fit_plot(best_result, modeling_df, output_dir):
    x_curve = np.linspace(
        modeling_df["avg_flow_l_min"].min(),
        modeling_df["avg_flow_l_min"].max(),
        200,
    )
    y_curve = best_result["model"].predict(x_curve.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.scatter(
        modeling_df["avg_flow_l_min"],
        modeling_df["esc_us"],
        alpha=0.75,
        label="Measured data",
    )
    plt.plot(
        x_curve,
        y_curve,
        color="red",
        linewidth=2,
        label=f"Best fit: {best_result['model_name']}",
    )
    plt.xlabel("Flow (L/min)")
    plt.ylabel("ESC pulse width (us)")
    plt.title("Flow to ESC Pulse Width Model Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = output_dir / "best_model_fit_flow_to_pulse.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_validation_plot(best_result, x_test, y_test, output_dir):
    test_pred = best_result["model"].predict(x_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, test_pred, alpha=0.8)
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
    plt.xlabel("True ESC pulse width (us)")
    plt.ylabel("Predicted ESC pulse width (us)")
    plt.title(f"Validation Check: {best_result['model_name']}")
    plt.grid(True, alpha=0.3)
    path = output_dir / "best_model_validation_parity.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_report(results, best_result, output_dir, predictions_path, fit_plot_path, validation_plot_path, sample_count):
    report_lines = [
        "Flow -> ESC Pulse Width Modeling Report",
        f"Data file: {DATA_PATH}",
        "Filter: avg_flow_l_min > 0",
        f"Samples used: {sample_count}",
        f"Train/Test split: {1 - TEST_SIZE:.1f}/{TEST_SIZE:.1f}",
        "",
    ]

    for result in results:
        report_lines.extend(
            [
                f"{result['model_name']}:",
                f"  Train R2   : {result['train_r2']:.4f}",
                f"  Test R2    : {result['test_r2']:.4f}",
                f"  Train MAE  : {result['train_mae']:.4f}",
                f"  Test MAE   : {result['test_mae']:.4f}",
                f"  Train RMSE : {result['train_rmse']:.4f}",
                f"  Test RMSE  : {result['test_rmse']:.4f}",
                "",
            ]
        )

    report_lines.extend(
        [
            f"Best model by validation RMSE: {best_result['model_name']}",
            f"Metrics CSV: {output_dir / 'model_comparison_metrics.csv'}",
            f"Predictions CSV: {predictions_path}",
            f"Fit plot: {fit_plot_path}",
            f"Validation plot: {validation_plot_path}",
        ]
    )

    path = output_dir / "model_comparison_report.txt"
    path.write_text("\n".join(report_lines), encoding="utf-8")
    return path, report_lines


def main():
    output_dir = ensure_output_dir()

    df = pd.read_csv(DATA_PATH)
    df["avg_flow_l_min"] = pd.to_numeric(df["avg_flow_l_min"])
    df["esc_us"] = pd.to_numeric(df["esc_us"])

    df = df[df["avg_flow_l_min"] > 0].copy()

    modeling_df = df[["source_file", "avg_flow_l_min", "esc_us"]].copy()
    modeling_df = modeling_df.sort_values(["avg_flow_l_min", "esc_us"]).reset_index(drop=True)

    x = modeling_df[["avg_flow_l_min"]].to_numpy()
    y = modeling_df["esc_us"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []

    linear_model = LinearRegression()
    results.append(
        evaluate_sklearn_model(
            linear_model, x_train, x_test, y_train, y_test, "linear_regression"
        )
    )

    for degree in (2, 3):
        poly_model = build_polynomial_model(degree)
        results.append(
            evaluate_sklearn_model(
                poly_model,
                x_train,
                x_test,
                y_train,
                y_test,
                f"polynomial_degree_{degree}",
            )
        )

    results.append(evaluate_piecewise_model(x_train, x_test, y_train, y_test))

    best_result = min(results, key=lambda result: result["test_rmse"])

    metrics_path = save_metrics(results, output_dir)
    predictions_path = save_predictions(best_result, modeling_df, output_dir)
    fit_plot_path = save_fit_plot(best_result, modeling_df, output_dir)
    validation_plot_path = save_validation_plot(best_result, x_test, y_test, output_dir)
    report_path, report_lines = save_report(
        results,
        best_result,
        output_dir,
        predictions_path,
        fit_plot_path,
        validation_plot_path,
        len(modeling_df),
    )

    print("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
