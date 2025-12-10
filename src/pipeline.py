import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    from orbit.models import ETS
    from pygam import LinearGAM, s
    from econml.dml import LinearDML
    from prophet import Prophet
except ImportError:
    # Let the methods raise clearer errors if these are truly missing at runtime.
    ETS = None
    LinearGAM = None
    LinearDML = None
    Prophet = None


class SynapsePipeline:
    """
    Full Nike forecasting and causal analysis pipeline, adapted from forecast.py
    into a class with four main methods used by the agent:
      - load_and_prep
      - train_and_compare
      - run_causal_analysis
      - generate_forecast
    """

    def __init__(self, data_path: str = "data.xlsx") -> None:
        self.data_path = data_path
        self.df: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.future_data: pd.DataFrame | None = None

        # Trained models (set in train_and_compare)
        self.orbit_model = None
        self.prophet_model = None
        self.gam_trained = None

        self.X_features = [
            "log_consumer_exp",
            "log_gross_income",
            "log_plastic",
            "log_rubber",
            "scouring_agents",
            "colourants",
            "pop_growth",
        ]

        # Cached artifacts / metrics
        self.orbit_metrics: dict | None = None
        self.prophet_metrics: dict | None = None
        self.gam_metrics: dict | None = None
        self.best_model_name: str | None = None

        self.sensitivity_df: pd.DataFrame | None = None
        self.avg_rubber_effect: float | None = None
        self.avg_consumer_effect: float | None = None
        self.rubber_ci: tuple[float, float] | None = None
        self.consumer_ci: tuple[float, float] | None = None

        self.forecast_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # 1. DATA LOADING AND PREPROCESSING
    # ------------------------------------------------------------------
    def load_and_prep(self) -> str:
        """Load data, clean types, construct train / future splits, and log-features."""
        try:
            df = pd.read_excel(self.data_path)

            # Convert string columns that should be numeric
            for col in df.columns:
                if is_string_dtype(df[col]):
                    df[col] = df[col].astype(str).str.replace(",", "")
                    df[col] = pd.to_numeric(df[col], errors="ignore")

            # Nike sales to numeric
            df["nike"] = (
                df["nike"].astype(str).str.replace(",", "", regex=True).astype(float)
            )

            # Train vs future (where nike is NaN)
            train_data = df[df["nike"].notna()].copy()
            future_data = df[df["nike"].isna()].copy()

            # Log transforms
            train_data["log_nike"] = np.log1p(train_data["nike"])
            train_data["log_consumer_exp"] = np.log1p(
                train_data["consumer_expenditure_footwear"]
            )
            train_data["log_gross_income"] = np.log1p(train_data["gross_income"])
            train_data["log_plastic"] = np.log1p(train_data["plastic_prd"])
            train_data["log_rubber"] = np.log1p(train_data["rubber_price"])

            self.df = df
            self.train_data = train_data
            self.future_data = future_data

            return (
                "Data loaded and preprocessed.\n"
                f"- Training rows: {len(train_data)} (2016–2025)\n"
                f"- Forecast horizon rows: {len(future_data)} (2026–2030)\n"
                f"- Features: {', '.join(self.X_features)}"
            )
        except Exception as exc:
            return f"Error loading or preprocessing data: {exc}"

    # ------------------------------------------------------------------
    # 2. ROLLING FORECAST VALIDATION
    # ------------------------------------------------------------------
    def _rolling_forecast_validation(
        self, model_func, data: pd.DataFrame, n_splits: int = 3
    ) -> dict:
        """Time-series cross validation helper, identical to forecast.py logic."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mape_scores: list[float] = []
        mae_scores: list[float] = []
        rmse_scores: list[float] = []

        for train_idx, test_idx in tscv.split(data):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]

            predictions = model_func(train, test)

            mape = mean_absolute_percentage_error(test["nike"], predictions)
            mae = mean_absolute_error(test["nike"], predictions)
            rmse = np.sqrt(mean_squared_error(test["nike"], predictions))

            mape_scores.append(mape)
            mae_scores.append(mae)
            rmse_scores.append(rmse)

        return {
            "mape": float(np.mean(mape_scores)),
            "mae": float(np.mean(mae_scores)),
            "rmse": float(np.mean(rmse_scores)),
        }

    # ------------------------------------------------------------------
    # 3. ORBIT ETS MODEL
    # ------------------------------------------------------------------
    def _orbit_model_with_tuning(
        self, train: pd.DataFrame, test: pd.DataFrame | None = None, predict_future: bool = False
    ):
        if ETS is None:
            raise ImportError("orbit-ml is required for Orbit ETS modeling.")

        orbit_df = train[["year", "nike"]].copy()
        orbit_df.columns = ["ds", "y"]
        orbit_df["ds"] = pd.to_datetime(orbit_df["ds"], format="%Y")

        for feat in self.X_features:
            orbit_df[feat] = train[feat].values

        # Slightly reduced grid vs forecast.py for speed
        param_grid = {
            "num_sample": [1000, 2000],
            "seasonality": [1, 2],
        }

        best_model = None
        best_score = float("inf")
        best_params: dict | None = None

        for samp in param_grid["num_sample"]:
            for seas in param_grid["seasonality"]:
                ets = ETS(
                    response_col="y",
                    date_col="ds",
                    estimator="stan-mcmc",
                    seasonality=seas,
                    seed=8888,
                    num_warmup=400,
                    num_sample=samp,
                    stan_mcmc_args={"show_progress": True},
                )
                ets.fit(df=orbit_df)
                pred = ets.predict(df=orbit_df, decompose=True)
                score = mean_absolute_error(orbit_df["y"], pred["prediction"])

                if score < best_score:
                    best_score = score
                    best_model = ets
                    best_params = {"seasonality": seas, "num_sample": samp}

        if best_params is not None:
            print(f"Best Orbit parameters: {best_params}")

        if test is not None and not predict_future:
            test_df = test[["year", "nike"]].copy()
            test_df.columns = ["ds", "y"]
            test_df["ds"] = pd.to_datetime(test_df["ds"], format="%Y")
            for feat in self.X_features:
                test_df[feat] = test[feat].values
            pred = best_model.predict(test_df)
            return pred["prediction"].values

        return best_model

    # ------------------------------------------------------------------
    # 4. PROPHET MODEL
    # ------------------------------------------------------------------
    def _prophet_model_with_tuning(
        self, train: pd.DataFrame, test: pd.DataFrame | None = None, predict_future: bool = False
    ):
        if Prophet is None:
            raise ImportError("prophet is required for Prophet modeling.")

        prophet_df = train[["year", "nike"]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")

        param_grid = {
            "changepoint_prior_scale": [0.01, 0.1],
            "seasonality_prior_scale": [0.1, 1.0],
            "seasonality_mode": ["additive", "multiplicative"],
        }

        best_model = None
        best_score = float("inf")
        best_params: dict | None = None

        for cp_scale in param_grid["changepoint_prior_scale"]:
            for seas_scale in param_grid["seasonality_prior_scale"]:
                for seas_mode in param_grid["seasonality_mode"]:
                    model = Prophet(
                        changepoint_prior_scale=cp_scale,
                        seasonality_prior_scale=seas_scale,
                        seasonality_mode=seas_mode,
                        yearly_seasonality=True,
                        daily_seasonality=False,
                        weekly_seasonality=False,
                    )

                    for feat in self.X_features:
                        model.add_regressor(feat)

                    train_with_features = prophet_df.copy()
                    for feat in self.X_features:
                        train_with_features[feat] = train[feat].values

                    model.fit(train_with_features)
                    forecast = model.predict(train_with_features)
                    score = mean_absolute_error(prophet_df["y"], forecast["yhat"])

                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_params = {
                            "changepoint_prior_scale": cp_scale,
                            "seasonality_prior_scale": seas_scale,
                            "seasonality_mode": seas_mode,
                        }

        if best_params is not None:
            print(f"Best Prophet parameters: {best_params}")

        if test is not None and not predict_future:
            test_df = test[["year", "nike"]].copy()
            test_df.columns = ["ds", "y"]
            test_df["ds"] = pd.to_datetime(test_df["ds"], format="%Y")
            test_with_features = test_df.copy()
            for feat in self.X_features:
                test_with_features[feat] = test[feat].values
            forecast = best_model.predict(test_with_features)
            return forecast["yhat"].values

        return best_model

    # ------------------------------------------------------------------
    # 5. GAM MODEL
    # ------------------------------------------------------------------
    def _gam_model_with_tuning(
        self, train: pd.DataFrame, test: pd.DataFrame | None = None, predict_future: bool = False
    ):
        if LinearGAM is None:
            raise ImportError("pygam is required for GAM modeling.")

        X_train = train[self.X_features].values
        y_train = train["nike"].values

        param_grid = {
            "n_splines": [10, 20],
            "lam": [0.1, 1.0],
        }

        best_model = None
        best_score = float("inf")
        best_params: dict | None = None

        for n_spline in param_grid["n_splines"]:
            for lambda_val in param_grid["lam"]:
                terms = s(0, n_splines=n_spline, lam=lambda_val)
                for i in range(1, len(self.X_features)):
                    terms += s(i, n_splines=n_spline, lam=lambda_val)

                gam = LinearGAM(terms)
                gam.fit(X_train, y_train)

                pred = gam.predict(X_train)
                score = mean_absolute_error(y_train, pred)

                if score < best_score:
                    best_score = score
                    best_model = gam
                    best_params = {"n_splines": n_spline, "lam": lambda_val}

        if best_params is not None:
            print(f"Best GAM parameters: {best_params}")

        if test is not None and not predict_future:
            X_test = test[self.X_features].values
            return best_model.predict(X_test)

        return best_model

    # ------------------------------------------------------------------
    # PUBLIC METHOD: TRAIN + COMPARE
    # ------------------------------------------------------------------
    def train_and_compare(self) -> str:
        """Train Orbit, Prophet, and GAM, run rolling validation, and compare metrics."""
        if self.train_data is None:
            msg = self.load_and_prep()
            if msg.startswith("Error"):
                return msg

        data = self.train_data

        # Train final models (for reuse) and compute rolling metrics
        print("\n=== MODEL 1: Orbit ETS ===")
        self.orbit_model = self._orbit_model_with_tuning(data, predict_future=True)

        def orbit_rolling_func(train, test):
            return self._orbit_model_with_tuning(train, test, predict_future=False)

        self.orbit_metrics = self._rolling_forecast_validation(orbit_rolling_func, data)

        print("\n=== MODEL 2: Prophet ===")
        self.prophet_model = self._prophet_model_with_tuning(
            data, predict_future=True
        )

        def prophet_rolling_func(train, test):
            return self._prophet_model_with_tuning(train, test, predict_future=False)

        self.prophet_metrics = self._rolling_forecast_validation(
            prophet_rolling_func, data
        )

        print("\n=== MODEL 3: GAM ===")
        self.gam_trained = self._gam_model_with_tuning(data, predict_future=True)

        def gam_rolling_func(train, test):
            return self._gam_model_with_tuning(train, test, predict_future=False)

        self.gam_metrics = self._rolling_forecast_validation(gam_rolling_func, data)

        # Identify best model by MAPE
        metrics_table = {
            "Orbit ETS": self.orbit_metrics,
            "Prophet": self.prophet_metrics,
            "GAM": self.gam_metrics,
        }
        best_name = min(metrics_table, key=lambda m: metrics_table[m]["mape"])
        self.best_model_name = best_name

        out = [
            "MODEL PERFORMANCE COMPARISON (Rolling Forecast Validation)",
            f"- Orbit ETS: MAPE={self.orbit_metrics['mape']*100:.2f}%, "
            f"MAE=${self.orbit_metrics['mae']:.2f}, RMSE=${self.orbit_metrics['rmse']:.2f}",
            f"- Prophet:   MAPE={self.prophet_metrics['mape']*100:.2f}%, "
            f"MAE=${self.prophet_metrics['mae']:.2f}, RMSE=${self.prophet_metrics['rmse']:.2f}",
            f"- GAM:       MAPE={self.gam_metrics['mape']*100:.2f}%, "
            f"MAE=${self.gam_metrics['mae']:.2f}, RMSE=${self.gam_metrics['rmse']:.2f}",
            f"\n🏆 Best Model: {best_name} "
            f"with MAPE of {metrics_table[best_name]['mape']*100:.2f}%",
        ]

        return "\n".join(out)

    # ------------------------------------------------------------------
    # PUBLIC METHOD: CAUSAL ANALYSIS (DOUBLE ML + ELASTICITY)
    # ------------------------------------------------------------------
    def run_causal_analysis(self) -> str:
        """Run GAM-based elasticity and Double ML causal analyses."""
        if self.train_data is None:
            msg = self.load_and_prep()
            if msg.startswith("Error"):
                return msg
        if self.gam_trained is None or self.gam_metrics is None:
            _ = self.train_and_compare()

        if LinearDML is None:
            raise ImportError("econml is required for Double ML causal analysis.")

        train_data = self.train_data

        # --- GAM-based feature elasticity (Section 8) ---
        X_full = train_data[self.X_features].values
        y_full = train_data["nike"].values

        mean_vals = X_full.mean(axis=0)
        epsilon = 0.01

        sensitivity_results: list[dict] = []

        for i, feature in enumerate(self.X_features):
            X_perturbed = mean_vals.copy()
            X_perturbed[i] += epsilon

            y_base = self.gam_trained.predict(mean_vals.reshape(1, -1))[0]
            y_perturbed = self.gam_trained.predict(X_perturbed.reshape(1, -1))[0]

            partial_derivative = (y_perturbed - y_base) / epsilon
            elasticity = (partial_derivative * mean_vals[i]) / y_base
            impact_10pct = elasticity * 0.10

            sensitivity_results.append(
                {
                    "Feature": feature,
                    "Partial Derivative": partial_derivative,
                    "Elasticity": elasticity,
                    "Impact of 10% increase (%)": impact_10pct * 100,
                }
            )

        sensitivity_df = pd.DataFrame(sensitivity_results).sort_values(
            "Elasticity", ascending=False
        )
        self.sensitivity_df = sensitivity_df

        # --- Double ML: Rubber price impact ---
        X_confounders = train_data[
            [
                "log_consumer_exp",
                "log_gross_income",
                "log_plastic",
                "scouring_agents",
                "colourants",
                "pop_growth",
            ]
        ].values
        T_treatment = train_data["log_rubber"].values
        Y_outcome = train_data["nike"].values

        dml_rubber = LinearDML(
            model_y=RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            model_t=RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            discrete_treatment=False,
            cv=3,
        )
        dml_rubber.fit(Y_outcome, T_treatment, X=X_confounders)

        rubber_effect = dml_rubber.effect(X_confounders)
        avg_treatment_effect = float(np.mean(rubber_effect))
        rubber_ci = dml_rubber.effect_interval(X_confounders, alpha=0.05)
        ci_lower = float(np.mean(rubber_ci[0]))
        ci_upper = float(np.mean(rubber_ci[1]))

        self.avg_rubber_effect = avg_treatment_effect
        self.rubber_ci = (ci_lower, ci_upper)

        # --- Double ML: Consumer expenditure impact ---
        X_confounders_2 = train_data[
            [
                "log_gross_income",
                "log_plastic",
                "log_rubber",
                "scouring_agents",
                "colourants",
                "pop_growth",
            ]
        ].values
        T_treatment_2 = train_data["log_consumer_exp"].values
        Y_outcome_2 = train_data["nike"].values

        dml_consumer = LinearDML(
            model_y=RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            model_t=RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            discrete_treatment=False,
            cv=3,
        )
        dml_consumer.fit(Y_outcome_2, T_treatment_2, X=X_confounders_2)

        consumer_effect = dml_consumer.effect(X_confounders_2)
        avg_consumer_effect = float(np.mean(consumer_effect))
        consumer_ci = dml_consumer.effect_interval(X_confounders_2, alpha=0.05)
        ci_lower_c = float(np.mean(consumer_ci[0]))
        ci_upper_c = float(np.mean(consumer_ci[1]))

        self.avg_consumer_effect = avg_consumer_effect
        self.consumer_ci = (ci_lower_c, ci_upper_c)

        # Build concise text summary
        top_feature = sensitivity_df.iloc[0]
        txt = []
        txt.append("FEATURE SENSITIVITY (GAM Elasticities)")
        txt.append(
            f"- Most influential: {top_feature['Feature']} "
            f"(elasticity={top_feature['Elasticity']:.3f})"
        )
        txt.append(
            f"- 10% increase in {top_feature['Feature']} "
            f"→ {top_feature['Impact of 10% increase (%)']:.2f}% change in Nike sales"
        )
        txt.append("")
        txt.append("CAUSAL INFERENCE #1: RUBBER PRICE → NIKE SALES (Double ML)")
        txt.append(
            f"- Average Treatment Effect (ATE): {avg_treatment_effect:.2f} "
            f"with 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
        )
        txt.append(
            f"- Approx effect of 10% rubber price increase: {avg_treatment_effect*0.1:.2f} units change in sales"
        )
        txt.append("")
        txt.append(
            "CAUSAL INFERENCE #2: CONSUMER EXPENDITURE → NIKE SALES (Double ML)"
        )
        txt.append(
            f"- Average Treatment Effect (ATE): +{avg_consumer_effect:.2f} "
            f"with 95% CI [{ci_lower_c:.2f}, {ci_upper_c:.2f}]"
        )
        txt.append(
            f"- Approx effect of 10% spending increase: {avg_consumer_effect*0.1:.2f} units increase in sales"
        )

        return "\n".join(txt)

    # ------------------------------------------------------------------
    # PUBLIC METHOD: FINAL FORECASTS
    # ------------------------------------------------------------------
    def generate_forecast(self) -> str:
        if self.train_data is None or self.future_data is None:
            msg = self.load_and_prep()
            if msg.startswith("Error"):
                return msg
        if (
            self.orbit_metrics is None
            or self.prophet_metrics is None
            or self.gam_metrics is None
            or self.best_model_name is None
        ):
            _ = self.train_and_compare()
        if self.sensitivity_df is None or self.avg_rubber_effect is None:
            _ = self.run_causal_analysis()

        future_data = self.future_data.copy()
        future_data["log_consumer_exp"] = np.log1p(
            future_data["consumer_expenditure_footwear"]
        )
        future_data["log_gross_income"] = np.log1p(future_data["gross_income"])
        future_data["log_plastic"] = np.log1p(future_data["plastic_prd"])
        future_data["log_rubber"] = np.log1p(future_data["rubber_price"])

        X_future = future_data[self.X_features].values

        # Choose forecast source and metrics based on best model selected
        metrics = None
        if self.best_model_name == "GAM":
            y_future = self.gam_trained.predict(X_future)
            model_label = "GAM"
            metrics = self.gam_metrics
        elif self.best_model_name == "Orbit ETS":
            if self.orbit_model is None:
                self.orbit_model = self._orbit_model_with_tuning(
                    self.train_data, predict_future=True
                )
            orbit_future_df = future_data[["year"]].copy()
            orbit_future_df.columns = ["ds"]
            orbit_future_df["ds"] = pd.to_datetime(orbit_future_df["ds"], format="%Y")
            for feat in self.X_features:
                orbit_future_df[feat] = future_data[feat].values
            orbit_pred = self.orbit_model.predict(df=orbit_future_df, decompose=True)
            y_future = orbit_pred["prediction"].values
            model_label = "Orbit ETS"
            metrics = self.orbit_metrics
        elif self.best_model_name == "Prophet":
            if self.prophet_model is None:
                self.prophet_model = self._prophet_model_with_tuning(
                    self.train_data, predict_future=True
                )
            prophet_future_df = future_data[["year"]].copy()
            prophet_future_df.columns = ["ds"]
            prophet_future_df["ds"] = pd.to_datetime(
                prophet_future_df["ds"], format="%Y"
            )
            for feat in self.X_features:
                prophet_future_df[feat] = future_data[feat].values
            forecast = self.prophet_model.predict(prophet_future_df)
            y_future = forecast["yhat"].values
            model_label = "Prophet"
            metrics = self.prophet_metrics
        else:
            return "Error: best model not determined. Please run train_and_compare first."

        forecast_df = pd.DataFrame(
            {
                "Year": future_data["year"].values,
                "Forecast ($M)": y_future / 1000.0,
                "Lower Bound ($M)": (y_future * 0.92) / 1000.0,
                "Upper Bound ($M)": (y_future * 1.08) / 1000.0,
            }
        )
        self.forecast_df = forecast_df

        # Headline numbers similar to forecast.py bottom section
        last_year_forecast_m = y_future[-1] / 1000.0
        last_train = self.train_data["nike"].iloc[-1]
        growth_pct = (y_future[-1] - last_train) / last_train * 100.0

        top_cons = self.sensitivity_df[
            self.sensitivity_df["Feature"] == "log_consumer_exp"
        ]["Elasticity"].values[0]
        top_inc = self.sensitivity_df[
            self.sensitivity_df["Feature"] == "log_gross_income"
        ]["Elasticity"].values[0]

        txt = []
        txt.append(f"FORECASTS: NIKE SALES 2026–2030 ({model_label})")
        txt.append(forecast_df.to_string(index=False))
        txt.append("")
        if metrics is not None:
            txt.append(
                f"✅ Best Model: {model_label} with MAPE {metrics['mape']*100:.2f}%"
            )
        else:
            txt.append(f"✅ Best Model: {model_label}")
        txt.append(f"✅ 2030 Forecast: ${last_year_forecast_m:.2f}M")
        txt.append(f"✅ Growth 2025–2030: {growth_pct:.1f}%")
        txt.append("")
        txt.append("📈 Key Drivers (Elasticities):")
        txt.append(f"- Consumer Expenditure: {top_cons:.3f}")
        txt.append(f"- Gross Income:        {top_inc:.3f}")
        txt.append(
            f"- Rubber Price (ATE, Double ML): {self.avg_rubber_effect:.2f} per unit change"
        )

        return "\n".join(txt)