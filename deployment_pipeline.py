import os
import hashlib
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


class DeploymentPipeline:
    REQUIRED_COLUMNS = [
        "income",
        "debt",
        "region",
        "load_shedding_hours",
        "financial_strain"
    ]

    ETHICAL_CONSTRAINTS = [
        "Human review required for debt_to_income > 0.6",
        "Never deny service based solely on model output",
        "Monthly fairness audits mandatory"
    ]

    MODEL_VERSION = "1.2"
    TRAINING_DATE = "2026-09-13"

    def __init__(
        self,
        model_path="models/churn_pipeline.pkl",
        inference_path="models/inference_pipeline.pkl"
    ):
        self.model_path = model_path
        self.inference_path = inference_path
        self.pipeline = None
        self.metadata = {}
        self.error_count = 0
        self.total_predictions = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _calculate_file_hash(self, file_path):
        if not os.path.exists(file_path):
            return None

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        if isinstance(artifact, dict) and "pipeline" in artifact:
            self.pipeline = artifact["pipeline"]
            self.metadata = artifact.get("metadata", {})
        else:
            self.pipeline = artifact
            self.metadata = {}

        stored_version = self.metadata.get(
            "version",
            self.MODEL_VERSION
        )

        if str(stored_version) != self.MODEL_VERSION:
            raise ValueError(
                f"Model version mismatch. Expected {self.MODEL_VERSION}, "
                f"found {stored_version}."
            )

        stored_hash = self.metadata.get("artifact_hash")

        if stored_hash:
            current_hash = self._calculate_file_hash(self.model_path)

            if current_hash != stored_hash:
                raise ValueError(
                    "Model hash validation failed. The model artifact "
                    "may have been modified."
                )
        else:
            logging.warning(
                "No stored artifact hash was found. "
                "Continuing with version validation."
            )

        logging.info(
            "Model loaded successfully. Version: %s",
            stored_version
        )

        return self.pipeline

    def _error_rate(self):
        if self.total_predictions == 0:
            return 0.0

        return self.error_count / self.total_predictions

    def _circuit_breaker_triggered(self):
        return self._error_rate() > 0.05

    def _safe_default(self, rows):
        logging.warning(
            "Circuit breaker active. Returning safe default predictions."
        )

        return np.zeros(rows, dtype=int)

    def _derive_financial_strain(self, data):
        if "financial_strain" in data.columns:
            return data

        data = data.copy()

        if "debt_to_income" in data.columns:
            data["financial_strain"] = (
                pd.to_numeric(
                    data["debt_to_income"],
                    errors="coerce"
                )
                .fillna(0)
                .clip(0, 1)
            )

        elif "debt" in data.columns and "income" in data.columns:
            income = pd.to_numeric(
                data["income"],
                errors="coerce"
            )

            debt = pd.to_numeric(
                data["debt"],
                errors="coerce"
            )

            ratio = debt / income.replace(0, np.nan)

            data["financial_strain"] = (
                ratio
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .clip(0, 1)
            )

        else:
            logging.warning(
                "financial_strain could not be calculated. "
                "Using safe default value of 0."
            )

            data["financial_strain"] = 0.0

        return data

    def _get_known_regions(self):
        known_regions = []

        try:
            if hasattr(self.pipeline, "named_steps"):
                preprocessor = self.pipeline.named_steps.get(
                    "preprocessor"
                )

                if preprocessor is not None:
                    for transformer in preprocessor.transformers_:
                        if len(transformer) < 3:
                            continue

                        name = transformer[0]
                        transformer_object = transformer[1]
                        columns = transformer[2]

                        if name == "cat":
                            if hasattr(
                                transformer_object,
                                "named_steps"
                            ):
                                encoder = (
                                    transformer_object.named_steps.get(
                                        "onehotencoder"
                                    )
                                )

                                if encoder is not None:
                                    if hasattr(
                                        encoder,
                                        "categories_"
                                    ):
                                        for category_list in (
                                            encoder.categories_
                                        ):
                                            known_regions.extend(
                                                [
                                                    str(x)
                                                    for x in category_list
                                                ]
                                            )

                            elif hasattr(
                                transformer_object,
                                "categories_"
                            ):
                                for category_list in (
                                    transformer_object.categories_
                                ):
                                    known_regions.extend(
                                        [
                                            str(x)
                                            for x in category_list
                                        ]
                                    )

        except Exception as error:
            logging.warning(
                "Could not determine training regions: %s",
                error
            )

        return sorted(set(known_regions))

    def _validate_input(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Input DataFrame is empty."
            )

        data = data.copy()

        missing_columns = []

        for column in self.REQUIRED_COLUMNS:
            if column == "financial_strain":
                continue

            if column not in data.columns:
                missing_columns.append(column)

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        data = self._derive_financial_strain(data)

        numeric_columns = [
            "income",
            "debt",
            "load_shedding_hours",
            "financial_strain"
        ]

        for column in numeric_columns:
            if column in data.columns:
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        if "region" in data.columns:
            data["region"] = data["region"].astype(str)

            known_regions = self._get_known_regions()

            if known_regions:
                unknown_regions = sorted(
                    set(data["region"]) - set(known_regions)
                )

                if unknown_regions:
                    logging.warning(
                        "Unknown regions detected: %s. "
                        "OneHotEncoder should handle unknown categories safely.",
                        unknown_regions
                    )

        if "financial_strain" in data.columns:
            data["financial_strain"] = (
                data["financial_strain"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .clip(0, 1)
            )

        return data

    def _prepare_model_input(self, data):
        if not hasattr(self.pipeline, "feature_names_in_"):
            return data

        expected_features = list(
            self.pipeline.feature_names_in_
        )

        missing_model_features = [
            column
            for column in expected_features
            if column not in data.columns
        ]

        if missing_model_features:
            raise ValueError(
                "Input does not contain all model features: "
                + ", ".join(missing_model_features)
            )

        return data[expected_features]

    def predict(self, data):
        if self.pipeline is None:
            raise RuntimeError(
                "Model is not loaded. Run load_model() first."
            )

        rows = len(data)

        self.total_predictions += 1

        if self._circuit_breaker_triggered():
            return self._safe_default(rows)

        try:
            validated_data = self._validate_input(data)

            model_input = self._prepare_model_input(
                validated_data
            )

            predictions = self.pipeline.predict(
                model_input
            )

            self.consecutive_errors = 0

            return predictions

        except Exception as error:
            self.error_count += 1
            self.consecutive_errors += 1

            logging.error(
                "Prediction failed: %s",
                error
            )

            if self.consecutive_errors >= self.max_consecutive_errors:
                logging.critical(
                    "Maximum consecutive prediction errors reached."
                )

            if self._circuit_breaker_triggered():
                return self._safe_default(rows)

            return self._safe_default(rows)

    def calculate_data_hash(
        self,
        data_path="data/processed/engineered_features.csv"
    ):
        data_hash = self._calculate_file_hash(data_path)

        if data_hash is None:
            logging.warning(
                "Training data file not found: %s",
                data_path
            )

            return "N/A"

        return data_hash

    def create_metadata(self):
        source_model_hash = self._calculate_file_hash(
            self.model_path
        )

        data_hash = self.calculate_data_hash()

        self.metadata = {
            "version": self.MODEL_VERSION,
            "training_date": self.TRAINING_DATE,
            "data_hash": data_hash,
            "source_model_hash": source_model_hash,
            "ethical_constraints": self.ETHICAL_CONSTRAINTS,
            "created_at": datetime.now().isoformat(),
            "circuit_breaker_threshold": 0.05,
            "safe_default": 0
        }

        return self.metadata

    def save_inference_pipeline(self):
        os.makedirs(
            os.path.dirname(self.inference_path),
            exist_ok=True
        )

        artifact = {
            "pipeline": self.pipeline,
            "metadata": self.metadata
        }

        joblib.dump(
            artifact,
            self.inference_path
        )

        logging.info(
            "Inference pipeline saved to: %s",
            self.inference_path
        )

    def generate_readme(
        self,
        output_path="README.md"
    ):
        lines = []

        lines.append("# NEXUSLEND Churn Prediction Deployment Pipeline")
        lines.append("")
        lines.append(
            "This package provides a deployment wrapper around the "
            "NEXUSLEND churn prediction model."
        )
        lines.append("")
        lines.append("## 🚀 Quick Start")
        lines.append("")
        lines.append("1. Install the required Python packages:")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install pandas numpy scikit-learn joblib")
        lines.append("```")
        lines.append("")
        lines.append("2. Run the deployment pipeline:")
        lines.append("")
        lines.append("```bash")
        lines.append("python deployment_pipeline.py")
        lines.append("```")
        lines.append("")
        lines.append("3. The deployment artifact will be created at:")
        lines.append("")
        lines.append("```text")
        lines.append("models/inference_pipeline.pkl")
        lines.append("```")
        lines.append("")
        lines.append("## ⚠️ Critical Ethical Constraints")
        lines.append("")
        lines.append(
            "Do NOT target high-strain customers with predatory offers"
        )
        lines.append("")
        lines.append(
            "- Human review required for debt_to_income > 0.6."
        )
        lines.append(
            "- Never deny service based solely on model output."
        )
        lines.append(
            "- Monthly fairness audits are mandatory."
        )
        lines.append(
            "- High-risk predictions must be reviewed by a human."
        )
        lines.append(
            "- Model predictions must not be treated as automatic "
            "financial decisions."
        )
        lines.append("")
        lines.append("## 🔍 Monitoring Commands")
        lines.append("")
        lines.append(
            "Check the deployment artifact:"
        )
        lines.append("")
        lines.append("```bash")
        lines.append(
            "python -c \"import joblib; "
            "x=joblib.load('models/inference_pipeline.pkl'); "
            "print(x['metadata'])\""
        )
        lines.append("```")
        lines.append("")
        lines.append(
            "Run the deployment script again to regenerate the "
            "inference artifact:"
        )
        lines.append("")
        lines.append("```bash")
        lines.append("python deployment_pipeline.py")
        lines.append("```")
        lines.append("")
        lines.append(
            "Retrain if township recall drops >10%"
        )
        lines.append("")
        lines.append("## Circuit Breaker")
        lines.append("")
        lines.append(
            "The circuit breaker activates when the prediction "
            "error rate exceeds 5%."
        )
        lines.append(
            "When activated, the pipeline returns safe default "
            "predictions instead of continuing normal inference."
        )
        lines.append("")
        lines.append("## Input Validation")
        lines.append("")
        lines.append(
            "- Validates that the input is a pandas DataFrame."
        )
        lines.append(
            "- Checks required input fields."
        )
        lines.append(
            "- Converts numeric fields safely."
        )
        lines.append(
            "- Handles unknown region categories."
        )
        lines.append(
            "- Rejects missing model features."
        )
        lines.append("")
        lines.append("## Model Metadata")
        lines.append("")
        lines.append(
            f"- Version: {self.MODEL_VERSION}"
        )
        lines.append(
            f"- Training date: {self.TRAINING_DATE}"
        )
        lines.append(
            f"- Ethical constraints: {len(self.ETHICAL_CONSTRAINTS)}"
        )
        lines.append(
            f"- Source model: {self.model_path}"
        )
        lines.append("")
        lines.append("## Stakeholder Guidance")
        lines.append("")
        lines.append(
            "Per CFO: 'Every 1% churn reduction = R420k saved'"
        )
        lines.append("")
        lines.append(
            "The model is a decision-support tool and must not "
            "replace human judgement."
        )
        lines.append("")

        readme_content = "\n".join(lines)

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(readme_content)

        logging.info(
            "README generated: %s",
            output_path
        )

    def __str__(self):
        return (
            f"Pipeline v{self.MODEL_VERSION} | "
            f"Trained: {self.TRAINING_DATE} | "
            f"Ethical constraints: "
            f"{len(self.ETHICAL_CONSTRAINTS)}"
        )


def main():
    print("Starting Phase 5 deployment pipeline...")
    print("")

    deployment = DeploymentPipeline()

    print("Loading model...")
    deployment.load_model()

    print("Creating deployment metadata...")
    deployment.create_metadata()

    print("Saving inference pipeline...")
    deployment.save_inference_pipeline()

    print("Generating README...")
    deployment.generate_readme()

    print("")
    print("Phase 5 deployment completed successfully.")
    print(deployment)
    print("")
    print(
        "Inference pipeline: models/inference_pipeline.pkl"
    )
    print(
        "README: README.md"
    )


if __name__ == "__main__":
    main()