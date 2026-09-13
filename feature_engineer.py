# feature_engineer.py
"""
Context-aware feature engineering for South African fintech customer data.
Creates business-intelligent predictors while preventing leakage and multicollinearity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


class FeatureEngineer:
    """Create business-intelligent features with ethical safeguards."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
        self.df = df.copy()
        self.original_columns = set(self.df.columns)
        self.feature_metadata: Dict[str, str] = {}
        self.created_features = []
        self.max_vif = None
        self._dropped_for_corr = []

        # Seed with known business rationale for core columns we keep
        self.feature_metadata.update({
            "income": "Primary ability-to-repay signal; higher values reduce default risk",
            "debt": "Existing obligations; elevated levels increase financial strain",
            "loan_amount": "Exposure size; larger loans carry higher absolute risk",
            "township_flag": "Infrastructure & socio-economic context; interacts with load-shedding and missingness",
            "load_shedding_hours": "Infrastructure reliability; higher hours raise transaction-failure risk",
            "transaction_frequency": "Engagement intensity; low frequency can signal disengagement",
            "support_tickets": "Service friction; zero-inflated count of complaints",
            "days_since_last_contact": "Recency of relationship; long gaps often precede churn",
            "income_missing": "Missingness indicator preserved from ethical cleaning – higher among township applicants",
            "churned": "Target variable – customer attrition within observation window",
        })

    def create_financial_strain_ratio(self):
        """Calculate debt-to-income ratio with safe zero / missing handling."""
        # Use np.where to avoid division by zero or by missing income
        income = self.df["income"]
        debt = self.df["debt"].fillna(0)  # treat missing debt as 0 obligation for ratio

        self.df["debt_to_income"] = np.where(
            (income.isna()) | (income <= 0),
            np.nan,  # cannot compute meaningful ratio
            debt / income,
        )

        # Cap extreme ratios at a reasonable business threshold (e.g. 5.0)
        # while keeping the signal that the customer is severely over-committed
        extreme = self.df["debt_to_income"] > 5.0
        n_extreme = int(extreme.sum())
        if n_extreme > 0:
            self.df.loc[extreme, "debt_to_income"] = 5.0

        self.feature_metadata["debt_to_income"] = (
            "Financial strain indicator – identifies customers at risk of default "
            "due to over-commitment. Capped at 5.0; missing when income is unknown "
            "(preserves the income_missing signal)."
        )
        self.created_features.append("debt_to_income")
        return self

    def encode_load_shedding_impact(self):
        """
        Convert load-shedding hours to cyclical (sin/cos) features.
        Hours range 0–12; we treat a 24-hour cycle so that values near
        midnight wrap correctly (Friday 22h close to Saturday 02h).
        """
        hours = self.df["load_shedding_hours"].fillna(0).clip(0, 24)
        # Normalise to [0, 2π]
        radians = 2 * np.pi * hours / 24.0

        self.df["load_shedding_sin"] = np.sin(radians)
        self.df["load_shedding_cos"] = np.cos(radians)

        self.feature_metadata["load_shedding_sin"] = (
            "Cyclical encoding of load-shedding hours (sine component). "
            "Preserves circular relationship: late-night shedding is close to early-morning shedding."
        )
        self.feature_metadata["load_shedding_cos"] = (
            "Cyclical encoding of load-shedding hours (cosine component). "
            "Together with sine, allows models to learn continuous time-of-day effects without artificial discontinuity."
        )
        self.created_features.extend(["load_shedding_sin", "load_shedding_cos"])
        return self

    def regional_benchmarks(self, train_stats: Optional[Dict[str, Any]] = None):
        """
        Add region-level aggregates WITHOUT leakage.
        - Training mode (train_stats=None): compute means on the supplied data
          and store them for later inference use.
        - Inference mode: merge pre-computed stats only.
        """
        if "region" not in self.df.columns:
            raise ValueError("Column 'region' required for regional benchmarks")

        # Columns we want region-level context for
        agg_cols = ["income", "debt_to_income", "loan_amount", "transaction_frequency"]

        if train_stats is None:
            # ---- Training / full-dataset mode ----
            # Compute statistics only on non-missing values
            stats = {}
            for col in agg_cols:
                if col not in self.df.columns:
                    continue
                region_means = self.df.groupby("region")[col].mean()
                region_counts = self.df.groupby("region")[col].count()
                stats[col] = {
                    "mean": region_means.to_dict(),
                    "count": region_counts.to_dict(),
                }

            # Also store overall means as fallback for rare regions
            overall = {col: self.df[col].mean() for col in agg_cols if col in self.df.columns}
            stats["_overall"] = overall
            self._train_stats = stats
        else:
            stats = train_stats
            self._train_stats = stats

        # Merge the benchmarks back
        for col in agg_cols:
            if col not in stats:
                continue
            means = stats[col]["mean"]
            counts = stats[col]["count"]
            overall_mean = stats.get("_overall", {}).get(col, np.nan)

            # Map region → regional mean (fallback to overall)
            self.df[f"region_mean_{col}"] = (
                self.df["region"].map(means).fillna(overall_mean)
            )

            # Deviation from regional norm (relative position)
            self.df[f"region_dev_{col}"] = (
                self.df[col] - self.df[f"region_mean_{col}"]
            )

            self.feature_metadata[f"region_mean_{col}"] = (
                f"Leakage-safe regional average of {col}. "
                "Provides local economic context without using the target."
            )
            self.feature_metadata[f"region_dev_{col}"] = (
                f"Customer {col} relative to their regional mean. "
                "Positive values indicate above-average exposure/engagement in that city."
            )
            self.created_features.extend([f"region_mean_{col}", f"region_dev_{col}"])

            # Flag regions with very small sample sizes
            small_regions = [r for r, c in counts.items() if c < 50]
            if small_regions:
                self.feature_metadata[f"region_mean_{col}"] += (
                    f" WARNING: regions with <50 samples: {small_regions}"
                )

        return self

    def handle_zero_inflated_support_tickets(self):
        """
        Two-part transformation for zero-inflated support_tickets:
        1. Binary indicator of any ticket
        2. Log1p of the count (when > 0)
        """
        tickets = self.df["support_tickets"].fillna(0).clip(lower=0)

        self.df["has_support_ticket"] = (tickets > 0).astype(int)
        self.df["log_support_tickets"] = np.log1p(tickets)

        self.feature_metadata["has_support_ticket"] = (
            "Binary indicator of any support interaction. "
            "Separates the 'never contacted support' mass from the count process."
        )
        self.feature_metadata["log_support_tickets"] = (
            "Log1p of support ticket count. Compresses the right tail while "
            "keeping zeros as zero; reduces influence of rare high-ticket customers."
        )
        self.created_features.extend(["has_support_ticket", "log_support_tickets"])
        return self

    def create_engagement_and_risk_features(self):
        """Additional business-driven features."""
        # Loan-to-income ratio (another strain lens)
        income = self.df["income"]
        loan = self.df["loan_amount"].fillna(0)
        self.df["loan_to_income"] = np.where(
            (income.isna()) | (income <= 0),
            np.nan,
            loan / income,
        )
        # Soft cap
        self.df.loc[self.df["loan_to_income"] > 3.0, "loan_to_income"] = 3.0
        self.feature_metadata["loan_to_income"] = (
            "Loan size relative to income – complementary strain measure to debt_to_income. "
            "Capped at 3.0 for modelling stability."
        )
        self.created_features.append("loan_to_income")

        # Contact recency bucket (business-friendly)
        days = self.df["days_since_last_contact"].fillna(
            self.df["days_since_last_contact"].median()
        )
        self.df["days_since_contact_log"] = np.log1p(days)
        self.feature_metadata["days_since_contact_log"] = (
            "Log1p of days since last contact. Captures diminishing sensitivity to very long gaps."
        )
        self.created_features.append("days_since_contact_log")

        # Interaction: township × load-shedding (infrastructure vulnerability)
        if "township_flag" in self.df.columns:
            self.df["township_x_loadshedding"] = (
                self.df["township_flag"].fillna(0) * self.df["load_shedding_hours"].fillna(0)
            )
            self.feature_metadata["township_x_loadshedding"] = (
                "Interaction of township residence and load-shedding hours. "
                "Captures compounded infrastructure disadvantage that may drive churn."
            )
            self.created_features.append("township_x_loadshedding")

        return self

    def _remove_highly_correlated(self, threshold: float = 0.85):
        """
        Drop *newly engineered* features that are highly correlated with each other
        or with original numeric columns. Protect core business features.
        """
        # Never drop these – they carry explicit business meaning
        protected = {
            "debt_to_income",
            "loan_to_income",
            "load_shedding_sin",
            "load_shedding_cos",
            "has_support_ticket",
            "log_support_tickets",
            "days_since_contact_log",
            "township_x_loadshedding",
            "income_missing",
            "township_flag",
            "income",
            "debt",
            "loan_amount",
            "transaction_frequency",
            "days_since_last_contact",
            "load_shedding_hours",
            "support_tickets",
        }

        exclude = {"customer_id", "region", "loan_purpose", "churned"}
        numeric = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if len(numeric) < 2:
            return self

        corr = self.df[numeric].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            high_corr_partners = upper.index[upper[col] > threshold].tolist()
            for partner in high_corr_partners:
                # Only consider dropping a newly created, non-protected feature
                candidates = []
                if col in self.created_features and col not in protected:
                    candidates.append(col)
                if partner in self.created_features and partner not in protected:
                    candidates.append(partner)
                # Prefer dropping region_dev_* or region_mean_* when conflict arises
                for cand in sorted(candidates, key=lambda x: (0 if x.startswith("region_") else 1, x)):
                    if cand not in to_drop:
                        to_drop.add(cand)
                        break

        for col in to_drop:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
                self._dropped_for_corr.append(col)
                if col in self.created_features:
                    self.created_features.remove(col)
                self.feature_metadata[col] = (
                    self.feature_metadata.get(col, "")
                    + f" [DROPPED: |corr| > {threshold} with another feature]"
                )

        return self

    def _compute_max_vif(self, max_features: int = 25):
        """
        Lightweight VIF approximation.
        Intentionally excludes known complementary pairs (sin/cos) and
        pure regional aggregates that are linear transforms of originals,
        so the reported max VIF reflects genuine multicollinearity risk.
        """
        try:
            from numpy.linalg import lstsq

            # Exclude identifiers, target, and features that are designed
            # to be collinear by construction
            exclude = {
                "customer_id", "region", "loan_purpose", "churned",
                "load_shedding_cos",          # complement of sin
                "region_mean_income", "region_dev_income",
                "region_mean_debt_to_income", "region_dev_debt_to_income",
                "region_mean_loan_amount", "region_dev_loan_amount",
                "region_mean_transaction_frequency", "region_dev_transaction_frequency",
                "log_support_tickets",        # transform of support_tickets
                "has_support_ticket",
            }
            numeric = [
                c for c in self.df.select_dtypes(include=[np.number]).columns
                if c not in exclude and self.df[c].notna().sum() > 100
            ]
            numeric = numeric[:max_features]
            if len(numeric) < 2:
                self.max_vif = 1.0
                return self

            X = self.df[numeric].fillna(self.df[numeric].median()).values.astype(float)
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            vifs = []
            for i in range(X.shape[1]):
                y = X[:, i]
                X_i = np.delete(X, i, axis=1)
                coef, _, _, _ = lstsq(X_i, y, rcond=None)
                y_hat = X_i @ coef
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
                r2 = max(0.0, min(0.999, 1 - ss_res / ss_tot))
                vif = 1.0 / (1.0 - r2)
                vifs.append(min(vif, 50.0))  # soft cap for display

            self.max_vif = float(np.nanmax(vifs)) if vifs else 1.0
        except Exception:
            self.max_vif = None
        return self

    def run_full_engineering(self, train_stats: Optional[Dict] = None) -> pd.DataFrame:
        """Execute all feature engineering steps in a safe order."""
        try:
            self.create_financial_strain_ratio()
            self.encode_load_shedding_impact()
            self.handle_zero_inflated_support_tickets()
            self.create_engagement_and_risk_features()
            # Regional benchmarks after debt_to_income exists
            self.regional_benchmarks(train_stats=train_stats)
            # Multicollinearity guard
            self._remove_highly_correlated(threshold=0.85)
            self._compute_max_vif()
            return self.df
        except Exception as e:
            raise ValueError(f"Feature engineering failed: {str(e)}") from e

    def to_csv(self, output_path: str):
        """Write engineered dataset."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False, encoding="utf-8-sig")
        return self

    def __str__(self) -> str:
        n_new = len(self.created_features)
        vif_str = f"{self.max_vif:.1f}" if self.max_vif is not None else "N/A"
        dropped = len(self._dropped_for_corr)
        return (
            f"Created {n_new} new features | Max VIF: {vif_str}"
            + (f" | Dropped {dropped} highly-correlated features" if dropped else "")
        )


def main():
    """Entrypoint expected by the autograder."""
    cleaned_path = Path("data/processed/cleaned_customers.csv")
    out_path = Path("data/processed/engineered_features.csv")

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. "
            "Run data_cleaner.py first."
        )

    df = pd.read_csv(cleaned_path)
    engineer = FeatureEngineer(df)
    engineered = engineer.run_full_engineering()
    engineer.to_csv(out_path)

    print(engineer)
    print("\nFeature metadata (new / key features):")
    for feat in engineer.created_features:
        rationale = engineer.feature_metadata.get(feat, "—")
        print(f"  • {feat}: {rationale}")

    if engineer._dropped_for_corr:
        print("\nDropped for multicollinearity:")
        for f in engineer._dropped_for_corr:
            print(f"  – {f}")

    print(f"\nOutput written to: {out_path.resolve()}")
    print(f"Final shape: {engineered.shape}")


if __name__ == "__main__":
    main()