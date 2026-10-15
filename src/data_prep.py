"""
Loading, cleaning, splitting, encoding, and standardizing bina.az data.

Data-leakage protections:
- unit_price and total_price are never used as features.
- Rows are split before medians and category levels are learned.
- Missing-value medians are calculated from training data only.
- Standardization statistics are calculated from training data only.
- The price-tier threshold is calculated from training prices only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "bina_az_sale.csv"

if not DATA_PATH.is_file():
    DATA_PATH = (
        PROJECT_ROOT
        / "ml_final_project"
        / "data"
        / "bina_az_sale.csv"
    )
TARGET_COLUMN = "price"

# These columns directly reveal or are derived from the target.
LEAKAGE_COLUMNS = [
    "unit_price",
    "total_price",
]

# A restricted feature set prevents identifiers, URLs, owner names, addresses,
# and free text from producing enormous one-hot-encoded matrices.
RAW_FEATURE_COLUMNS = [
    "city",
    "location",
    "lat",
    "lng",
    "views",
    "Kateqoriya",
    "Mərtəbə",
    "Otaq sayı",
    "Sahə",
    "Torpaq sahəsi",
    "Təmir",
    "Çıxarış",
    "İpoteka",
    "vip",
    "featured",
    "products_label",
]

NUMERIC_FEATURES = [
    "lat",
    "lng",
    "views",
    "Otaq sayı",
    "Sahə_m2",
    "Torpaq_sahəsi_m2",
    "Mərtəbə_cari",
    "Mərtəbə_ümumi",
    "vip_flag",
    "featured_flag",
    "Sahə_missing",
    "Torpaq_missing",
]

CATEGORICAL_FEATURES = [
    "city",
    "location",
    "Kateqoriya",
    "Təmir",
    "Çıxarış",
    "İpoteka",
    "products_label",
]


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read and return the raw bina.az CSV."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Run the program from the ml_final_project directory."
        )

    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8-sig")


def _extract_number(value) -> float:
    """
    Extract the first numeric value from text.

    Examples:
        "145 m²" -> 145.0
        "12,5 m²" -> 12.5
        "3 450 AZN/m²" -> 3450.0
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace(",", ".")

    # Spaces between digits usually represent thousands separators.
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if match is None:
        return np.nan

    return float(match.group())


def _parse_land_area(value) -> float:
    """
    Convert land area to square metres.

    Supported examples:
        "6 sot"   -> 600 m²
        "600 m²"  -> 600 m²
        "1 ha"    -> 10000 m²
    """
    number = _extract_number(value)

    if not np.isfinite(number):
        return np.nan

    text = str(value).lower()

    if "sot" in text:
        return number * 100.0

    if "hektar" in text or re.search(r"\bha\b", text):
        return number * 10_000.0

    return number


def _parse_floor(value) -> tuple[float, float]:
    """
    Extract current and total floor values.

    Examples:
        "5 / 16" -> (5, 16)
        "5/16"   -> (5, 16)
        "5"      -> (5, NaN)
    """
    if pd.isna(value):
        return np.nan, np.nan

    numbers = re.findall(r"\d+(?:[.,]\d+)?", str(value))

    if not numbers:
        return np.nan, np.nan

    current_floor = float(numbers[0].replace(",", "."))

    if len(numbers) >= 2:
        total_floors = float(numbers[1].replace(",", "."))
    else:
        total_floors = np.nan

    return current_floor, total_floors


def _present_flag(series: pd.Series) -> pd.Series:
    """
    Convert sparse promotion fields to binary flags.

    A populated value means the feature is present; a missing value means it
    was not specified or not active.
    """
    return series.notna().astype(float)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready raw features without learning any statistics."""
    features = pd.DataFrame(index=df.index)

    for column in ["lat", "lng", "views", "Otaq sayı"]:
        features[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # In this actual CSV, Sahə for Torpaq listings is in sot, not m².
    # Parse units on BOTH area fields, not only on Torpaq sahəsi.
    features["Sahə_m2"] = df["Sahə"].map(_parse_land_area)
    features["Torpaq_sahəsi_m2"] = df["Torpaq sahəsi"].map(
        _parse_land_area
    )
    features["Sahə_missing"] = features["Sahə_m2"].isna().astype(float)
    features["Torpaq_missing"] = features["Torpaq_sahəsi_m2"].isna().astype(float)

    parsed_floors = df["Mərtəbə"].map(_parse_floor)
    features["Mərtəbə_cari"] = parsed_floors.map(
        lambda values: values[0]
    )
    features["Mərtəbə_ümumi"] = parsed_floors.map(
        lambda values: values[1]
    )

    features["vip_flag"] = _present_flag(df["vip"])
    features["featured_flag"] = _present_flag(df["featured"])

    for column in CATEGORICAL_FEATURES:
        features[column] = df[column]

    return features


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean records using fixed validity rules.

    Rules:
    - Remove exact duplicates.
    - Keep the latest occurrence of duplicate estate IDs.
    - Remove missing and non-positive target prices.
    - Remove invalid property areas, room counts, floors, and coordinates.
    - Retain only the target and selected raw feature columns.
    - Exclude unit_price and total_price because they leak price.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        raise ValueError("Column names collide after stripping whitespace")
    required_columns = {TARGET_COLUMN, "Sahə", "Kateqoriya"}
    missing_columns = required_columns.difference(cleaned.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    audit = {"raw_rows": len(cleaned)}
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    audit["exact_duplicates_removed"] = before - len(cleaned)
    cleaned["source_row_id"] = cleaned.index.to_numpy()

    # This file is entirely AZN. Refuse unknown currencies rather than silently
    # treating USD/EUR numbers as AZN; conversion needs a documented policy.
    for column in ["currency_x", "currency_y"]:
        if column in cleaned:
            currencies = cleaned[column].astype("string").str.strip().str.upper()
            if currencies.isna().any() or not currencies.eq("AZN").all():
                raise ValueError(f"{column} is not entirely AZN; supply a verified conversion policy")
    audit["currency"] = "AZN"
    for column in RAW_FEATURE_COLUMNS:
        if column not in cleaned:
            cleaned[column] = np.nan

    # The same listing should not appear in separate dataset splits.
    before = len(cleaned)
    if "estate_id" in cleaned:
        # Use an actual timestamp for latest, not physical CSV row order.
        if "datetime_scrape_y" in cleaned:
            cleaned["_scrape_time"] = pd.to_datetime(cleaned["datetime_scrape_y"], format="mixed", utc=True, errors="coerce")
            cleaned = cleaned.sort_values("_scrape_time", kind="stable", na_position="first")
        valid_id = cleaned["estate_id"].notna()
        repeated = valid_id & cleaned["estate_id"].duplicated(keep="last")
        cleaned = cleaned.loc[~repeated].sort_values("source_row_id", kind="stable")
    audit["duplicate_ids_removed"] = before - len(cleaned)

    cleaned[TARGET_COLUMN] = pd.to_numeric(
        cleaned[TARGET_COLUMN],
        errors="coerce",
    )

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned[
        cleaned[TARGET_COLUMN].between(1_000, 20_000_000)
    ]
    audit["price_scope_exclusions"] = before - len(cleaned)
    audit["price_scope_note"] = "Preserved author's fixed eligibility range: 1000 <= price <= 20000000 AZN."

    # Create temporary numeric values only for fixed validity checks.
    area = cleaned["Sahə"].map(_parse_land_area)
    rooms = pd.to_numeric(
        cleaned["Otaq sayı"],
        errors="coerce",
    )
    latitude = pd.to_numeric(cleaned["lat"], errors="coerce")
    longitude = pd.to_numeric(cleaned["lng"], errors="coerce")

    land_listing = cleaned["Kateqoriya"].astype("string").str.strip().eq("Torpaq").fillna(False)
    # Keep the author's 5..5000 m² scope for built property. A land plot can
    # legitimately exceed 5000 m²; do not apply that building limit to it.
    valid_area = area.isna() | ((area >= 5) & (land_listing | (area <= 5_000)))
    valid_rooms = rooms.isna() | rooms.between(1, 100)

    # Approximate Azerbaijan coordinate limits. Missing coordinates are kept
    # so they can be imputed from training data.
    valid_latitude = latitude.isna() | latitude.between(38.0, 42.5)
    valid_longitude = longitude.isna() | longitude.between(44.0, 51.5)

    before = len(cleaned)
    cleaned = cleaned[
        valid_area
        & valid_rooms
        & valid_latitude
        & valid_longitude
    ]
    audit["feature_scope_exclusions"] = before - len(cleaned)

    selected_columns = [
        TARGET_COLUMN,
        *RAW_FEATURE_COLUMNS,
        "source_row_id",
    ]

    cleaned = cleaned[selected_columns].copy()

    # Explicit assertion that leakage fields cannot reach the model.
    if any(
        column in cleaned.columns
        for column in LEAKAGE_COLUMNS
    ):
        raise RuntimeError("A leakage column was not removed")

    cleaned = cleaned.reset_index(drop=True)
    audit["clean_rows"] = len(cleaned)
    cleaned.attrs["cleaning_audit"] = audit
    return cleaned


def split_dataframe(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a cleaned DataFrame before fitting preprocessing rules.

    Default proportions:
        training:   70%
        validation: 15%
        test:       15%
    """
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")

    if not 0 <= test_size < 1:
        raise ValueError("test_size must be in [0, 1)")

    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    if len(df) < 3:
        raise ValueError("At least three rows are required")

    random_generator = np.random.default_rng(seed)
    shuffled_indices = random_generator.permutation(len(df))

    n_validation = int(round(len(df) * val_size))
    n_test = int(round(len(df) * test_size))
    n_train = len(df) - n_validation - n_test

    if n_train < 1 or (val_size > 0 and n_validation < 1) or (test_size > 0 and n_test < 1):
        raise ValueError("Split sizes leave no training records")

    train_indices = shuffled_indices[:n_train]
    validation_indices = shuffled_indices[
        n_train:n_train + n_validation
    ]
    test_indices = shuffled_indices[n_train + n_validation:]

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[validation_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
    )


class TabularPreprocessor:
    """
    Learn numeric imputation and category encoding from training data.

    Numeric missing values use training medians. Categorical missing values
    become "Missing". Categories unseen during training produce all-zero
    one-hot columns.
    """

    def __init__(self):
        self.numeric_medians_: dict[str, float] = {}
        self.category_levels_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    @staticmethod
    def _clean_category(series: pd.Series) -> pd.Series:
        """Normalize missing and blank categorical values."""
        return (
            series.astype("string")
            .fillna("Missing")
            .str.strip()
            .replace("", "Missing")
        )

    def fit(self, train_df: pd.DataFrame) -> TabularPreprocessor:
        """Learn preprocessing parameters from training records only."""
        if len(train_df) == 0:
            raise ValueError("Cannot fit preprocessing on empty training data")
        raw_features = _engineer_features(train_df)

        self.numeric_medians_ = {}

        for column in NUMERIC_FEATURES:
            median = raw_features[column].median()

            if pd.isna(median):
                median = 0.0

            self.numeric_medians_[column] = float(median)

        self.category_levels_ = {}

        for column in CATEGORICAL_FEATURES:
            values = self._clean_category(raw_features[column])
            self.category_levels_[column] = sorted(
                values.unique().tolist()
            )

        transformed = self._transform_engineered(raw_features)
        self.feature_names_ = transformed.columns.tolist()
        self.is_fitted_ = True

        return self

    def _transform_engineered(
        self,
        raw_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply currently stored imputation and encoding rules."""
        output: dict[str, pd.Series] = {}

        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(
                raw_features[column],
                errors="coerce",
            )

            # Zero is a missing-land PLACEHOLDER, not a claim of measured zero.
            # Torpaq_missing retains this distinction for the model.
            if column == "Torpaq_sahəsi_m2":
                fill_value = 0.0
            else:
                fill_value = self.numeric_medians_[column]

            output[column] = values.fillna(fill_value).astype(float)

        for column in CATEGORICAL_FEATURES:
            values = self._clean_category(raw_features[column])

            for category in self.category_levels_[column]:
                encoded_name = f"{column}={category}"
                output[encoded_name] = (
                    values == category
                ).astype(float)

        return pd.DataFrame(output, index=raw_features.index)

    def transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform records into numeric X and target y."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' is missing"
            )

        raw_features = _engineer_features(df)
        transformed = self._transform_engineered(raw_features)

        transformed = transformed.reindex(
            columns=self.feature_names_,
            fill_value=0.0,
        )

        y = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)

        X = transformed.to_numpy(dtype=float)

        if not np.all(np.isfinite(X)):
            raise ValueError("Transformed features contain invalid values")

        if not np.all(np.isfinite(y)):
            raise ValueError("Target contains invalid values")

        return X, y

    def fit_transform(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the preprocessor and transform training records."""
        self.fit(train_df)
        return self.transform(train_df)


def make_features(
    df: pd.DataFrame,
    preprocessor: TabularPreprocessor | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Transform using a processor already fitted on TRAINING rows.
    Requiring state prevents the starter's build-before-split leakage trap.
    """
    if preprocessor is None:
        raise ValueError("Pass a TabularPreprocessor fitted on TRAIN first")
    X, y = preprocessor.transform(df)

    return X, y, preprocessor.feature_names_


def make_tier_label(
    y_price,
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Create premium-versus-standard labels.

    For training data, omit threshold so the training median is calculated.
    Pass the returned threshold when processing validation and test prices.
    """
    prices = np.asarray(y_price, dtype=float).reshape(-1)

    if len(prices) == 0:
        raise ValueError("y_price cannot be empty")

    if not np.all(np.isfinite(prices)):
        raise ValueError("y_price contains invalid values")

    if threshold is None:
        threshold = float(np.median(prices))
    else:
        threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    labels = (prices > threshold).astype(int)

    return labels, threshold


def train_val_test_split(
    X,
    y,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
):
    """
    Reproducibly split already transformed arrays.

    The preferred final pipeline uses split_dataframe before preprocessing.
    """
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if len(X_array) != len(y_array):
        raise ValueError("X and y must have the same number of rows")

    if not 0 <= val_size < 1 or not 0 <= test_size < 1:
        raise ValueError("Split sizes must be in [0, 1)")

    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    random_generator = np.random.default_rng(seed)
    shuffled_indices = random_generator.permutation(len(X_array))

    n_validation = int(round(len(X_array) * val_size))
    n_test = int(round(len(X_array) * test_size))
    n_train = len(X_array) - n_validation - n_test

    if n_train < 1:
        raise ValueError("Split sizes leave no training samples")

    train_indices = shuffled_indices[:n_train]
    validation_indices = shuffled_indices[
        n_train:n_train + n_validation
    ]
    test_indices = shuffled_indices[n_train + n_validation:]

    return (
        X_array[train_indices],
        y_array[train_indices],
        X_array[validation_indices],
        y_array[validation_indices],
        X_array[test_indices],
        y_array[test_indices],
    )


def standardize(X_tr, *others):
    """
    Standardize features using training statistics only.

    Returns:
        scaled_train, *scaled_other_arrays, training_mean, training_scale
    """
    training = np.asarray(X_tr, dtype=float)

    if training.ndim != 2:
        raise ValueError("X_tr must be two-dimensional")
    if not len(training) or not np.isfinite(training).all():
        raise ValueError("X_tr must be nonempty and finite")

    mean = np.mean(training, axis=0)
    scale = np.std(training, axis=0)

    # Constant dummy columns should not cause division by zero.
    scale = np.where(scale == 0.0, 1.0, scale)

    transformed = [(training - mean) / scale]

    for array in others:
        values = np.asarray(array, dtype=float)

        if values.ndim != 2:
            raise ValueError(
                "Every feature array must be two-dimensional"
            )

        if values.shape[1] != training.shape[1]:
            raise ValueError(
                "Every feature array must have the same columns"
            )
        if not np.isfinite(values).all():
            raise ValueError("Feature arrays must be finite")

        transformed.append((values - mean) / scale)

    return (*transformed, mean, scale)
