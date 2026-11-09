from __future__ import annotations

import math

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

_FEATURES: tuple[str, ...] = ("FSIQ", "VIQ", "PIQ", "Weight", "Height")
_TARGET = "MRI_Count"
_NA_TOKENS = ["NA", "N/A", "na", "n/a", "NaN", "nan", "?", ".", ""]

_MALE_TOKENS = {"male", "m", "man", "1", "1.0", "м", "мужской", "мужчина"}
_FEMALE_TOKENS = {"female", "f", "woman", "2", "2.0", "ж", "женский", "женщина"}


def _load_brain_table(csv_path: str) -> pd.DataFrame:
    """Читает brainsize.txt (tab-разделитель) с корректной обработкой пропусков NA.

    Если файл фактически разделён произвольными пробелами (частый вариант
    выгрузки brainsize), выполняется повторное чтение с sep=r"\\s+".
    """
    df = pd.read_csv(csv_path, sep="\t", na_values=_NA_TOKENS, skipinitialspace=True)
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep=r"\s+", na_values=_NA_TOKENS)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _gender_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Возвращает булевы маски мужчин и женщин независимо от способа кодировки пола."""
    gender = df["Gender"].astype(str).str.strip().str.lower()
    men_mask = gender.isin(_MALE_TOKENS)
    women_mask = gender.isin(_FEMALE_TOKENS)
    return men_mask, women_mask


def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляции Пирсона признаков с MRI_Count внутри одной группы.

    Для каждой пары используется попарное удаление пропусков (pairwise complete),
    как это делает DataFrame.corr(method="pearson").
    """
    columns = [column for column in (*_FEATURES, _TARGET) if column in group.columns]
    numeric = group[columns].apply(pd.to_numeric, errors="coerce")

    correlations: dict[str, float] = {}
    for feature in _FEATURES:
        if feature not in numeric.columns or _TARGET not in numeric.columns:
            continue
        pair = numeric[[feature, _TARGET]].dropna()
        enough_data = (
            len(pair) >= 2
            and pair[feature].nunique() > 1
            and pair[_TARGET].nunique() > 1
        )
        correlations[feature] = (
            float(pair[feature].corr(pair[_TARGET], method="pearson"))
            if enough_data
            else float("nan")
        )
    return correlations


def _strongest_feature(*groups: dict[str, float]) -> str:
    """Признак с максимальным |r| среди объединённых результатов всех групп."""
    best_feature = ""
    best_value = -math.inf
    for group in groups:
        for feature, value in group.items():
            if feature == _TARGET or value is None or math.isnan(value):
                continue
            if abs(value) > best_value:
                best_feature, best_value = feature, abs(value)
    return best_feature


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = _load_brain_table(data.csv_path)
    men_mask, women_mask = _gender_masks(df)

    men = df.loc[men_mask]
    women = df.loc[women_mask]

    men_mri_correlation = _mri_correlations(men)
    women_mri_correlation = _mri_correlations(women)

    return BrainCorrelationSummary(
        men_count=int(men_mask.sum()),
        women_count=int(women_mask.sum()),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=_strongest_feature(women_mri_correlation, men_mri_correlation),
    )