"""Задачи первой части лабораторной: Pandas и Titanic."""

from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def _to_int_keys(series: pd.Series) -> dict[int, float]:
    """Преобразует результат groupby-агрегации в словарь с целыми ключами классов."""
    result: dict[int, float] = {}
    for key, value in series.items():
        if pd.isna(key):
            continue
        result[int(key)] = float(value)
    return result


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    row_count = int(len(df))
    missing_by_column = {str(column): int(count) for column, count in df.isna().sum().items()}

    age = pd.to_numeric(df["Age"], errors="coerce")
    pclass = pd.to_numeric(df["Pclass"], errors="coerce")
    survived = pd.to_numeric(df["Survived"], errors="coerce")
    fare = pd.to_numeric(df["Fare"], errors="coerce")

    adults_over_30_count = int((age > 30).sum())

    mean_age_by_pclass = _to_int_keys(age.groupby(pclass).mean())

    survival_rate_by_pclass = _to_int_keys(survived.groupby(pclass).mean())

    highest_fares = [float(value) for value in fare.dropna().nlargest(5).to_list()]

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )