from datetime import datetime

import numpy as np
import pandas as pd

from etl.loader import prepare_for_sql


def test_prepare_for_sql_converts_all_pandas_missing_values_to_none():
    frame = pd.DataFrame(
        {
            "float_value": [1.5, np.nan],
            "datetime_value": [pd.Timestamp("2026-08-30"), pd.NaT],
            "nullable_value": pd.Series([4, pd.NA], dtype="Int64"),
        }
    )

    result = prepare_for_sql(frame)

    assert result.iloc[0].to_dict() == {
        "float_value": 1.5,
        "datetime_value": pd.Timestamp(datetime(2026, 8, 30)),
        "nullable_value": 4,
    }
    assert result.iloc[1].to_dict() == {
        "float_value": None,
        "datetime_value": None,
        "nullable_value": None,
    }
