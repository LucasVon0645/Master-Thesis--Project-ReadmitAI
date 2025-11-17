import pandas as pd
import numpy as np
import pytest
from recurrent_health_events_prediction.preprocessing.preprocessors import DataPreprocessorMIMIC

@pytest.fixture
def prepared_events_df():
    config = {"consider_death_after_discharge_gt": 2}
    processor = DataPreprocessorMIMIC(config)

    df = pd.DataFrame({
        'SUBJECT_ID': [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7],
        'HADM_ID':    ['a','b','c','d','e','f','g','h','i','j','k','l'],
        'ADMITTIME':  [1,  2,  3, 1,  2, 1, 1,  2, 1,  2, 1, 1],
        'READMISSION_EVENT': [1,1,0,1,0,0,0,0,0,0,0,0],
        'IN_HOSP_DEATH_EVENT': [0,0,0,0,1,0,0,0,0,0,0,0],
        'AFTER_HOSP_DEATH_EVENT': [0,0,0,0,0,0,0,1,0,1,1,1],
        'DEATH_TIME_AFTER_LAST_DISCHARGE': [
            None, None, None, None, None, None, None, 5, None, 2, 5, 2
        ]
    })

    df = processor._define_last_events(df)
    return processor, df

def test_define_last_events(prepared_events_df):
    config = {"consider_death_after_discharge_gt": 2}
    processor = DataPreprocessorMIMIC(config)

    df = pd.DataFrame({
        'SUBJECT_ID': [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7],
        'HADM_ID':    ['a','b','c','d','e','f','g','h','i','j','k','l'],
        'ADMITTIME':  [1,  2,  3, 1,  2, 1, 1,  2, 1,  2, 1, 1],
        'READMISSION_EVENT': [1,1,0,1,0,0,0,0,0,0,0,0],
        'IN_HOSP_DEATH_EVENT': [0,0,0,0,1,0,0,0,0,0,0,0],
        'AFTER_HOSP_DEATH_EVENT': [0,0,0,0,0,0,0,1,0,1,1,1],
        'DEATH_TIME_AFTER_LAST_DISCHARGE': [
            None, None, None, None, None, None, None, 5, None, 2, 5, 2
        ]
    })

    result_df = processor._define_last_events(df)

    # Explanation:
    # SUBJECT_ID 1: multiple → penultimate = 'b'
    # SUBJECT_ID 2: 'e' invalid → penultimate = 'd'
    # SUBJECT_ID 3: single, valid → 'f'
    # SUBJECT_ID 4: multiple → penultimate = 'g'
    # SUBJECT_ID 5: 'j' invalid → penultimate = 'i'
    # SUBJECT_ID 6: single, AFTER_HOSP_DEATH, death_time = 5 → valid → 'k'
    # SUBJECT_ID 7: single, AFTER_HOSP_DEATH, death_time = 2 → invalid → none

    expected = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    assert result_df['IS_LAST_EVENT'].tolist() == expected

def test_define_historical_past_events(prepared_events_df):
    processor, df = prepared_events_df

    result_df = processor._define_historical_past_events(df)

    # Based on the IS_LAST_EVENT logic:
    # SUBJECT_IDs and their IS_LAST_EVENT ADMITTIMEs:
    # 1 → 'b' (ADMITTIME 2) → historical: 'a' (1)
    # 2 → 'd' (1) → historical: none
    # 3 → 'f' (1) → historical: none
    # 4 → 'g' (1) → historical: none
    # 5 → 'i' (1) → historical: none
    # 6 → 'k' (1) → historical: none
    # 7 → no IS_LAST_EVENT → all NaN → historical: none

    expected_historical = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result_df['IS_HISTORICAL_EVENT'].tolist() == expected_historical
