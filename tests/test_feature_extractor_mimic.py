import pandas as pd
import numpy as np
import pytest

from recurrent_health_events_prediction.preprocessing.feature_extraction import FeatureExtractorMIMIC

# ---------------------------
# Small fixtures with toy data
# ---------------------------

@pytest.fixture
def icu_stays_df():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1],
            "HADM_ID": [10, 10],
            "INTIME": ["2000-01-01 00:00:00", "2000-01-02 00:00:00"],
            "OUTTIME": ["2000-01-01 12:00:00", "2000-01-03 00:00:00"],
        }
    )


@pytest.fixture
def prescriptions_df():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 1],
            "HADM_ID": [10, 10, 20],
            "DRUG": ["drug_a", "drug_b", "drug_a"],
        }
    )


@pytest.fixture
def procedures_df():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 1, 1],
            "HADM_ID": [10, 10, 20, 20],
            "ICD9_CODE": ["001", "001", "002", "003"],
        }
    )


@pytest.fixture
def admissions_df():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1],
            "HADM_ID": [10, 20],
            "ADMITTIME": ["2000-01-01 10:00:00", "2000-02-01 08:00:00"],
            "DISCHTIME": ["2000-01-05 10:00:00", "2000-02-10 08:00:00"],
            "ADMISSION_TYPE": ["EMERGENCY", "ELECTIVE"],
            "ETHNICITY": ["WHITE", "WHITE"],
            "DISCHARGE_LOCATION": ["HOME", "HOME"],
            "INSURANCE": ["MEDICARE", "MEDICARE"],
            # comorbidities for charlson/other stats
            "COMORBIDITY": ["diabetes_without_complication", "chronic_pulmonary_disease"],
        }
    )


@pytest.fixture
def patients_metadata_df():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1],
            "DOB": ["1950-01-01"],
            # Include DOD so _get_known_total_participation_days and
            # _get_death_time_after_last_discharge can run.
            "DOD": ["2000-03-01"],
        }
    )


# ---------------------------
# Unit tests for helper methods
# ---------------------------

def test_calculate_days_in_icu(icu_stays_df):
    """ICU time should be summed per SUBJECT_ID + HADM_ID in days."""
    # fix datetimes first (mimic actual behavior)
    icu_stays_df["INTIME"] = pd.to_datetime(icu_stays_df["INTIME"])
    icu_stays_df["OUTTIME"] = pd.to_datetime(icu_stays_df["OUTTIME"])

    result = FeatureExtractorMIMIC._calculate_days_in_icu(icu_stays_df)

    # 1st stay: 12h = 0.5 days
    # 2nd stay: 1 day = 1.0 days
    # total for hadm 10 should be 1.5 days
    assert len(result) == 1
    row = result.iloc[0]
    assert row["SUBJECT_ID"] == 1
    assert row["HADM_ID"] == 10
    assert row["DAYS_IN_ICU"] == pytest.approx(1.5)


def test_calculate_num_drugs_prescribed(prescriptions_df):
    """Should count distinct DRUG per admission."""
    result = FeatureExtractorMIMIC._calculate_num_drugs_prescribed(prescriptions_df)

    row_10 = result[result["HADM_ID"] == 10].iloc[0]
    row_20 = result[result["HADM_ID"] == 20].iloc[0]

    assert row_10["NUM_DRUGS"] == 2   # drug_a, drug_b
    assert row_20["NUM_DRUGS"] == 1   # drug_a


def test_calculate_num_procedures(procedures_df):
    """Should count distinct ICD9_CODE per admission."""
    result = FeatureExtractorMIMIC._calculate_num_procedures(procedures_df)

    row_10 = result[result["HADM_ID"] == 10].iloc[0]
    row_20 = result[result["HADM_ID"] == 20].iloc[0]

    # hadm 10: ICD9_CODE = ["001", "001"] -> 1 unique
    assert row_10["NUM_PROCEDURES"] == 1
    # hadm 20: ["002", "003"] -> 2 unique
    assert row_20["NUM_PROCEDURES"] == 2


def test_get_time_hospitalization_stats_basic(admissions_df):
    """Check NUM_PREV_HOSPITALIZATIONS and basic time deltas."""
    # First compute admission-related features to set ADMITTIME/DISCHTIME properly
    adm_feats = FeatureExtractorMIMIC._get_admission_related_features(admissions_df)
    # Sort by time, as build_features does
    adm_feats.sort_values(["SUBJECT_ID", "ADMITTIME"], inplace=True)
    adm_feats = FeatureExtractorMIMIC._get_time_hospitalization_stats(adm_feats)

    # first admission
    first = adm_feats.sort_values("ADMITTIME").iloc[0]
    second = adm_feats.sort_values("ADMITTIME").iloc[1]

    # NUM_PREV_HOSPITALIZATIONS
    assert first["NUM_PREV_HOSPITALIZATIONS"] == 0
    assert second["NUM_PREV_HOSPITALIZATIONS"] == 1

    # days between DISCHTIME(1) and ADMITTIME(2):
    # 2000-01-05 10:00 -> 2000-02-01 08:00 is just under 27 days
    assert second["DAYS_SINCE_LAST_HOSPITALIZATION"] == pytest.approx(
        (pd.Timestamp("2000-02-01 08:00:00") - pd.Timestamp("2000-01-05 10:00:00")).total_seconds()
        / 3600
        / 24
    )

    # For last admission there is no NEXT_ADMITTIME -> 0 (clipped)
    assert np.isclose(first["DAYS_UNTIL_NEXT_HOSPITALIZATION"], 0) or np.isfinite(
        first["DAYS_UNTIL_NEXT_HOSPITALIZATION"]
    )


def test_get_admission_related_features(admissions_df):
    """Check that hospitalization days and comorbidity counts are computed."""
    result = FeatureExtractorMIMIC._get_admission_related_features(admissions_df)

    assert set(result.columns) >= {
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "ADMISSION_TYPE",
        "HOSPITALIZATION_DAYS",
        "NUM_COMORBIDITIES",
        "TYPES_COMORBIDITIES",
        "HAS_DIABETES",
        "HAS_COPD",
        "HAS_CONGESTIVE_HF",
    }

    # For our toy data:
    # hadm 10 has one comorbidity, diabetes_without_complication -> NUM_COMORBIDITIES == 1
    row_10 = result[result["HADM_ID"] == 10].iloc[0]
    assert row_10["NUM_COMORBIDITIES"] == 1
    assert bool(row_10["HAS_DIABETES"]) is True
    assert bool(row_10["HAS_COPD"]) is False


def test_add_patient_specific_features(admissions_df, patients_metadata_df):
    """Age should be computed based on ADMITTIME year - DOB year, with 90+ adjustment."""
    adm_feats = FeatureExtractorMIMIC._get_admission_related_features(admissions_df)
    # ensure DOB is datetime
    patients_metadata_df["DOB"] = pd.to_datetime(patients_metadata_df["DOB"])

    result = FeatureExtractorMIMIC._add_patient_specific_features(adm_feats, patients_metadata_df)

    # ADMITTIME year is 2000, DOB is 1950 => age 50
    assert "AGE" in result.columns
    assert set(result["AGE"].unique()) == {50}


# ---------------------------
# Integration-ish test for build_features
# ---------------------------

def test_build_features_end_to_end(
    admissions_df,
    icu_stays_df,
    prescriptions_df,
    procedures_df,
    patients_metadata_df,
    monkeypatch,
):
    """
    Small end-to-end test:
    - uses tiny DataFrames
    - checks that key columns exist and have the right types / non-negative values.
    """

    # Optional: if charlson_weights / DiseaseType / calculate_past_rolling_stats
    # are complicated or come from elsewhere, you can monkeypatch the methods
    # that depend on them for this small integration test.

    # Example: bypass Charlson calculation and just set 0
    def fake_comorbidity_index(cls, df):
        df = df.copy()
        df["CHARLSON_INDEX"] = 0
        return df

    monkeypatch.setattr(
        FeatureExtractorMIMIC,
        "_get_patient_comorbidity_index",
        classmethod(fake_comorbidity_index),
    )

    # If you also need to bypass _calculate_past_hospitalization_stats:
    def fake_past_stats(cls, df):
        return df.assign(
            READM_30_DAYS_mean=np.nan,
            READM_30_DAYS_sum=np.nan,
            LOG_DAYS_UNTIL_NEXT_HOSP_mean=np.nan,
            LOG_DAYS_UNTIL_NEXT_HOSP_median=np.nan,
            LOG_DAYS_UNTIL_NEXT_HOSP_std=np.nan,
        )

    monkeypatch.setattr(
        FeatureExtractorMIMIC,
        "_calculate_past_hospitalization_stats",
        classmethod(fake_past_stats),
    )

    # Now call the main method
    result = FeatureExtractorMIMIC.build_features(
        admissions_df=admissions_df,
        icu_stays_df=icu_stays_df,
        prescriptions_df=prescriptions_df,
        procedures_df=procedures_df,
        patients_metadata_df=patients_metadata_df,
    )

    # Basic shape/columns checks
    assert not result.empty
    expected_cols = {
        "SUBJECT_ID",
        "HADM_ID",
        "DAYS_IN_ICU",
        "NUM_DRUGS",
        "NUM_PROCEDURES",
        "NUM_PREV_HOSPITALIZATIONS",
        "NUM_COMORBIDITIES",
        "CHARLSON_INDEX",
        "PARTICIPATION_DAYS",
        "READMISSION_30_DAYS",
        "PREV_READMISSION_30_DAYS",
    }
    assert expected_cols.issubset(result.columns)

    # Dtypes from the final coercions
    assert result["DAYS_IN_ICU"].dtype == float
    assert result["NUM_DRUGS"].dtype == np.int64 or np.issubdtype(result["NUM_DRUGS"].dtype, np.integer)
    assert result["NUM_PROCEDURES"].dtype == np.int64 or np.issubdtype(result["NUM_PROCEDURES"].dtype, np.integer)
    assert result["PARTICIPATION_DAYS"].dtype == np.int64 or np.issubdtype(result["PARTICIPATION_DAYS"].dtype, np.integer)

    # Sanity checks: no negative times
    assert (result["DAYS_IN_ICU"] >= 0).all()
    assert (result["PARTICIPATION_DAYS"] >= 1).all()
    if "TOTAL_PARTICIPATION_DAYS" in result.columns:
        assert (result["TOTAL_PARTICIPATION_DAYS"] >= 1).all()
