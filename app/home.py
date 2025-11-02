import streamlit as st
from pathlib import Path

def get_this_file_dir_abspath() -> str:
    return str(Path(__file__).resolve().parent)

THIS_FILE_DIR_ABSPATH = get_this_file_dir_abspath()

def render_home():
    st.divider()
    # ---- What is 30-day readmission? ----
    st.header("What is a 30-day hospital readmission?")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
    **Hospital readmission** is when a patient returns to the hospital **unplanned** after being discharged [1].  
    The most common window used by clinicians and health systems is **within 30 days** of discharge  
    (though 90-day, 6-month, and 1-year windows are also studied) [2].

    **Why 30 days? 💡** 
    It's a practical timeframe where post-discharge care, medication changes, follow-ups, and recovery challenges
    most strongly influence outcomes, making it a useful quality and planning metric [2].
    """
        )
    with col2:
        st.image(
            f"{THIS_FILE_DIR_ABSPATH}/images/hosp-readm-cycle.png",
            caption="Simplified illustration of the hospital readmission cycle within 30 days post-discharge.",
            use_container_width=True
        )

    st.divider()

    # ---- Motivation ----
    st.header("Why this matters?")
    st.markdown(
        """
Real-world data show that 30-day, all-cause readmissions are common and costly. For example:

- In the U.S., recent all-cause 30-day readmission are around **~14%**, what translates to 3.8M readmissions per year [5].  
- Readmissions often cost 12.4% **more** than the initial admission on average [5].

Besides, readmissions impact [5]:

-- **Patients:** Emotional/physical strain, loss of confidence, and higher risk of complications.  
-- **Capacity:** Readmissions consume scarce beds and shift staff away from other patients.

**Chronic conditions amplify risk**:  
Chronic obstructive pulmonary disease (COPD), heart failure (HF), diabetes, chronic kidney disease, and septicemia are among the leading contributors to early readmissions, with some conditions showing notably high 30-day and 1-year recurrence rates. [2]
        """
    )

    st.divider()

    # ---- How the predictive model helps ----
    st.header("How the predictive model helps")
    st.markdown(
        """
This system estimates whether a patient is **likely to be readmitted within 30 days** after discharge.
(Depending on your configuration, it can also score other windows such as 60 or 90 days.)

**Who benefits and how**
- **Physicians & Care Teams 🧑🏻‍⚕️:** Identify high-risk patients, schedule timely follow-ups, and adjust therapy.  
- **Patients & Caregivers 🤒:** Understand risk to support adherence and lifestyle changes.  
- **Hospital Operations 🏥:** Anticipate demand for beds and staff, improving resource planning.

The model's goal is not to replace clinical judgment, but to leverage statistics to **support** physicians and the care team in making informed decisions.
        """
    )

    # ---- What you'll see in this app ----
    st.subheader("What you'll see in this app")
    st.markdown(
        """
1. **Upload** CSV files with patient records (see sidebar).
2. **Run predictions** to get a per-patient 30-day readmission risk.
3. **Review specific patient results**, including explanations for their risk scores.
4. Check overall **model performance metrics** if true outcomes are provided.
        """
    )

    st.divider()

    st.header("Dataset used for model development 📊")
    st.markdown(
        """
The model was developed and validated using the publicly available MIMIC-III database [3, 4], which contains de-identified health data from over 40,000 patients.
This diverse dataset includes a wide range of clinical conditions, treatments, and outcomes, making it a valuable resource for training and evaluating predictive models in healthcare.
A subset of patients were used to train the readmission model, focusing on adult patients and with specific inclusion criteria.
    """
    )
    # ---- Ethics, fairness, and safe use ----
    st.header("Responsible use 💡")
    st.markdown(
        """
- **Clinical decision support, not diagnosis:** Always pair predictions with clinical judgment.  
- **Data quality:** Garbage in → garbage out; ensure up-to-date, accurate inputs.          """
    )
    st.warning("This tool does **not** provide medical advice. It is intended to support—not replace—professional clinical judgment.")

    st.subheader("Ready to begin? 🚀")
    st.markdown("Head to **Sidebar**, upload the required files and run predictions.")

    st.divider()

    st.header("References and further reading 📚")

    st.markdown(
        """
    1. **Dhaliwal, J. S.; Dang, A. K.** (2024). *Reducing Hospital Readmissions.* In **StatPearls [Internet]**. StatPearls Publishing, Treasure Island (FL). [Available online](https://www.ncbi.nlm.nih.gov/books/NBK606114/) — PMID: [39163436](https://pubmed.ncbi.nlm.nih.gov/39163436/)
    
    2. **Jiang, H. J.; Hensche, M.** (2023). *Characteristics of 30-Day All-Cause Hospital Readmissions.* HCUP Statistical Brief No. 304. Agency for Healthcare Research and Quality. [https://hcup-us.ahrq.gov/reports/statbriefs/sb304-readmissions-2016-2020.jsp](https://hcup-us.ahrq.gov/reports/statbriefs/sb304-readmissions-2016-2020.jsp)  

    3. **Weiss, A. J.; Jiang, H. J.** (2021). *Overview of Clinical Conditions With Frequent and Costly Hospital Readmissions by Payer, 2018.* HCUP Statistical Brief No. 278. Agency for Healthcare Research and Quality.[https://pubmed.ncbi.nlm.nih.gov/34460186/](https://pubmed.ncbi.nlm.nih.gov/34460186/)  

    4. **Johnson, A. E. W.; Pollard, T. J.; Shen, L.; Lehman, L.-W. H.; Feng, M.; Ghassemi, M.; Moody, B.; Szolovits, P.; Celi, L. A.; Mark, R. G.** (2016). *MIMIC-III, a freely accessible critical care database.* **Scientific Data**, 3, 160035. [DOI:10.1038/sdata.2016.35](https://doi.org/10.1038/sdata.2016.35)  

    5. **Johnson, A.; Pollard, T.; Mark, R.** (2016). *MIMIC-III Clinical Database (version 1.4).*PhysioNet. RRID: SCR_007345. [DOI:10.13026/C2XW26](https://doi.org/10.13026/C2XW26)  
    """
    )
