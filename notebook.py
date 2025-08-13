import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # This file is used to initialize the Marimo app.
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Marketing-Intelligence-A-Predictive-Model-for-Lead-Conversion""")
    return


# Table of Contents (added)
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Table of Contents
    - [1. Introduction](#intro)
    - [2. Data Audit & Preprocessing](#data-prep)
    - [3. Exploratory Data Analysis (EDA)](#eda)
    - [4. Feature Engineering & Encoding](#feature-eng)
    - [5. Train/Validation/Test Split](#split)
    - [6. Baseline Model — Logistic Regression](#baseline)
    - [7. Alternative Models (Optional)](#alternatives)
    - [8. Evaluation & Business Metrics](#evaluation)
    - [9. Calibration & Thresholds](#calibration)
    - [10. Lead Scoring & Segmentation](#scoring)
    - [11. Save Artifacts (Model & Reports)](#artifacts)
    - [12. Next Steps & Notes](#next-steps)
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="intro"></a>
    ## 📚 1.0 INTRODUCTION


    <div style="font-family: Avenir, sans-serif; font-size: 16px; line-height: 1.6; color: white; background-color: #333; padding: 10px; border-radius: 5px;">
    This section provides an overview of the dataset and the problem we are trying to solve. We will also discuss the data overview, project objective, methodology and the tools (libaries) we will use to solve the problem.

    </div>
    """
    )
    return

# Scope-aligned section anchors (placeholders to enable ToC navigation)
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="data-prep"></a>
    ## 2. Data Audit & Preprocessing
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="eda"></a>
    ## 3. Exploratory Data Analysis (EDA)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="feature-eng"></a>
    ## 4. Feature Engineering & Encoding
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="split"></a>
    ## 5. Train/Validation/Test Split
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="baseline"></a>
    ## 6. Baseline Model — Logistic Regression
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="alternatives"></a>
    ## 7. Alternative Models (Optional)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="evaluation"></a>
    ## 8. Evaluation & Business Metrics
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="calibration"></a>
    ## 9. Calibration & Thresholds
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="scoring"></a>
    ## 10. Lead Scoring & Segmentation
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="artifacts"></a>
    ## 11. Save Artifacts (Model & Reports)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <a id="next-steps"></a>
    ## 12. Next Steps & Notes
    """
    )
    return


app._unparsable_cell(
    r"""
    # Lead Conversion Predictor: A Machine Learning Approach to Marketing Effectiveness

    ## 🔗 Link to Dataset: [Kaggle - Leads Dataset](https://www.kaggle.com/ashydv/leads-dataset)

    ---

    ## 📌 Project Description

    This project focuses on analyzing lead behavioral patterns and evaluating the impact of different lead sources and engagement attributes on conversion outcomes. By using classification modeling, the project aims to:

    - Predict whether a lead will convert into a customer  
    - Identify high-impact features influencing conversions  
    - Score and segment leads based on their conversion likelihood  

    This tool is beneficial for marketing teams, sales strategists, and data-driven business managers seeking to optimize lead nurturing, campaign effectiveness, and customer acquisition.

    ---

    ## 📊 Dataset Summary

    This project uses a real-world dataset containing lead generation and marketing interaction data from an educational services provider. Each row represents a single lead and includes categorical and numerical attributes relevant to understanding conversion behavior and lead quality.

    ---

    ## 📁 Dataset Features

    | **Feature**                                    | **Description**                                                                 |
    |------------------------------------------------|---------------------------------------------------------------------------------|
    | `Prospect ID`                                  | Unique identifier for each prospect                                            |
    | `Lead Number`                                  | System-generated lead number                                                   |
    | `Lead Origin`                                  | Origin of the lead (e.g., Landing Page, API, etc.)                             |
    | `Lead Source`                                  | Source of lead acquisition (e.g., Google, Reference, Direct Traffic)           |
    | `Do Not Email`                                 | Indicates if the lead opted out of email communication                         |
    | `Do Not Call`                                  | Indicates if the lead opted out of phone calls                                 |
    | `Converted`                                    | **Target variable**: 1 if the lead converted, 0 otherwise                       |
    | `TotalVisits`                                  | Number of website visits by the lead                                           |
    | `Total Time Spent on Website`                  | Total time (in seconds) the lead spent on the website                          |
    | `Page Views Per Visit`                         | Average number of pages viewed per session                                     |
    | `Last Activity`                                | Most recent activity performed by the lead                                     |
    | `Country`                                      | Country of the lead                                                            |
    | `Specialization`                               | Area of interest or specialization (e.g., Finance, Marketing)                  |
    | `How did you hear about X Education`           | Marketing channel through which the lead heard about the service               |
    | `What is your current occupation`              | Current job role of the lead                                                   |
    | # `What matters most to you in choosing a course`| Lead’s top priority when selecting a course                                    |
    | `Search`                                       | Binary: 1 if searched for relevant course, 0 otherwise                          |
    | `Magazine`, `Newspaper Article`, `Newspaper`   | Binary indicators for traditional media exposure                               |
    | `X Education Forums`, `Digital Advertisement`  | Binary: Engaged with online media sources                                      |
    | `Through Recommendations`                      | Whether the lead came through a referral                                       |
    | `Receive More Updates About Our Courses`       | Indicates if the lead wants further course updates                             |
    | `Tags`                                         | Internal tags used by sales/marketing teams                                    |
    | `Lead Quality`                                 | Assessment of lead quality (e.g., High, Low, Might be)                         |
    | `Update me on Supply Chain Content`            | Opted to receive updates on supply chain content                               |
    | `Get updates on DM Content`                    | Opted to receive updates on digital marketing content                          |
    | `Lead Profile`                                 | System-generated profile of the lead                                           |
    | `City`                                         | City where the lead resides                                                    |
    | `Asymmetrique Activity/Profile Index/Score`    | Proprietary metrics assessing lead behavior and alignment                      |
    | `I agree to pay the amount through cheque`     | Binary: 1 if lead agreed to pay by cheque, 0 otherwise                         |
    | `A free copy of Mastering The Interview`       | Binary: 1 if free e-book was requested, 0 otherwise                            |
    | `Last Notable Activity`                        | Most notable recent interaction (e.g., SMS sent, Email opened)                 |

    ---

    ## 🧠 Target Variable

    # **`Converted`** – This is the classification target indicating whether the lead eventually became a customer.  
    - `1` = Converted  
    - `0` = Did not convert

    ---

    ## 👥 Use Cases

    - **Marketing Strategy Optimization** – Identify the most effective lead sources and nurturing tactics  
    - **Sales Prioritization** – Focus sales efforts on leads with the highest conversion probability  
    - **Customer Profiling** – Understand characteristics and behaviors of converting leads

    ---

    ## ⚙️ Techniques Used

    - Supervised Learning (Classification – Logistic Regression)
    - Data Preprocessing (Missing value handling, Encoding, Scaling)
    - Feature Importance and Interpretation
    - Evaluation Metrics (Accuracy, Precision, Recall, ROC-AUC)
    - Optional Deployment with Streamlit

    ---

    ## ✍️ Author

    **Teslim Adeyanju**  
    Chartered Accountant | Financial Data Analyst | Data Science Enthusiast  
    🌐 [adeyanjuteslim.co.uk](https://adeyanjuteslim.co.uk) | [LinkedIn](https://www.linkedin.com/in/adeyanjuteslimuthman)


    """,
    name="_"
)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(pd):
    df = pd.read_csv('Leads.csv', engine='pyarrow')
    df
    return (df,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    df.columns.to_list()
    return


if __name__ == "__main__":
    app.run()
