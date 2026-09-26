# OptiML Suite

A no-code AutoML web app: upload a CSV, clean and explore the data, and automatically train and compare machine learning models.

> **Companion app:** [OptiML_App](https://github.com/2004ranjith/OptiML_App) is the prediction app that loads the model package exported by this suite ([live demo](https://optimlsuite-app.streamlit.app/)).

## The problem

Building a machine learning model usually needs coding for data cleaning, exploration and model selection. OptiML Suite lets users without ML coding experience go from a raw CSV file to a trained, downloadable model through a simple web interface.

## How it works

1. **Upload File** – upload a CSV dataset and preview it.
2. **Data Profiling and Cleaning** – one-click cleaning (removes duplicates, empty and constant columns; splits date/time columns into parts; converts currency strings to numbers; normalizes text; fills missing values with mean / most frequent value). Download the cleaned CSV and view a profile report (dataset statistics, per-variable stats and charts, correlation heatmap).
3. **Data Analysis** – pick X and Y columns; the app suggests suitable interactive chart types based on the detected column types.
4. **Model Generation** – choose a target column. Numeric targets run regression; binary/categorical targets run classification. Data is preprocessed (IQR outlier removal, label encoding, scaling) and PyCaret trains and compares multiple models, showing a leaderboard and the best model.
5. **Export** – download `model_package.zip` (best model, label encoders, input-schema JSON) and use it in the companion [OptiML_App](https://github.com/2004ranjith/OptiML_App) for predictions.

## Tech stack

- **Python**, **Streamlit** (+ streamlit-option-menu) – web UI
- **Pandas**, **NumPy** – data loading and cleaning
- **scikit-learn** – imputation (SimpleImputer), LabelEncoder, StandardScaler
- **PyCaret** – automated model training and comparison (regression & classification)
- **Matplotlib**, **Seaborn** – profiling charts and correlation heatmap
- **Plotly** – interactive analysis charts
- **python-dateutil** – date/time detection
- SciPy is included as a dependency of the ML libraries

## Project structure

```
app.py               # Streamlit entry point and navigation
autocleandata.py     # Automatic data cleaning
profilingdata.py     # Profile report (statistics, variables, correlations)
data_ana.py          # Interactive chart builder
col_datatype.py      # Column type detection (Numeric / Categorical / Binary / Text)
preprocessingdata.py # Outlier removal, encoding, scaling
mlmodels.py          # PyCaret training and model package export
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations and future improvements

- Cleaning uses fixed rules (e.g. mean/mode imputation, IQR outlier removal) for every dataset; making these configurable would suit more data types.
- The numeric scaler and target encoder are not yet included in the exported package; saving the full preprocessing pipeline would keep predictions consistent with training.
- Model selection uses PyCaret's `compare_models` only; hyperparameter tuning and model explainability (feature importance) are natural next steps.
- Supports regression and classification; clustering and time-series tasks could be added.
- Automated tests and more detailed documentation would improve reliability.

## Team

Built as a B.Tech team project at SRM Institute of Science and Technology, Ramapuram.

Team: SJ Yogesh ([github.com/sjyogesh23](https://github.com/sjyogesh23)) and B.S Ranjith ([github.com/2004ranjith](https://github.com/2004ranjith)).
