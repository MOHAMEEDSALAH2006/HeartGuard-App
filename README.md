# Heart Disease Predictor — Streamlit UI-only phase

This version intentionally contains **only the Streamlit interface**. It does not train, download, load, or use any Machine Learning model. It does not include a dataset, model pickle, scaler artifact, fake prediction, or fake confidence score.

## Included flow

Home → Choose User Type → Normal User or Medical Specialist → Input Form → Review Inputs → Prediction Result Placeholder.

The Normal User flow displays only the five basic inputs identified in the supplied table:

- Age
- Sex
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar

The Medical Specialist flow displays all 13 supplied features, organized into patient profile, symptoms, clinical measurements, and medical history.

## Run

```bash
cd heart-disease-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Connect the final model later

The integration boundary is:

```text
prediction/predictor.py
```

The UI calls:

```python
from prediction.predictor import predict_heart_disease
result = predict_heart_disease(input_data)
```

At this stage the function returns `None` by design. Later, replace its body with loading and applying the user's final trained model and preprocessing artifacts. Do not change the UI flow unless the final model requires a different input contract.
