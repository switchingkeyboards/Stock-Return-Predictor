# Stock Return Predictor

## Spin up virtual environment

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Adding SICCD data to dataset

Feed WRDS dataset into data folder, then run `python3 helpers/export_permno.py`