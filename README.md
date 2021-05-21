# Stock Return Predictor

## Spin up virtual environment

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data preprocessing and cleaning

```
python3 helpers/preprocessing.py
```

## Adding SICCD data to dataset

Export `permno.txt` from dataframe

```
python3 helpers/export_permno.py
```

Call WRDS service to convert `permno` into SICCD code.
Read the resulting `SICCD.csv` into helper to map sector information to dataframe.

```
python3 helpers/export_permno.py
```
