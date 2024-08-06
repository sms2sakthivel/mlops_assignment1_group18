FROM python:3.9

RUN mkdir aditya/

WORKDIR /aditya

COPY Code/train.py Code/train.py
COPY Service/load_model.py Service/load_model.py
COPY Service/router.py Service/router.py
COPY app.py app.py
COPY requirements.txt requirements.txt
COPY Model/rf_model.pkl Model/rf_model.pkl
COPY Model/scaler.pkl Model/scaler.pkl

RUN python3 -m pip install -r requirements.txt

# ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# EXPOSE 8000