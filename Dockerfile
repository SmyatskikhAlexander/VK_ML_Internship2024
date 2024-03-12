FROM python:3.9

RUN python -m pip install flask flask-cors numpy pandas scikit-learn catboost

WORKDIR /app

ADD server.py server.py
ADD model.pkl model.pkl

EXPOSE 5000

CMD [ "python",  "server.py"]
