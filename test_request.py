import requests
files = {'123.csv': open('123.csv','rb')}
r = requests.post('http://localhost:5000/predict', files=files)
print(r.text)