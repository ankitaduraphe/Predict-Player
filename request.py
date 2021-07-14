import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'playerid':1, 'thirtyplus':1, 'fiftyplus':0, 'seventyplus':0, 'strikerate':1, 'manofthematch':0, 'wicketsinhand':6, 'matchtype':3, 'inningstype':0})

print(r.json())