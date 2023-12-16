import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': '../data/test/baseball/4.jpg'}

result = requests.post(url, json=data).json()
print(result)