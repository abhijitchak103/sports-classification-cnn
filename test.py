import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
# URL for AWS Lambda API
# url = 'https://fqt51agph3.execute-api.ap-south-1.amazonaws.com/test/predict'

data = {'url': 'https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1'}

result = requests.post(url, json=data).json()
print(result)