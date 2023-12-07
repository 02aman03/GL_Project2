import requests
r = requests.post('http://127.0.0.1:5000/predict', json={"my_data": "[[1,7,5,7]]"})
print("Status Code", r.status_code)
print("\nYour answer is\n", r.text)