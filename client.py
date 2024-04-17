import requests

url = "http://localhost:8000"

response = requests.get(url)
print(response)
print(response.json())

pitcher_data = {"id": 696136}

response = requests.post(url + "/mix", json=pitcher_data)
print(response)
print(response.json())

pitch_data = {
    "pitcher": {"id": 696136},
    "pitch": {
        "release_speed": [70.9],
        "release_spin_rate": [2076.0],
        "pfx_x": [0.82],
        "pfx_z": [1.38],
        "stand": ["R"],
    },
}

response = requests.post(url + "/predict", json=pitch_data)
print(response)
print(response.json())
