import requests

base_url = "http://127.0.0.1:8000"
url = f"{base_url}/predict"

image_path = "/home/mattias/dtu/mlops/MLOps_project/data/raw/ai/_a_peaceful_countryside_Beaches_with_a__2edf54e5-d4e4-4507-abde-4f81b9754b54.png"

with open(image_path, "rb") as f:
    files = {
        "file": (
            "image.png",  # filename
            f,  # file object
            "image/png",  # content type (IMPORTANT)
        )
    }
    resp = requests.post(url, files=files)

print("POSTing to:", url)
print(resp.status_code)
print(resp.json())
