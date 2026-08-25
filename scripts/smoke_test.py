import base64
import json
import urllib.request

base = "http://127.0.0.1:8000"


def get(path: str):
    with urllib.request.urlopen(base + path) as r:
        return json.load(r)


health = get("/api/health")
print("health", health)
stats = get("/api/stats")
print("storms", stats["n_storms"], "range", stats["season_min"], stats["season_max"], "strongest", stats["strongest"]["name"])
storms = get("/api/storms?limit=5")["storms"]
print("sample storms", [f"{s['name']} {s['season']} {s['category']}" for s in storms])
sid = storms[0]["sid"]
pred = get(f"/api/predict/{sid}")
print(
    "predict",
    pred["storm"]["name"],
    "lstm",
    len(pred["lstm"]["forecast"]),
    "peak",
    pred["lstm"]["peak_forecast_wind_kt"],
    "threat",
    pred["lstm"]["land_threat"]["level"],
    "analogs",
    [a.get("name") for a in pred["analog"]["analogs"][:4]],
)
sample = get("/api/sample-image")
print("sample", {k: sample["label"][k] for k in sample["label"] if k != "pattern_code"})
raw = base64.b64decode(sample["image"].split(",", 1)[1])
boundary = "----CycloneBound"
body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"s.png\"\r\nContent-Type: image/png\r\n\r\n"
).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    base + "/api/identify",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
with urllib.request.urlopen(req) as r:
    ident = json.load(r)
print(
    "identify",
    ident["is_cyclone"],
    ident["pattern"]["name"],
    ident["category"]["code"],
    ident["wind_kt"],
    "cam",
    bool(ident.get("gradcam")),
)
