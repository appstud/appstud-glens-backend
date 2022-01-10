curl -H "Content-Type: application/json" -d '{"instances" : [{"input_image": "$(base64 ../../beardDataset/nobeard/098260.jpg)"}]}' -X POST http://localhost:8501/v1/models/beardModel:predict -v

