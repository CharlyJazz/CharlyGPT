# Model Sandbox API

## Run

```bash
pip install -r requirements.txt
set MODEL_NAME=tu-modelo
uvicorn app:app --host 0.0.0.0 --port 8000
```

`MODEL_NAME` can be a local path or a Hugging Face model id.

## Endpoint

`POST /chat`

```json
{
  "prompt": "Hola mundo",
  "max_new_tokens": 256
}
```
