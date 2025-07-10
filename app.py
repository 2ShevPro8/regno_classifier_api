import base64
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from handler import Handler

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

handler = Handler()


@app.post("/predict_b64")
async def predict_b64(request: Request):
    try:
        data_b64 = await request.body()
        decoded_json = base64.b64decode(data_b64).decode("utf-8")
        data = json.loads(decoded_json)

        result = await handler.pipe(data)

        result_json = json.dumps(result).encode("utf-8")
        result_b64 = base64.b64encode(result_json)

        return Response(content=result_b64, media_type="text/plain")

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during prediction: {str(e)}"
        )


@app.get("/health_check")
async def health_check():
    return Response(status_code=200)


@app.get("/")
async def root():
    return {"message": "Welcome to the API"}
