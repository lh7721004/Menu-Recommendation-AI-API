import tensorflow as tf
import numpy as np
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,Response
import uvicorn

model = tf.keras.models.load_model('./train_model.h5', compile=False)
app = FastAPI()

@app.route('/predict')
async def model_predict(request: Request):
    try:
        json_data = await request.json()
        value_list = np.array(list(json_data.values()), dtype=np.float32).reshape(1, -1)
        predict_result = model.predict(value_list)
        predict_class = np.argmax(predict_result, axis=1)[0]
        return JSONResponse(content={"recommended_menu":int(predict_class)},status_code=200)
    except ValueError:
        return Response(content="Invalid Value",status_code=400)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8060)