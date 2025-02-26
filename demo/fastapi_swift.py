from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import numpy as np
import io
import hashlib
import traceback
import json
import requests


app = FastAPI()
understand_image_and_question_url = "http://localhost:8000/v1/chat/completions"


@app.post("/understand_image_and_question/")
async def understand_image_and_question(
    file: UploadFile = File(...),
    question: str = Form(...),
    seed: int = Form(42),
    top_p: float = Form(0.95),
    temperature: float = Form(0.1)
):
    # images file max size 8mb
    maxfilesize = 8 * 1024 * 1024
    image_data = await file.read(maxfilesize)
    try:
        # Upload file directory
        imagedirectory = "./uploads/"
        # Need to match version with Swift service
        JanusVersion = "Janus-Pro-7B"
        #JanusVersion = "Janus-Pro-1B"

        file = Image.open(io.BytesIO(image_data))
        hash_obj = hashlib.md5()
        hash_obj.update(image_data)
        file_hash = hash_obj.hexdigest()
        filename =  imagedirectory + file_hash + ".png"
        file.save(filename, format='PNG')
        file.close()

        outjson =   {"model": JanusVersion,
            "messages": [{"role": "user", 
                          "content": "<image>"  + question} ],
                 "images": [filename]}
        
        outjson = json.dumps(outjson,ensure_ascii=False)
        response = requests.post(understand_image_and_question_url, data=outjson, stream=False)
        response_data = response.json()
        return response_data

    except  Exception as e:
        print("-----------------------------------------------")
        error_type = type(e).__name__
        error_msg = str(e)
        print(error_type)
        print(error_msg)
        traceback.print_exc()
        print("-----------------------------------------------")
        
        return "images file bad"




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
