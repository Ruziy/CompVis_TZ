import io
import base64
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np
from yolov5 import YOLOv5
from ultralytics import YOLO
import os
import torch
import pathlib
import platform

app = FastAPI()

# Определяем корректный путь в зависимости от операционной системы
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_status = f"Using device: {device}"
print(device_status)

# Load models with specified device
model_yolov5x6 = YOLOv5(r"/app/weights/best_yolov5x6.pt", device=device)
model_yolov5 = YOLOv5(r"/app/weights/best_yolov5.pt", device=device)
model_yolov8 = YOLO(r"/app/weights/best_yolov8.pt").to(device)

class DetectionResult(BaseModel):
    image: str
    bboxes: list

def draw_bboxes(image: Image, boxes: list) -> Image:
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

@app.get("/", response_class=HTMLResponse)
async def main():
    return f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f8f9fa;
                    color: #333;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    flex-direction: column;
                }}
                h2 {{
                    color: #444;
                }}
                form {{
                    text-align: center;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    background-color: #ffffff;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    width: 300px;
                }}
                select, input[type="file"], input[type="submit"] {{
                    width: 100%;
                    padding: 10px;
                    margin-top: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <h2>Загрузка изображения для детекции объектов</h2>
            <p>{device_status}</p>
            <form action="/detect" method="post" enctype="multipart/form-data">
                <label for="model">Выберите модель:</label>
                <select name="model" id="model">
                    <option value="yolov5">YOLOv5</option>
                    <option value="yolov5x6">YOLOv5x6</option>
                    <option value="yolov8">YOLOv8</option>
                </select>
                <br><br>
                <input name="file" type="file" required>
                <input type="submit" value="Обработать">
            </form>
        </body>
    </html>
    """

@app.post("/detect", response_class=HTMLResponse)
async def detect(file: UploadFile = File(...), model: str = Form(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image_np = np.array(image)
        bboxes = []

        # Check model selection and apply respective prediction
        if model == "yolov5":
            results = model_yolov5.predict(image_np, size=640)
            for box in results.xyxy[0]:
                bboxes.append([int(coord) for coord in box[:4]])
        
        elif model == "yolov5x6":
            results = model_yolov5x6.predict(image_np, size=640)
            for box in results.xyxy[0]:
                bboxes.append([int(coord) for coord in box[:4]])

        elif model == "yolov8":
            results = model_yolov8.predict(image, imgsz=640, save=False)
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes.xyxy.tolist():
                        bboxes.append([int(coord) for coord in box])

        masked_image = draw_bboxes(image.copy(), bboxes)
        
        buffered = io.BytesIO()
        masked_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return HTMLResponse(content=f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f8f9fa;
                        color: #333;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        flex-direction: column;
                    }}
                    h2 {{
                        color: #444;
                    }}
                    .container {{
                        text-align: center;
                        width: 80%;
                        max-width: 600px;
                        padding: 20px;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        background-color: #ffffff;
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    }}
                    img {{
                        border: 3px solid #333;
                        border-radius: 8px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        max-width: 100%;
                    }}
                    textarea {{
                        width: 100%;
                        height: 150px;
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        font-family: monospace;
                    }}
                    button {{
                        padding: 10px 20px;
                        font-size: 14px;
                        color: #333;
                        background-color: #ddd;
                        border: 1px solid #bbb;
                        border-radius: 5px;
                        cursor: pointer;
                        margin-top: 10px;
                    }}
                    button:hover {{
                        background-color: #ccc;
                    }}
                </style>
                <script>
                    function copyToClipboard() {{
                        const base64Text = document.getElementById("base64-text");
                        base64Text.select();
                        document.execCommand("copy");
                        alert("Base64-код скопирован в буфер обмена!");
                    }}
                </script>
            </head>
            <body>
                <div class="container">
                    <h2>Результаты детекции</h2>
                    <p>{device_status}</p>
                    <p><b>Base64-код изображения:</b></p>
                    <textarea id="base64-text" readonly>{img_str}</textarea>
                    <button onclick="copyToClipboard()">Копировать Base64-код</button>
                    <h3>Изображение с наложенными рамками:</h3>
                    <img src="data:image/jpeg;base64,{img_str}" alt="Detected Image">
                </div>
            </body>
        </html>
        """)
    except Exception as e:
        return HTMLResponse(content=f"<h2>Ошибка: {str(e)}</h2>", status_code=500)


