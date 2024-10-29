# Используйте официальный образ Python в качестве базового
FROM python:3.10

# Установите рабочую директорию внутри контейнера
WORKDIR /app

# Копируйте файл требований в контейнер
COPY requirements.txt ./

# Копируйте веса моделей в папку weights
RUN mkdir -p weights
COPY weights/best_yolov5.pt weights/best_yolov8.pt weights/

# Установите системные зависимости, включая OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Обновите pip и установите зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir torchvision==0.15.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Копируйте остальные файлы вашего приложения
COPY . .

# Добавьте команду для выполнения start.sh
RUN chmod +x ./start.sh

# Укажите запуск через start.sh
CMD ["./start.sh"]



