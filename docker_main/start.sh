#!/bin/sh

# Запустите uvicorn с подстановкой переменных окружения HOST и PORT
uvicorn API_main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
