#!/bin/bash
python -m pip install -r requirements.txt

uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port ${PORT} --reload