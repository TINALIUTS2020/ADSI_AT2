#!/bin/bash
python -m pip install -r requirements.txt

uvicorn app.main:app --reload --port ${PORT}