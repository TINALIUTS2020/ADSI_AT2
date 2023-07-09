# for running prod
uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port $INTERNAL_PORT

# for running dev
uvicorn app.main:app --reload --port ${PORT}

