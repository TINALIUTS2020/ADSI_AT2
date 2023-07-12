pip install --upgrade firebase-admin

from datetime import datetime


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account.
cred = credentials.Certificate('path/to/serviceAccount.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()

doc_ref = db.collection("users").document("alovelace")

city_ref = db.collection("objects").document("some-id")
city_ref.update({"timestamp": firestore.SERVER_TIMESTAMP})

time_stamp = datetime.utcnow().isoformat()