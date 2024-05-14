from flask import Flask

app = Flask(__name__)

# Importing views after creating the app to avoid circular imports
from app import views