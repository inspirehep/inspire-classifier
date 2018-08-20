from flask import Flask
import datetime
app = Flask(__name__)

@app.route("/")
def date():
	now = datetime.datetime.now()
	return str(now)