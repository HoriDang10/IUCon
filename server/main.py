from flask import Flask, jsonify, render_template, request
from flask_cors import CORS 

app = Flask(__name__)
cors = CORS(app, origins='*')

@app.route("/")

def index_get():
    return render_template("chat.html")


def users(): 
    return jsonify(
        {
            "users": [
                'Mai', 
                'Tram Anh'
            ]
        }
    )

if __name__ == "__main__": 
    app.run(debug=True)