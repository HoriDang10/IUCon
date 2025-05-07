from flask import Flask, jsonify, render_template, request
from flask_cors import CORS 

app = Flask(__name__)
cors = CORS(app, origins='*')

#route predict
@app.route("/predict", methods=["POST"])
#route for testing 
@app.route("/api/users", methods=['GET'])
def users(): 
    return jsonify(
        {
            "users": [
                'Mai', 
                'Tram Anh'
            ]
        }
    )
#route for model
@app.route("/chat", methods=['GET'])
def index_get():
    return render_template("chat.html")
if __name__ == "__main__": 
    app.run(debug=True)