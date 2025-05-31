from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import get_ai_response  # make sure this is the function we defined

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"answer": "Bạn chưa nhập gì cả."}), 400

    reply = get_ai_response(user_message)
    return jsonify({"answer": reply})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
