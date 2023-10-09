import json
from flask import Flask, jsonify, request

app = Flask(__name__)

with open('sampleInput.json', 'r') as f:
    sampleInput = json.load(f)

with open('sampleOutput.json', 'r') as f:
    sampleOutput = json.load(f)


@app.route('/api', methods=['GET'])
def api():
    return jsonify(sampleOutput)


@app.route('/api', methods=['POST'])
def api_post():
    data = sampleInput
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=5000)
