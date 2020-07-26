from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/forex-expert/whatshouldido', methods=['POST'])
def what_should_i_do():
    return jsonify({"status": "success"})


@app.route('/forex-expert/upload', methods=['POST'])
def upload():
    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
