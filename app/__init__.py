from socket import SocketIO
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Load instance-specific configurations
    app.config.from_pyfile('config.py')

    from .api import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app
