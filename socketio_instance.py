# socketio_instance.py
from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="http://localhost:3000")  # Create the SocketIO instance
