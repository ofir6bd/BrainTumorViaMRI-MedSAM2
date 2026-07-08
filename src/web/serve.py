import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livereload import Server
from app import app

PORT = 5000

if __name__ == "__main__":
    server = Server(app.wsgi_app)
    server.watch("src/web/templates/")
    server.watch("src/web/static/")
    server.watch("src/web/app.py")
    print(f"Serving BraTS Slice Viewer at http://localhost:{PORT}")
    print("Watching src/web/ for changes (edit and the browser reloads).")
    webbrowser.open(f"http://localhost:{PORT}")
    server.serve(port=PORT, host="localhost", root=".")
