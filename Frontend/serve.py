import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livereload import Server
from app import app

PORT = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    if "--port" in sys.argv:  # a second copy (e.g. for development) next to the usual one
        PORT = int(sys.argv[sys.argv.index("--port") + 1])
    server = Server(app.wsgi_app)
    server.watch("Frontend/templates/")
    server.watch("Frontend/static/")
    server.watch("YOLO_finetune/templates/")
    server.watch("YOLO_finetune/static/")
    server.watch("MedSAM2_Finetune/templates/")
    server.watch("MedSAM2_Finetune/static/")
    print(f"Serving BraTS Slice Viewer at http://localhost:{PORT}")
    print("Watching Frontend/ for changes (edit and the browser reloads).")
    server.serve(port=PORT, host="localhost", root=".")
