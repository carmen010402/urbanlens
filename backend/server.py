from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class RoverServer(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":

            data = {
                "status": "ready",
                "battery": 100,
                "distance": 0,
                "obstacles": 0
            }

            self.send_response(200)

            # Allow the dashboard to communicate with Python
            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.end_headers()

            self.wfile.write(
                json.dumps(data).encode()
            )

        else:

            self.send_response(404)

            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )

            self.end_headers()


server = HTTPServer(
    ("localhost", 5000),
    RoverServer
)

print("Rover backend running...")
print("http://localhost:5000")

server.serve_forever()