import http.server
import socketserver
import os
import webbrowser
from .config import CONFIG_DIR, DEFAULT_PORT


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def list_directory(self, path):
        """Generate a custom directory listing page with a friendly message."""
        try:
            file_list = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None

        file_list.sort(key=lambda a: a.lower())

        # Custom HTML content
        message = """
        <html>
        <head>
            <title>WEAS Visualization Files</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                h1 { color: #333; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 10px 0; }
                a { text-decoration: none; color: #007bff; font-size: 18px; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>WEAS Visualization Files</h1>
            <p>Click on an HTML file below to visualize your atomic structure:</p>
            <ul>
        """

        for name in file_list:
            full_name = os.path.join(path, name)
            display_name = name
            if os.path.isdir(full_name):
                continue  # Skip directories, only show files

            # Add links to the files
            message += f'<li><a href="{display_name}">{display_name}</a></li>\n'

        message += """
            </ul>
        </body>
        </html>
        """

        encoded = message.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        """Suppress default logging to keep the output clean."""
        return


def run_http_server(PORT=DEFAULT_PORT):
    os.chdir(CONFIG_DIR)
    handler = CustomHTTPRequestHandler  # Use the custom handler

    # Try to find an available port if the default one is occupied
    while True:
        try:
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                url = f"http://localhost:{PORT}"

                # Display port forwarding instructions for remote access
                print(
                    "\nIf you are running this on a remote server, access it locally with SSH port forwarding:"
                )
                print("Run the following command on your local machine:")
                print(
                    f"    ssh -L {PORT}:localhost:{PORT} your_remote_user@your_remote_host"
                )
                print("\nThen open the same URL in your local browser.")

                print(f"\nServing at {url}")
                print("Open this URL in your browser to access the visualization.")

                # Automatically open the URL in the default browser (if running locally)
                if "SSH_CONNECTION" not in os.environ:
                    webbrowser.open(url)

                httpd.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {PORT} is occupied, trying {PORT + 1}...")
                PORT += 1
            else:
                raise
