"""
Brainnet Web GUI - Browser-based launcher
Opens a simple HTML page in the browser for symbol selection
"""

import webbrowser
import http.server
import socketserver
import threading
import time
import urllib.parse
from typing import Optional

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 BRAINNET</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a1628 0%, #1a2744 50%, #0a1628 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #ffffff;
        }
        
        .container {
            text-align: center;
            padding: 40px;
        }
        
        .header {
            margin-bottom: 40px;
        }
        
        .title {
            font-size: 4rem;
            font-weight: 800;
            color: #00d4ff;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #8899aa;
            letter-spacing: 2px;
        }
        
        .instruction {
            font-size: 1rem;
            color: #ffffff;
            margin-bottom: 30px;
            letter-spacing: 3px;
            font-weight: 600;
        }
        
        .buttons {
            display: flex;
            flex-direction: column;
            gap: 15px;
            align-items: center;
        }
        
        .market-btn {
            width: 300px;
            padding: 25px 40px;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            background: #1a2744;
            border: 2px solid #2a3f5f;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: block;
        }
        
        .market-btn:hover {
            background: #2a3f5f;
            border-color: #00d4ff;
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        }
        
        .market-btn .name {
            display: block;
            font-size: 0.8rem;
            font-weight: 400;
            color: #8899aa;
            margin-top: 5px;
        }
        
        .exit-btn {
            margin-top: 30px;
            width: 300px;
            padding: 20px 40px;
            font-size: 1.2rem;
            font-weight: 700;
            color: #ffffff;
            background: #ff4466;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: block;
        }
        
        .exit-btn:hover {
            background: #cc3355;
            transform: translateY(-2px);
        }
        
        .footer {
            margin-top: 50px;
            font-size: 0.9rem;
            color: #8899aa;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .loading {
            display: none;
            font-size: 1.5rem;
            color: #00d4ff;
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠 BRAINNET</div>
            <div class="subtitle">ADAPTIVE QUANT TRADING SYSTEM</div>
        </div>
        
        <div class="instruction">SELECT A MARKET TO ANALYZE</div>
        
        <div class="buttons">
            <a href="/select?symbol=BTC-USD" class="market-btn">
                BTC
                <span class="name">Bitcoin</span>
            </a>
            <a href="/select?symbol=ETH-USD" class="market-btn">
                ETH
                <span class="name">Ethereum</span>
            </a>
            <a href="/select?symbol=ES=F" class="market-btn">
                ES
                <span class="name">S&P 500 Futures</span>
            </a>
            <a href="/select?symbol=NQ=F" class="market-btn">
                NQ
                <span class="name">Nasdaq Futures</span>
            </a>
            <a href="/exit" class="exit-btn">✕ EXIT</a>
        </div>
        
        <div class="loading" id="loading">Launching analysis...</div>
        
        <div class="footer">
            Powered by Phi-3.5-Mini • GAF Pattern Recognition
        </div>
    </div>
    
    <script>
        document.querySelectorAll('.market-btn, .exit-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                document.getElementById('loading').style.display = 'block';
            });
        });
    </script>
</body>
</html>
"""

CLOSE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Brainnet</title>
    <style>
        body {
            background: #0a1628;
            color: #00d4ff;
            font-family: sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .msg {
            text-align: center;
            font-size: 2rem;
        }
    </style>
</head>
<body>
    <div class="msg">
        <p>🧠 Analysis started!</p>
        <p style="font-size: 1rem; color: #8899aa;">Check your terminal for results.</p>
        <p style="font-size: 0.8rem; color: #8899aa; margin-top: 20px;">You can close this tab.</p>
    </div>
    <script>
        // Auto-close after 3 seconds
        setTimeout(() => window.close(), 3000);
    </script>
</body>
</html>
"""


class GUIHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for the GUI server."""
    
    selected_symbol = None
    should_exit = False
    
    def log_message(self, format, *args):
        """Suppress logging."""
        pass
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
            
        elif parsed.path == "/select":
            query = urllib.parse.parse_qs(parsed.query)
            symbol = query.get("symbol", [None])[0]
            GUIHandler.selected_symbol = symbol
            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(CLOSE_HTML.encode())
            
        elif parsed.path == "/exit":
            GUIHandler.should_exit = True
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body style='background:#0a1628;color:#fff;text-align:center;padding-top:200px;'><h1>Goodbye!</h1><script>setTimeout(()=>window.close(),1000)</script></body></html>")
            
        else:
            self.send_response(404)
            self.end_headers()


def launch_web_gui(port: int = 8765) -> Optional[str]:
    """
    Launch the web-based GUI selector.
    
    Args:
        port: Port to run the local server on
        
    Returns:
        Selected symbol string or None if exited
    """
    # Reset state
    GUIHandler.selected_symbol = None
    GUIHandler.should_exit = False
    
    # Create server
    with socketserver.TCPServer(("", port), GUIHandler) as httpd:
        # Open browser
        url = f"http://localhost:{port}"
        print(f"Opening browser: {url}")
        webbrowser.open(url)
        
        # Wait for selection
        while GUIHandler.selected_symbol is None and not GUIHandler.should_exit:
            httpd.handle_request()
        
        return GUIHandler.selected_symbol


if __name__ == "__main__":
    selected = launch_web_gui()
    if selected:
        print(f"Selected: {selected}")
    else:
        print("Exited")

