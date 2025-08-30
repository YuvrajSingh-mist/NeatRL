#!/usr/bin/env python3
"""
Simple web server for NeatRL game interface
Serves the HTML game interface and handles WebSocket connections
"""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path

class GameWebServer:
    """Web server for NeatRL game interface"""
    
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.clients = set()
        self.current_frame = None
        self.game_info = {}
        self.running = False
        
    async def start_websocket_server(self):
        """Start the WebSocket server"""
        self.running = True
        async with websockets.serve(self.handle_client, self.host, self.port + 1):
            print(f"WebSocket server running on ws://{self.host}:{self.port + 1}")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Handle client messages (e.g., keyboard input)
                data = json.loads(message)
                if data.get("type") == "keyboard":
                    await self.handle_keyboard_input(data["key"])
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
    
    async def handle_keyboard_input(self, key: str):
        """Handle keyboard input from web client"""
        # This would be connected to the game loop
        print(f"Received keyboard input: {key}")
    
    def update_frame(self, frame: np.ndarray, game_info: dict = None):
        """Update the current frame and game info"""
        self.current_frame = frame
        if game_info:
            self.game_info = game_info
    
    async def broadcast_frame(self):
        """Broadcast current frame to all connected clients"""
        if self.current_frame is None or not self.clients:
            return
        
        try:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Create message with frame and game info
            message = {
                "type": "frame",
                "frame": jpg_as_text,
                "game_info": self.game_info,
                "timestamp": time.time()
            }
            
            # Broadcast to all clients
            if self.clients:
                await asyncio.gather(
                    *[client.send(json.dumps(message)) for client in self.clients],
                    return_exceptions=True
                )
        except Exception as e:
            print(f"Error broadcasting frame: {e}")
    
    async def broadcast_loop(self):
        """Continuous broadcast loop"""
        while self.running:
            await self.broadcast_frame()
            await asyncio.sleep(0.033)  # ~30 FPS

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler to serve the game interface"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve the game interface
            self.path = '/game_interface.html'
        elif self.path.startswith('/game_interface.html'):
            # Serve the game interface with proper headers
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Read and serve the game interface HTML
            try:
                # Get the current working directory
                import os
                current_dir = os.getcwd()
                print(f"Current directory: {current_dir}")
                
                # Try multiple possible paths
                possible_paths = [
                    os.path.join(current_dir, 'frontend', 'game_interface.html'),
                    os.path.join(current_dir, 'game_interface.html'),
                    'frontend/game_interface.html',
                    './frontend/game_interface.html',
                    '../frontend/game_interface.html',
                    'game_interface.html'
                ]
                
                html_content = None
                for path in possible_paths:
                    print(f"Trying path: {path}")
                    try:
                        with open(path, 'rb') as f:
                            html_content = f.read()
                            print(f"Successfully loaded from: {path}")
                            break
                    except FileNotFoundError:
                        print(f"File not found: {path}")
                        continue
                
                if html_content:
                    self.wfile.write(html_content)
                else:
                    self.send_error(404, f"Game interface not found. Tried: {possible_paths}")
            except Exception as e:
                print(f"Error serving game interface: {str(e)}")
                self.send_error(500, f"Error serving game interface: {str(e)}")
            return
        
        # Default behavior for other files
        super().do_GET()
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_http_server(host="0.0.0.0", port=8080):
    """Start the HTTP server"""
    server = HTTPServer((host, port), CustomHTTPRequestHandler)
    print(f"HTTP server running on http://{host}:{port}")
    print(f"Game interface available at: http://{host}:{port}/game_interface.html")
    server.serve_forever()

def main():
    """Main function to start both HTTP and WebSocket servers"""
    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Start WebSocket server
    game_server = GameWebServer()
    
    # Start WebSocket server in a separate thread
    def run_websocket_server():
        asyncio.run(game_server.start_websocket_server())
    
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()
    
    print("NeatRL Web Server started!")
    print("HTTP Server: http://localhost:8080")
    print("WebSocket Server: ws://localhost:8081")
    print("Game Interface: http://localhost:8080/game_interface.html")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        game_server.running = False

if __name__ == "__main__":
    main()
