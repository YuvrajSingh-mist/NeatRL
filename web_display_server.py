#!/usr/bin/env python3
"""
Web-based display server for NeatRL games
Streams game frames to web browser for easy access
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
import base64
from io import BytesIO
from PIL import Image
import threading
import time
from typing import Optional, Dict, Any

class WebDisplayServer:
    """Web server that streams game frames to browser"""
    
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.clients = set()
        self.current_frame = None
        self.game_info = {}
        self.running = False
        
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"Web display server running on http://{self.host}:{self.port}")
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
    
    def update_frame(self, frame: np.ndarray, game_info: Dict[str, Any] = None):
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

def create_html_interface():
    """Create HTML interface for web display"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>NeatRL Game Display</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        .game-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .game-frame {
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .game-info {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            min-width: 400px;
        }
        .controls {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .key {
            display: inline-block;
            background: #444;
            padding: 8px 12px;
            margin: 4px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: bold;
        }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .status.connected { background: #2d5a2d; }
        .status.disconnected { background: #5a2d2d; }
    </style>
</head>
<body>
    <div class="game-container">
        <h1>🎮 NeatRL Game Display</h1>
        
        <div id="status" class="status disconnected">
            Connecting to game server...
        </div>
        
        <canvas id="gameCanvas" class="game-frame" width="800" height="600"></canvas>
        
        <div class="game-info">
            <h3>Game Information</h3>
            <div id="gameInfo">Waiting for game data...</div>
        </div>
        
        <div class="controls">
            <h3>Keyboard Controls</h3>
            <div>
                <span class="key">W</span> <span class="key">A</span> <span class="key">S</span> <span class="key">D</span>
                <span class="key">F</span> <span class="key">G</span> <span class="key">Q</span>
            </div>
            <p>Use these keys to control your character in the game!</p>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        const gameInfo = document.getElementById('gameInfo');
        
        let ws = null;
        let img = new Image();
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.hostname}:8080`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                status.textContent = 'Connected to game server';
                status.className = 'status connected';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'frame') {
                    // Update game frame
                    img.onload = function() {
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.src = 'data:image/jpeg;base64,' + data.frame;
                    
                    // Update game info
                    if (data.game_info) {
                        gameInfo.innerHTML = `
                            <p><strong>Environment:</strong> ${data.game_info.environment || 'Unknown'}</p>
                            <p><strong>AI Score:</strong> ${data.game_info.ai_score || 0}</p>
                            <p><strong>Your Score:</strong> ${data.game_info.human_score || 0}</p>
                            <p><strong>Current Agent:</strong> ${data.game_info.current_agent || 'Unknown'}</p>
                        `;
                    }
                }
            };
            
            ws.onclose = function() {
                status.textContent = 'Disconnected from game server';
                status.className = 'status disconnected';
                setTimeout(connect, 3000); // Reconnect after 3 seconds
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                status.textContent = 'Connection error';
                status.className = 'status disconnected';
            };
        }
        
        // Handle keyboard input
        document.addEventListener('keydown', function(event) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'keyboard',
                    key: event.key.toLowerCase()
                }));
            }
        });
        
        // Start connection
        connect();
    </script>
</body>
</html>
    """
    return html

if __name__ == "__main__":
    # Create and start the web display server
    server = WebDisplayServer()
    
    # Start the server in a separate thread
    def run_server():
        asyncio.run(server.start_server())
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("Web display server started!")
    print("Access the game at: http://localhost:8080")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down web display server...")
        server.running = False
