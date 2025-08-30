#!/bin/bash

# Start VNC server for display support
echo "Starting VNC server..."
vncserver :1 -geometry 1024x768 -depth 24 -localhost no &
VNC_PID=$!

# Wait for VNC to start
sleep 2

# Set display environment
export DISPLAY=:1

# Start window manager
fluxbox &
FLUXBOX_PID=$!

# Wait for window manager
sleep 1

# Start the web server in background
echo "Starting Web Server..."
python web_server.py &
WEB_SERVER_PID=$!

# Start the arena battle system
echo "Starting Arena Battle System..."
python arena_battle_system.py

# Cleanup on exit
echo "Shutting down..."
kill $VNC_PID 2>/dev/null
kill $FLUXBOX_PID 2>/dev/null
kill $WEB_SERVER_PID 2>/dev/null
vncserver -kill :1 2>/dev/null
