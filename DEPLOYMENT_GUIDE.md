# 🚀 NeatRL Web Interface Deployment Guide

## 🐳 **Docker Deployment (Recommended)**

### **Complete Setup - No Changes to Existing Leaderboard**

The web interface is **fully integrated** into the existing Docker Compose setup without affecting the current leaderboard functionality.

### **1. Start Everything with Docker Compose**

```bash
# Start the complete NeatRL system
docker-compose up -d

# This will start:
# - Main API server (port 8000)
# - Gradio interface (port 7860) 
# - Arena system with web interface (ports 8080, 8081)
# - VNC server (port 5901)
# - All other services
```

### **2. Access Points**

After starting Docker Compose:

- **🎯 Main Gradio Interface:** http://localhost:7860
  - All existing functionality preserved
  - New "🌐 Web Game Interface" tab in RL Arena section
  
- **🎮 Web Game Interface:** http://localhost:8080/game_interface.html
  - Direct access to the beautiful game interface
  - Real-time keyboard controls
  - Live game statistics

- **🔌 WebSocket Server:** ws://localhost:8081
  - Real-time communication for game streaming

### **3. User Flow**

1. **Go to Gradio:** http://localhost:7860
2. **Navigate to:** RL Arena → 🌐 Web Game Interface
3. **Select Game:** Choose from 24 PettingZoo environments
4. **Launch Game:** Click "🚀 Launch Web Game"
5. **Play:** Game opens in new tab with keyboard controls

---

## 🌐 **Standalone Web Server Deployment**

### **Option 1: Simple Python Server**

```bash
# Install dependencies
pip install websockets opencv-python pillow

# Start web server
python web_server.py

# Access at: http://localhost:8080/game_interface.html
```

### **Option 2: Nginx Deployment**

```bash
# Copy files to web directory
sudo cp frontend/game_interface.html /var/www/html/
sudo cp web_server.py /var/www/html/

# Configure Nginx
sudo nano /etc/nginx/sites-available/neatrl-game

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /var/www/html;
        index game_interface.html;
        try_files $uri $uri/ =404;
    }
    
    location /ws {
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/neatrl-game /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### **Option 3: Cloud Deployment (Vercel/Netlify)**

```bash
# Create deployment package
mkdir neatrl-web-deploy
cp frontend/game_interface.html neatrl-web-deploy/
cp web_server.py neatrl-web-deploy/

# Add vercel.json for Vercel
echo '{
  "version": 2,
  "builds": [
    {
      "src": "web_server.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/",
      "dest": "/game_interface.html"
    },
    {
      "src": "/ws",
      "dest": "/web_server.py"
    }
  ]
}' > neatrl-web-deploy/vercel.json

# Deploy to Vercel
cd neatrl-web-deploy
vercel --prod
```

---

## 🔧 **Configuration Options**

### **Environment Variables**

```bash
# Web server configuration
export WEB_HOST=0.0.0.0
export WEB_PORT=8080
export WS_PORT=8081

# Game server configuration  
export GAME_SERVER_URL=http://localhost:8000
export WEBSOCKET_URL=ws://localhost:8081
```

### **Custom Ports**

```bash
# Modify docker-compose.yml ports if needed
ports:
  - "8080:8080"  # Web interface
  - "8081:8081"  # WebSocket
  - "7860:7860"  # Gradio (existing)
  - "8000:8000"  # API (existing)
```

---

## 🎯 **Integration with Existing System**

### **No Breaking Changes**

✅ **Existing Gradio Interface:** Fully preserved
✅ **Leaderboard Functionality:** Unchanged  
✅ **API Endpoints:** All working
✅ **Database:** No modifications
✅ **User Submissions:** Continue as normal

### **New Features Added**

✅ **Web Game Interface:** New tab in RL Arena
✅ **Real-time Gaming:** Keyboard-controlled games
✅ **Beautiful UI:** Modern, responsive design
✅ **Cross-platform:** Works on all devices

---

## 🚀 **Quick Start Commands**

### **Development**
```bash
# Start everything
docker-compose up

# Access interfaces
open http://localhost:7860  # Gradio
open http://localhost:8080/game_interface.html  # Web Game
```

### **Production**
```bash
# Start in background
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f arena
```

---

## ✅ **Verification Checklist**

- [ ] Docker Compose starts without errors
- [ ] Gradio interface accessible at port 7860
- [ ] Web game interface accessible at port 8080
- [ ] WebSocket server running on port 8081
- [ ] Keyboard controls working in web interface
- [ ] Game statistics updating in real-time
- [ ] All existing functionality preserved

---

## 🎉 **Ready to Deploy!**

The web interface is **fully functional** and **ready for deployment**. It integrates seamlessly with the existing NeatRL system without any breaking changes.

**Start with:** `docker-compose up -d`
