# 🎮 NeatRL User Guide - How to See and Play the Game

## 🖥️ **Multiple Ways to View the Game**

NeatRL provides **3 different methods** for users to see and interact with the game, depending on their setup and preferences:

---

## 🌐 **Method 1: Web Browser (Recommended - Easiest)**

### **How it works:**
- Game frames are streamed to your web browser
- No additional software needed
- Works on any device with a web browser
- Real-time keyboard input through the browser

### **Steps:**
1. **Start the system:**
   ```bash
   docker-compose up
   ```

2. **Open your web browser:**
   ```
   http://localhost:8080
   ```

3. **Start playing:**
   - The game will appear in your browser
   - Use keyboard controls (WASD, F, G, Q)
   - Real-time score updates
   - No installation required!

### **Features:**
- ✅ **Cross-platform** (Windows, Mac, Linux, Mobile)
- ✅ **No VNC client needed**
- ✅ **Real-time streaming** at 30 FPS
- ✅ **Keyboard input** through browser
- ✅ **Game information** display
- ✅ **Automatic reconnection**

---

## 🖥️ **Method 2: VNC Viewer (Traditional)**

### **How it works:**
- Connect to the VNC server running in Docker
- See the actual game window as if it's running locally
- Full desktop environment access

### **Steps:**
1. **Start the system:**
   ```bash
   docker-compose up
   ```

2. **Install a VNC viewer:**
   - **Windows:** RealVNC Viewer, TightVNC Viewer
   - **Mac:** RealVNC Viewer, Screen Sharing (built-in)
   - **Linux:** Remmina, TigerVNC Viewer

3. **Connect to VNC:**
   - **Host:** `localhost`
   - **Port:** `5901`
   - **Password:** (if configured)

4. **Play the game:**
   - You'll see the game window in the VNC session
   - Use keyboard controls directly

### **Features:**
- ✅ **Full desktop access**
- ✅ **Native performance**
- ✅ **Works with any VNC client**
- ✅ **Secure connection**

---

## 🖥️ **Method 3: X11 Forwarding (Linux/Mac)**

### **How it works:**
- Forward X11 display from Docker to your local display
- Best performance for Linux/Mac users
- Native window integration

### **Steps:**
1. **Enable X11 forwarding:**
   ```bash
   xhost +local:docker
   ```

2. **Start the system:**
   ```bash
   docker-compose up
   ```

3. **The game window appears:**
   - Game window opens directly on your desktop
   - Native performance and integration
   - No additional software needed

### **Features:**
- ✅ **Best performance**
- ✅ **Native window integration**
- ✅ **Lowest latency**
- ⚠️ **Linux/Mac only**

---

## 🎮 **Game Controls**

### **Standard Controls (All Methods):**
- **W** - Move Up
- **A** - Move Left  
- **S** - Move Down
- **D** - Move Right
- **F** - Fire/Action
- **G** - Special Action (some games)
- **Q** - Quit Game

### **Environment-Specific Controls:**
Each of the 24 PettingZoo Atari games has optimized controls:

- **Pong variants:** WASD for movement, F for fire
- **Combat games:** WASD for movement, F for attack, G for block
- **Sports games:** WASD for movement, F for action
- **Strategy games:** WASD for navigation, F for selection

---

## 🚀 **Quick Start Guide**

### **For New Users (Recommended):**

1. **Start NeatRL:**
   ```bash
   cd /path/to/neatrl
   docker-compose up
   ```

2. **Open web browser:**
   ```
   http://localhost:8080
   ```

3. **Select a game:**
   - Choose from 24 available environments
   - Pick your favorite Atari game

4. **Start playing:**
   - Game loads automatically
   - Use keyboard controls
   - Compete against AI opponents

### **For Advanced Users:**

1. **Use VNC for full control:**
   ```bash
   # Connect with VNC viewer
   vncviewer localhost:5901
   ```

2. **Or use X11 forwarding (Linux/Mac):**
   ```bash
   xhost +local:docker
   docker-compose up
   ```

---

## 🔧 **Troubleshooting**

### **Web Browser Issues:**
- **Game not loading:** Check if port 8080 is available
- **Slow performance:** Try reducing browser quality settings
- **Connection lost:** Browser will auto-reconnect

### **VNC Issues:**
- **Can't connect:** Verify port 5901 is open
- **No display:** Check VNC server is running
- **Slow performance:** Try different VNC client

### **X11 Issues:**
- **Permission denied:** Run `xhost +local:docker`
- **No display:** Ensure X11 is running
- **Performance issues:** Check graphics drivers

---

## 📊 **Performance Comparison**

| Method | Ease of Use | Performance | Setup Required | Cross-Platform |
|--------|-------------|-------------|----------------|----------------|
| **Web Browser** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **VNC Viewer** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **X11 Forwarding** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## 🎯 **Recommendations**

### **For Most Users:**
- **Use the web browser method** - easiest setup, works everywhere
- **No installation required** - just open a browser
- **Real-time performance** - good for casual gaming

### **For Power Users:**
- **Use VNC** - full control, better performance
- **Install VNC client** - more features
- **Better for competitive play**

### **For Linux/Mac Developers:**
- **Use X11 forwarding** - best performance
- **Native integration** - seamless experience
- **Lowest latency** - ideal for development

---

## 🎉 **Ready to Play!**

Choose your preferred method and start playing NeatRL's 24 multi-agent Atari games! The web browser method is recommended for most users as it provides the easiest setup and good performance.

**Happy gaming! 🎮**
