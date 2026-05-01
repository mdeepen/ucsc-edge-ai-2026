# WSL2 Setup Guide

## 1. Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install
```

Restart your machine. Ubuntu is installed by default. Set a username and password when prompted.

Verify:
```powershell
wsl --list --verbose
```

---

## 2. Python Environment (conda)

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Restart shell, then create environment
conda create -n edgeai python=3.10 -y
conda activate edgeai

# Install common packages
pip install numpy matplotlib jupyter
```

---

## 3. VS Code + WSL Integration

1. Install [VS Code](https://code.visualstudio.com) on Windows
2. Install the **WSL** extension in VS Code
3. Open a WSL terminal and run:

```bash
code .
```

VS Code opens with a remote WSL connection. Install the **Python** and **Jupyter** extensions inside WSL when prompted.

---

## 4. Git Config

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# Set up SSH key for GitHub
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub   # paste this into GitHub → Settings → SSH Keys
```

Verify connection:
```bash
ssh -T git@github.com
```

---

## 5. OpenMV IDE + AE3 Setup

### Install OpenMV IDE

Download from [openmv.io/pages/download](https://openmv.io/pages/download).

**Windows:** Run the installer directly.

**WSL2 (Ubuntu):**
```bash
# Download the Linux AppImage
wget https://github.com/openmv/openmv-ide/releases/latest/download/openmv-ide-linux-x86_64.AppImage
chmod +x openmv-ide-linux-x86_64.AppImage

# Install FUSE (required to run AppImages)
sudo apt install libfuse2 -y

./openmv-ide-linux-x86_64.AppImage
```

> For GUI apps in WSL2, ensure you have an X server running (e.g. WSLg on Windows 11, or VcXsrv on Windows 10).

### Connect OpenMV AE3

1. Plug the AE3 into your PC via USB
2. Open OpenMV IDE
3. Click the **Connect** button (bottom-left)
4. IDE detects the board automatically and connects via serial

If the board is not detected:
- Press the reset button on the AE3 and reconnect
- Try a different USB cable (data cable, not charge-only)
- Check Device Manager (Windows) → Ports — the board should appear as a COM port

### Run a Script

1. Open or write a script in the IDE editor
2. Click **Run** (green play button) to execute on the board
3. Use the **Serial Terminal** panel to see `print()` output
4. Use the **Frame Buffer** panel to preview camera output live

### USB Passthrough (if using WSL2)

To access the AE3 from inside WSL2, use `usbipd`:

```powershell
# Run in PowerShell as Administrator
winget install usbipd
usbipd list                      # find the AE3 bus ID
usbipd bind --busid <ID>
usbipd attach --wsl --busid <ID>
```

Then inside WSL2:
```bash
ls /dev/ttyACM*   # AE3 should appear here
```
