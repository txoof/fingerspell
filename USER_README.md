# Fingerspell - NGT Fingerspelling Recognition

A real-time Dutch Sign Language (NGT) fingerspelling recognition application.

## System Requirements

- macOS 10.14 or later
- Built-in or external webcam
- Approximately 500 MB of free disk space

## Installation

1. Unzip the `Fingerspell-mac.zip` file
2. Move `Fingerspell.app` to your Applications folder (optional)

## Launching the Application

### First Time Launch

Because this application is not signed by Apple, macOS Gatekeeper will block it. You need to remove the quarantine flag:

**Method 1: Using Terminal (Recommended)**

1. Open **Terminal** (found in Applications > Utilities)
2. Type the following command and press Enter:
   ```
   xattr -cr 
   ```
3. Drag `Fingerspell.app` from Finder into the Terminal window (this adds the path)
4. Press Enter
5. Now double-click `Fingerspell.app` to launch

**Example of complete command:**
```
xattr -cr /Users/YourName/Downloads/Fingerspell.app
```

**Method 2: Right-click Method (May Not Work on Newer macOS)**

1. **Right-click** (or Control-click) on `Fingerspell.app`
2. Select **"Open"** from the menu
3. If a dialog appears with an "Open" button, click it
4. If you only see "Move to Trash", use Method 1 instead

### After First Launch

Once you've successfully opened it using one of the methods above, you can double-click to launch it normally.

## Camera Permission

The first time you launch the app, macOS will ask for camera permission:

1. A dialog will appear: "Fingerspell would like to access the camera"
2. Click **"OK"** to allow access
3. The camera feed will start

If you accidentally denied permission:
1. Open **System Preferences** > **Security & Privacy** > **Privacy** > **Camera**
2. Check the box next to **Fingerspell**
3. Restart the application

## Using the Application

### Basic Controls

- **ESC** - Quit the application
- **Tab** - Toggle debug overlay (shows confidence scores and motion detection)

### Debug Mode Controls

When debug overlay is visible:

**Motion Threshold Adjustment:**
- **k** - Increase threshold by 0.01
- **j** - Decrease threshold by 0.01
- **K** (Shift+k) - Increase threshold by 0.05
- **J** (Shift+j) - Decrease threshold by 0.05

**Confidence Threshold Adjustment:**
- **w** - Increase low confidence threshold by 5%
- **s** - Decrease low confidence threshold by 5%
- **W** (Shift+w) - Increase high confidence threshold by 5%
- **S** (Shift+s) - Decrease high confidence threshold by 5%

### Sign Recognition

The application recognizes all 26 letters of the NGT fingerspelling alphabet:

**Static letters** (hold hand position):
A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, V, W, Y

**Dynamic letters** (require movement):
H, J, U, X, Z

## Troubleshooting

### "App is damaged and can't be opened" or Gatekeeper blocks the app

This is normal for unsigned applications. Use the terminal command:

1. Open **Terminal** (Applications > Utilities > Terminal)
2. Type: `xattr -cr ` (note the space at the end)
3. Drag `Fingerspell.app` into the Terminal window
4. Press Enter
5. Launch the app by double-clicking

**Complete command example:**
```bash
xattr -cr /Users/YourName/Downloads/Fingerspell.app
```

This removes the quarantine attribute that macOS adds to downloaded files.

### "Operation not permitted" or camera not working

1. Go to **System Preferences** > **Security & Privacy** > **Privacy** > **Camera**
2. Make sure **Fingerspell** is checked
3. Restart the application

### App doesn't recognize signs

1. Ensure good lighting
2. Position your hand clearly in front of the camera
3. Keep your hand within the camera frame
4. For dynamic letters (H, J, U, X, Z), make the required motion
5. Press **Tab** to see the debug overlay and check detection confidence

### App launches but window is blank

1. Quit the application (ESC or Command+Q)
2. Check that your camera isn't being used by another application
3. Try launching again

### Performance issues or lag

1. Close other applications using the camera
2. Ensure your Mac meets the system requirements
3. Try reducing the number of background applications

## Privacy

This application:
- Processes all video locally on your computer
- Does not send any data to external servers
- Does not store or record video
- Only accesses the camera while the application is running

## Support

For technical issues or questions, please contact your system administrator or the application provider.

## Credits

This application uses:
- MediaPipe for hand landmark detection
- OpenCV for video processing
- scikit-learn for sign classification

NGT reference video by Vera de Kok, released under CC BY-SA 4.0.
