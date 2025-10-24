# 🎹 Virtual-Hand-Controlled-Piano

This project is a real-time, hand-gesture-controlled virtual piano application built using Python, OpenCV, and MediaPipe. It transforms the user's webcam feed into an interactive instrument, allowing notes to be played by bending the fingertips over the projected on-screen keys.

✨ Project Demo:<br>

https://www.linkedin.com/posts/dhananjay-jaiswal_python-opencv-mediapipe-activity-7352610206398193665-QDVB?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFFZA08BX6OKTQcoZqc35lnZX2f5IpgrC0Y

💡 How It Works:<br>

Hand Tracking: Uses the MediaPipe framework to detect and track hand landmarks (specifically the fingertip and the PIP joint).

Gesture-to-Action Mapping:

Hovering: When a fingertip is positioned vertically within the key's detection zone.

Pressing: When the vertical distance between the fingertip and the PIP joint is small (indicating a bend) and the fingertip is over the key.

Real-Time Feedback: Uses OpenCV to overlay the virtual keyboard onto the camera feed and updates the key colors on hover/press while simultaneously playing the corresponding note via the simpleaudio library.<br>

🛠️ Configuration & Setup<br>

Prerequisites<br>
Python 3.x<br>
A webcam connected to your computer.<br>

Step-by-Step Installation-><br>

Clone the Repository:<br>
git clone [Your-Repository-Link]<br>
cd Virtual_Piano_CV<br>
Install Dependencies: All required libraries can be installed using the requirements.txt file:<br>

pip install -r requirements.txt<br>
Acquire Piano Sound Samples: This project requires individual piano note sound files.<br>
Find a set of .wav audio files for the required notes (C3, D3, E3... up to C5, including all sharps).<br>
Place all of these .wav files directly into the piano_wav/ directory within the project folder.<br>

Run the Application:<br>
python virtual_piano.py<br>
The application will start, opening a window displaying your live webcam feed with the virtual piano at the bottom.<br>

🖐️ Usage and Interaction:<br>

Once the application is running:<br>
Position your hands so they are clearly visible in the camera window.<br>
The virtual piano keyboard will appear at the bottom of the screen.<br>
To play a note, move a fingertip over a key and then bend that finger. The tip must cross the specified threshold to register a press.<br>
Real-time status information (Hands Detected, Notes Pressing, Notes Hovering) is displayed in the top-left corner.<br>

:scroll: License

This project is open-source and released under the MIT License.
