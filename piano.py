import cv2
import numpy as np
import mediapipe as mp
import imutils
import simpleaudio as sa
import os
import time

# ——— Load sound files ———
SAMPLES_DIR = r"D:\piano_wav"
sound_map = {}
#preventing crash of folder
if os.path.exists(SAMPLES_DIR):
    for fname in os.listdir(SAMPLES_DIR):
        if fname.lower().endswith('.wav'):
            note = os.path.splitext(fname)[0].lower()
            try:
                sound_map[note] = sa.WaveObject.from_wave_file(os.path.join(SAMPLES_DIR, fname))
            except:
                pass

# ——— Hand Detector ———
class HandDetector:
    def __init__(self, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=float(detectionCon),
            min_tracking_confidence=float(trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        self.img_h, self.img_w = img.shape[:2]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for lm in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, lm, mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec((0,255,0),2,3),
                    connection_drawing_spec=self.mpDraw.DrawingSpec((0,255,0),2,2))
        return img

    def handsCount(self):
        return len(self.results.multi_hand_landmarks) if self.results.multi_hand_landmarks else 0

    def getFingerData(self, handNo=0):
        fingers = []
        if not self.results.multi_hand_landmarks or len(self.results.multi_hand_landmarks) <= handNo:
            return fingers
        lm = self.results.multi_hand_landmarks[handNo]
        pip_map = {4:3, 8:6, 12:10, 16:14, 20:18}
        for tip_id, pip_id in pip_map.items():
            t, p = lm.landmark[tip_id], lm.landmark[pip_id]
            fingers.append({
                'tip_id': tip_id,
                'x': int(t.x * self.img_w), 'y': int(t.y * self.img_h),
                'pip_x': int(p.x * self.img_w), 'pip_y': int(p.y * self.img_h),
                'z': t.z
            })
        return fingers

# ——— Piano Keys ———
white_keys = [('c3','C3'),('d3','D3'),('e3','E3'),('f3','F3'),('g3','G3'),
              ('a3','A3'),('b3','B3'),('c4','C4'),('d4','D4'),('e4','E4'),
              ('f4','F4'),('g4','G4'),('a4','A4'),('b4','B4'),('c5','C5')]
black_keys = [('c#3','C#3',0.5),('d#3','D#3',1.5),('f#3','F#3',3.5),
              ('g#3','G#3',4.5),('a#3','A#3',5.5),('c#4','C#4',7.5),
              ('d#4','D#4',8.5),('f#4','F#4',10.5),('g#4','G#4',11.5),
              ('a#4','A#4',12.5)]

class PianoKeyCollision:
    def __init__(self):
        self.white_kw = 55
        self.white_kh = 300
        self.black_kw = 33
        self.black_kh = 140
        self.hover_zone = 70
        self.bend_thresh = 3  # ✅ Easier pressing

    def get_piano_dims(self, img_shape):
        total = len(white_keys) * self.white_kw
        sx = (img_shape[1] - total) // 2
        sy = img_shape[0] - self.white_kh - 80
        return sx, sy, total

    def point_in_rect(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def check_key_interaction(self, fingers, img_shape):
        hovered, pressed = set(), set()
        sx, sy, _ = self.get_piano_dims(img_shape)
        for f in fingers:
            x, y = f['x'], f['y']
            bent = (y - f['pip_y']) > self.bend_thresh
            if not (sy - self.hover_zone <= y <= sy + self.white_kh): continue
            hit = False
            for note, _, pos in black_keys:
                kx = sx + int(pos * self.white_kw) - self.black_kw//2
                hover_r = (kx, sy - self.hover_zone, kx + self.black_kw, sy + self.black_kh)
                press_r = (kx, sy, kx + self.black_kw, sy + self.black_kh)
                if self.point_in_rect(x, y, hover_r):
                    hovered.add(note)
                    if bent and self.point_in_rect(x, y, press_r): pressed.add(note)
                    hit = True; break
            if hit: continue
            for i, (note, _) in enumerate(white_keys):
                kx = sx + i * self.white_kw
                hover_r = (kx+2, sy - self.hover_zone, kx+self.white_kw-2, sy + self.white_kh)
                press_r = (kx+2, sy, kx+self.white_kw-2, sy + self.white_kh)
                if self.point_in_rect(x, y, hover_r):
                    hovered.add(note)
                    if bent and self.point_in_rect(x, y, press_r): pressed.add(note)
                    break
        return list(hovered), list(pressed)

# ——— Drawing ———
def draw_piano(img, hovered, pressed, coll):
    sx, sy, _ = coll.get_piano_dims(img.shape)
    for i, (note, disp) in enumerate(white_keys):
        x = sx + i * coll.white_kw
        clr = (255,255,255) if note not in pressed else (0,255,255)
        cv2.rectangle(img, (x,sy), (x+coll.white_kw, sy+coll.white_kh), clr, -1)
        if note in hovered:
            cv2.rectangle(img, (x,sy), (x+coll.white_kw, sy+coll.white_kh), (255,0,0), 2)
        cv2.putText(img, disp, (x+10, sy+coll.white_kh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    for note, disp, pos in black_keys:
        x = sx + int(pos * coll.white_kw) - coll.black_kw // 2
        clr = (25,25,25) if note not in pressed else (0,255,0)
        cv2.rectangle(img, (x,sy), (x+coll.black_kw, sy+coll.black_kh), clr, -1)
        if note in hovered:
            cv2.rectangle(img, (x,sy), (x+coll.black_kw, sy+coll.black_kh), (255,255,0), 2)
        cv2.putText(img, disp, (x+5, sy+coll.black_kh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return img

def draw_fingertips(img, fingers):
    for i, f in enumerate(fingers):
        cv2.circle(img, (f['x'], f['y']), 8, (0,255,0), -1)
        cv2.circle(img, (f['x'], f['y']), 12, (255,255,255), 2)
        cv2.putText(img, str(i+1), (f['x']-5, f['y']+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(img, f"{f['z']:.2f}", (f['x']-15, f['y']-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

# ——— Sound ———
def play_note(note):
    for v in (note, note.lower(), note.upper()):
        obj = sound_map.get(v)
        if obj:
            try: obj.play()
            except: pass
            break

class KeyPressManager:
    def __init__(self):
        self.prev = set()
        self.times = {}
        self.debounce = 0.05
        self.sustain_time = 0.3
        self.sustained = {}

    def process(self, curr):
        now = time.time()
        new_keys = set(curr) - self.prev
        to_play = []
        for k in new_keys:
            if now - self.times.get(k, 0) >= self.debounce:
                self.times[k] = now
                to_play.append(k)
        released = self.prev - set(curr)
        for r in released:
            self.sustained[r] = now
        self.prev = set(curr)
        self.cleanup_sustain(now)
        return to_play

    def cleanup_sustain(self, now):
        expired = [k for k, t in self.sustained.items() if now - t > self.sustain_time]
        for k in expired:
            del self.sustained[k]

    def get_sustained(self):
        return list(self.sustained.keys())

# ——— Utilities ———
def format_notes(notes):
    return ' '.join(sorted(n.upper() for n in notes)) if notes else ''

# ——— Main ———
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = HandDetector()
    coll = PianoKeyCollision()
    mgr  = KeyPressManager()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        img = imutils.resize(frame, width=1280)
        detector.findHands(img)
        fingers = []
        for h in range(detector.handsCount()):
            fingers += detector.getFingerData(h)
        hovered, pressed = coll.check_key_interaction(fingers, img.shape)
        for k in mgr.process(pressed): play_note(k)
        sustained = mgr.get_sustained()
        img = draw_piano(img, hovered, pressed, coll)
        draw_fingertips(img, fingers)

        x0, y0 = 10, 25
        cv2.putText(img, f"Hands: {detector.handsCount()}", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        def draw_status(label, notes, color, y_offset):
            if not notes: return
            cv2.putText(img, f"{label}: ", (x0, y0 + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(img, format_notes(notes), (x0 + 130, y0 + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        draw_status("Pressing", pressed, (0,255,0), 30)
        draw_status("Hovering", hovered, (255,0,0), 60)
        draw_status("Sustaining", sustained, (0,255,255), 90)

        cv2.imshow("Virtual Piano", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
