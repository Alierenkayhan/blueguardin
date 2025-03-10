from flask import Flask, Response, render_template, request, jsonify
import subprocess
import time
import threading
import torch
import cv2
import numpy as np
from gpiozero import AngularServo, Buzzer
from time import sleep

app = Flask(__name__)

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # Güven seviyesi eşiği

# Servo motorları tanımla
servo1 = AngularServo(13, min_angle=0, max_angle=180, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
servo2 = AngularServo(12, min_angle=0, max_angle=180, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Buzzer tanımla (Pin 26'ya bağlı)
buzzer = Buzzer(26)
buzzer_on = False  # Buzzer durumu

# Global değişken: en son işlenmiş JPEG frame
global_frame = None
frame_lock = threading.Lock()

def set_servo_angle(servo_num, angle):
    """Servo motor açısını ayarlar."""
    if servo_num == 1:
        servo1.angle = angle
    else:
        servo2.angle = angle
    sleep(0.5)

def detect_objects(frame):
    """YOLOv5 kullanarak nesne tespiti yapar ve frame üzerinde işaretler."""
    global buzzer_on

    results = model(frame)  # Modeli çalıştır
    detections = results.xyxy[0].cpu().numpy()  # Tespitleri al

    bus_detected = False  # Bus nesnesi tespiti

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        if conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # "person" tespitinde servo 1'i hareket ettir
            if label == "person":
                set_servo_angle(1, 90)
            # "bus" tespitinde buzzer'ı etkinleştir
            if label == "bus":
                bus_detected = True

    if bus_detected and not buzzer_on:
        buzzer.on()
        buzzer_on = True
    elif not bus_detected and buzzer_on:
        buzzer.off()
        buzzer_on = False

    return frame

def capture_frames():
    """Arka planda tek bir defa kamera alt süreci başlatıp, frame'leri sürekli günceller."""
    global global_frame
    cmd = [
        "libcamera-vid", "-t", "0",
        "--width", "640", "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg",
        "--output", "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b''

    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            continue

        buffer += chunk
        start_idx = buffer.find(b'\xff\xd8')  # JPEG başlangıcı
        end_idx = buffer.find(b'\xff\xd9', start_idx)  # JPEG bitişi
        
        if start_idx != -1 and end_idx != -1:
            frame_data = buffer[start_idx:end_idx+2]
            buffer = buffer[end_idx+2:]
            np_frame = np.frombuffer(frame_data, dtype=np.uint8)
            img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
            if img is not None:
                processed_img = detect_objects(img)
                ret, jpeg = cv2.imencode('.jpg', processed_img)
                if ret:
                    with frame_lock:
                        global_frame = jpeg.tobytes()

def generate_frames():
    """Her istemci için global frame'den sürekli veri gönderir."""
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            frame = global_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)  # Frame gönderim hızını ayarlayabilirsiniz

# Arka planda frame yakalama iş parçacığını başlatıyoruz
t = threading.Thread(target=capture_frames)
t.daemon = True
t.start()

@app.route('/video_feed')
def video_feed():
    """Video akışını sağlayan endpoint."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Ana sayfa."""
    return render_template('index.html')

@app.route('/set_servo', methods=['POST'])
def set_servo():
    """Servo motor açısını ayarlayan API endpoint'i."""
    data = request.get_json()
    servo = data.get('servo')
    angle = data.get('angle')

    if servo not in [1, 2]:
        return jsonify({'error': 'Geçersiz servo numarası.'}), 400
    if not (0 <= angle <= 180):
        return jsonify({'error': 'Açı değeri 0 ile 180 arasında olmalıdır.'}), 400

    set_servo_angle(servo, angle)
    return jsonify({'servo': servo, 'angle': angle, 'status': 'başarılı'})

@app.route('/detection_status')
def detection_status():
    """Bus tespit durumunu JSON formatında döner."""
    return jsonify({"bus_detected": buzzer_on})


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        servo1.detach()
        servo2.detach()
        buzzer.off()
        print("Program sonlandırıldı.")
