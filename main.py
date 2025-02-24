from flask import Flask, Response, render_template, request, jsonify
import subprocess
import time
from gpiozero import AngularServo
from time import sleep

app = Flask(__name__)

# Create AngularServo objects
servo1 = AngularServo(13, min_angle=0, max_angle=180, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
servo2 = AngularServo(12, min_angle=0, max_angle=180, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def set_servo_angle(servo_num, angle):
    """Set the angle of the specified servo"""
    if servo_num == 1:
        servo1.angle = angle
    else:
        servo2.angle = angle
    sleep(0.5)  # Give servo time to move

def gen_frames():
    """libcamera-vid kullanarak MJPEG formatında canlı video akışı sağlar."""
    cmd = [
        "libcamera-vid",
        "-t", "0",             # Süresiz çalışır
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg",
        "--inline",            # JPEG frame'leri inline gönderir
        "--listen",            # TCP/IP üzerinden stream yapar
        "--output", "-"        # stdout'a yönlendirir
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=-1
        )
        
        jpeg_start = b'\xff\xd8'
        jpeg_end = b'\xff\xd9'
        buffer = b''
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
                
            buffer += chunk
            
            # Buffer içinde tam JPEG frame'leri ayıkla
            while True:
                start_idx = buffer.find(jpeg_start)
                end_idx = buffer.find(jpeg_end, start_idx)
                if start_idx != -1 and end_idx != -1:
                    frame = buffer[start_idx:end_idx + 2]
                    buffer = buffer[end_idx + 2:]
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    break
            
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Streaming error: {e}")
        if process.stderr:
            error = process.stderr.read()
            print(f"libcamera error: {error.decode()}")
    finally:
        if process:
            process.terminate()
            process.wait()

@app.route('/video_feed')
def video_feed():
    """Video akışını sağlayan endpoint."""
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/')
def index():
    """Ana sayfa template'i."""
    return render_template('index.html')

@app.route('/set_servo', methods=['POST'])
def set_servo():
    """
    Servo motorları kontrol etmek için POST isteği bekler.
    Gönderilecek JSON örneği: {"servo": 1 veya 2, "angle": 0-180}
    """
    data = request.get_json()
    servo = data.get('servo')
    angle = data.get('angle')
    
    if servo not in [1, 2]:
        return jsonify({'error': 'Geçersiz servo numarası. 1 veya 2 olmalıdır.'}), 400
    if not (0 <= angle <= 180):
        return jsonify({'error': 'Açı değeri 0 ile 180 arasında olmalıdır.'}), 400

    set_servo_angle(servo, angle)
    
    return jsonify({'servo': servo, 'angle': angle, 'status': 'başarılı'})

if __name__ == '__main__':
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        # Clean up
        servo1.detach()
        servo2.detach()
        print("Program terminated")
