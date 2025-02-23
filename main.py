from flask import Flask, Response, render_template, request, jsonify
import subprocess
import time
import lgpio  # rpi-lgpio kütüphanesi

app = Flask(__name__)

# Servo pinleri (BCM numaralandırması)
SERVO1_PIN = 12
SERVO2_PIN = 13

# LGPIO bağlantısını açıyoruz (genelde /dev/gpiochip0)
h = lgpio.gpiochip_open(0)

def set_servo_angle(servo_pin, angle):
    """
    Belirtilen açıya göre servo motorun pulse genişliğini hesaplar ve lgpio üzerinden gönderir.
    0° için ~500 µs, 180° için ~2500 µs pulse gönderilir.
    """
    # Lineer dönüşüm: 0° => 500 µs, 180° => 2500 µs
    pulsewidth = int(500 + (angle / 180.0) * 2000)
    # Doğru fonksiyon: gpioSetServoPulsewidth
    lgpio.gpioSetServoPulsewidth(h, servo_pin, pulsewidth)
    time.sleep(0.5)  # Servo hareketi için kısa bekleme

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

    if servo == 1:
        set_servo_angle(SERVO1_PIN, angle)
    else:
        set_servo_angle(SERVO2_PIN, angle)
    
    return jsonify({'servo': servo, 'angle': angle, 'status': 'başarılı'})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        # Uygulama kapanırken LGPIO bağlantısını kapatıyoruz
        lgpio.gpiochip_close(h)
