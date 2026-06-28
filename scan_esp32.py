"""扫描局域网查找 ESP32-CAM"""
import socket, threading, time

def scan(ip_prefix, port=80):
    """扫描指定网段"""
    found = []
    lock = threading.Lock()
    
    def check(ip, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            result = s.connect_ex((ip, port))
            if result == 0:
                try:
                    s.send(b"GET / HTTP/1.0\r\nHost: " + ip.encode() + b"\r\n\r\n")
                    data = s.recv(256)
                    if b"ESP32" in data or b"esp32" in data or b"camera" in data or b"200 OK" in data:
                        with lock:
                            found.append(ip)
                    else:
                        # Also try /status endpoint
                        pass
                except:
                    pass
            s.close()
        except:
            pass
    
    threads = []
    for i in range(1, 255):
        ip = "%s.%d" % (ip_prefix, i)
        t = threading.Thread(target=check, args=(ip, port))
        t.daemon = True
        threads.append(t)
        t.start()
        if len(threads) >= 50:
            for t in threads:
                t.join(timeout=0.5)
            threads = []
    
    for t in threads:
        t.join(timeout=1)
    
    return found

# 获取本机 IP 前缀
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
except:
    local_ip = "192.168.1.1"
s.close()

prefix = ".".join(local_ip.split(".")[:3])
print("本机 IP:", local_ip)
print("扫描网段: %s.1-255 (端口80)..." % prefix)
print("请稍候...")

found = scan(prefix)
print("\n找到以下设备:")
for ip in found:
    print("  http://%s/" % ip)
    print("  http://%s/stream" % ip)

if not found:
    print("  未找到设备")
    print("\n也可能 ESP32 在其他网段，请检查串口监视器输出。")
