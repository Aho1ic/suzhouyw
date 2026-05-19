"""RTMP 流 + YOLO 推理 + Flask MJPEG 推流服务。
配置从 app_config 读取，敏感信息不再硬编码。"""
from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

import app_config

logger = logging.getLogger(__name__)


class YoloStreamServer:
    def __init__(
        self,
        rtmp_url: str | None = None,
        model_path: str | Path | None = None,
        port: int | None = None,
        host: str | None = None,
        conf_threshold: float | None = None,
        img_size: int = 1280,
        fps_limit: int = 15,
    ):
        self.rtmp_url = rtmp_url or app_config.DEFAULT_RTMP_URL
        self.port = port if port is not None else app_config.STREAM_PORT
        self.host = host or app_config.STREAM_HOST
        self.conf_threshold = conf_threshold if conf_threshold is not None else app_config.DET_CONF
        self.img_size = img_size
        self.fps_limit = fps_limit

        self.app = Flask(__name__)
        self._setup_routes()

        model_path = Path(model_path) if model_path else app_config.DET_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = YOLO(str(model_path))

        self.stream_thread: threading.Thread | None = None
        self.is_running = False
        self.should_stop = False
        self.latest_frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.wait_img = self._make_wait_img()

    def _make_wait_img(self) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Connecting to RTMP stream...", (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, self.rtmp_url, (50, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return img

    def _setup_routes(self):
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )

        @self.app.route('/')
        def index():
            return f"""
            <html>
              <head>
                <title>YOLO11 RTMP Stream Analyzer</title>
                <style>
                  html, body {{ height: 100%; margin: 0; padding: 0; }}
                  #stream-container {{
                    width: 100vw; height: 100vh;
                    display: flex; align-items: center; justify-content: center;
                    background: #000;
                  }}
                  #video-stream {{
                    max-width: 100vw; max-height: 100vh;
                    object-fit: contain; display: block;
                  }}
                </style>
              </head>
              <body>
                <div id="stream-container">
                  <img id="video-stream" src="/video_feed">
                </div>
                <script>
                  const img = document.getElementById('video-stream');
                  img.onerror = function() {{
                    setTimeout(function() {{
                      img.src = '/video_feed?t=' + new Date().getTime();
                    }}, 2000);
                  }};
                </script>
              </body>
            </html>
            """

    def _process_stream(self):
        cap: cv2.VideoCapture | None = None
        reconnect_delay = 5

        while not self.should_stop:
            try:
                if cap is None:
                    cap = cv2.VideoCapture()
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)

                if not cap.isOpened():
                    logger.info(f"连接 RTMP 流: {self.rtmp_url}")
                    if not cap.open(self.rtmp_url, cv2.CAP_FFMPEG):
                        logger.error(f"打开 RTMP 流失败: {self.rtmp_url}")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 60)
                        continue
                    logger.info("RTMP 流连接成功")
                    reconnect_delay = 5
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    cap.set(cv2.CAP_PROP_FPS, self.fps_limit)

                success, frame = cap.read()
                if not success:
                    logger.warning("流断开，准备重连")
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue

                try:
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        verbose=False,
                        imgsz=self.img_size,
                    )
                    annotated = results[0].plot()
                    with self.lock:
                        self.latest_frame = annotated
                except Exception as e:
                    logger.error(f"模型推理错误: {e}")
                    with self.lock:
                        self.latest_frame = frame
            except Exception as e:
                logger.error(f"流处理异常: {e}")
                if cap and cap.isOpened():
                    cap.release()
                cap = None
                time.sleep(5)

        if cap and cap.isOpened():
            cap.release()

    def _generate_frames(self):
        while not self.should_stop:
            with self.lock:
                frame_to_send = self.latest_frame if self.latest_frame is not None else self.wait_img

            ret, buffer = cv2.imencode('.jpg', frame_to_send)
            if not ret:
                logger.error("帧编码失败")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else self.wait_img.copy()

    def get_flask_app(self):
        return self.app

    def _install_signal_handlers(self):
        """注册 SIGTERM/SIGINT 优雅停止，仅在主线程中生效。"""
        if threading.current_thread() is not threading.main_thread():
            return

        def _handle(signum, _frame):
            logger.info(f"收到信号 {signum}，准备停止服务")
            self.stop()

        try:
            signal.signal(signal.SIGTERM, _handle)
            signal.signal(signal.SIGINT, _handle)
        except Exception as e:
            logger.warning(f"注册信号处理失败: {e}")

    def start(self):
        if self.is_running:
            logger.warning("服务已在运行")
            return
        self._install_signal_handlers()
        self.should_stop = False
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._process_stream, daemon=True)
        self.stream_thread.start()
        logging.getLogger('werkzeug').disabled = True
        logger.info(f"启动 Flask 服务 http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True, debug=False)

    def start_background(self):
        if self.is_running:
            logger.warning("服务已在运行")
            return
        self.should_stop = False
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._process_stream, daemon=True)
        self.stream_thread.start()
        logger.info("流处理已在后台启动")

    def stop(self):
        if not self.is_running:
            logger.warning("服务未运行")
            return
        self.should_stop = True
        self.is_running = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        logger.info("流处理已停止")

    def is_active(self):
        return self.is_running and not self.should_stop

    def update_rtmp_url(self, new_url: str):
        self.rtmp_url = new_url
        self.wait_img = self._make_wait_img()
        # 如果流线程正在运行，重启以应用新 URL
        if self.is_running:
            logger.info(f"RTMP URL 已更新，重启流处理: {new_url}")
            self.stop()
            self.should_stop = False
            self.is_running = True
            self.stream_thread = threading.Thread(target=self._process_stream, daemon=True)
            self.stream_thread.start()
        else:
            logger.info(f"RTMP URL 已更新: {new_url}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    server = YoloStreamServer()
    server.start()
