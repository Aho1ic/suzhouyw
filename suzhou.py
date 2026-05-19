"""视频流处理器：YOLO 跟踪 + 分割联合判定 + 事件落盘。
设计要点：
- 精准 monkey-patch 仅劫持 DetectionPredictor.get_obj_feats（避免影响其他方法）
- 业务配置通过 task_info 注入或从 app_config 兜底（不再硬编码）
- SIGTERM/SIGINT 优雅退出：信号只置 should_stop，主循环退出后统一写 summary
- detected_targets 增量持久化为快照，父进程异常退出时可恢复（不再伪造数据）
- id_counter / lost_counter 分离，跟丢不会衰减累计帧
"""
from __future__ import annotations

import logging
import os
import signal
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

import app_config
from summary import build_summary_json, is_overlap
from task_store import update_task, write_json_atomic

logger = logging.getLogger(__name__)


def _configure_subprocess_logging():
    """子进程日志配置：确保 suzhou 模块的 logger 在子进程中能输出到控制台和文件。
    使用标记避免重复添加 handler。"""
    if getattr(_configure_subprocess_logging, '_done', False):
        return
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        )
    # 确保模块级 logger 也能输出
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    _configure_subprocess_logging._done = True


def _patch_yolo_obj_feats() -> None:
    """ultralytics 某些版本 get_obj_feats 用 float idx 索引 tensor，强制转 long。
    只针对这一个方法做最小修补。"""
    try:
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
    except ImportError:
        logger.warning("ultralytics 未安装，跳过 monkey patch")
        return

    if not hasattr(DetectionPredictor, "get_obj_feats"):
        return

    original = DetectionPredictor.get_obj_feats

    def patched(self, obj_feats, idxs):
        try:
            fixed = []
            for idx in idxs:
                if idx is not None and torch.is_tensor(idx):
                    idx = idx.to(torch.long)
                fixed.append(idx)
            return original(self, obj_feats, fixed)
        except Exception as e:
            logger.warning(f"get_obj_feats 补丁兜底执行原方法: {e}")
            return original(self, obj_feats, idxs)

    DetectionPredictor.get_obj_feats = patched
    logger.info("已对 DetectionPredictor.get_obj_feats 应用 long 索引补丁")


_patch_yolo_obj_feats()

from ultralytics import YOLO  # noqa: E402  延迟导入，确保补丁先生效

# 模块导入时自动配置日志，确保子进程中日志能正常输出
_configure_subprocess_logging()

ROOT = Path(__file__).resolve().parent


class VideoStreamProcessor:
    def __init__(
        self,
        device_sn=None,
        task_id=None,
        task_name=None,
        task_info: dict | None = None,
    ):
        self.task_info = dict(task_info) if task_info else {}
        self._init_paths(device_sn)
        self._init_models()
        self._init_tracking_state()
        self._init_business_config(task_id, task_name)
        self._install_signal_handlers()

    def _init_paths(self, device_sn=None):
        if device_sn:
            if device_sn.startswith(("rtmp://", "rtsp://", "http://")):
                self.stream_url = device_sn
            else:
                self.stream_url = f"rtmp://127.0.0.1:1935/ly/{device_sn}"
        else:
            self.stream_url = app_config.DEFAULT_RTMP_URL

        self.output_dir = app_config.RESULT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        app_config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = app_config.LOG_DIR / "suzhou_processor.log"

        self._verify_writable(self.output_dir)

        self.detected_targets: list = []
        self.event_id = str(uuid.uuid4())
        logger.info(f"初始化事件ID: {self.event_id}")

    def _verify_writable(self, dir_path: Path) -> None:
        test_file = dir_path / ".write_test"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            logger.error(f"输出目录写入测试失败 {dir_path}: {e}")

    def _init_models(self):
        try:
            self.device = "cuda" if self._check_cuda() else "cpu"
            logger.info(f"使用设备: {self.device}")

            det_model_path = app_config.DET_MODEL_PATH
            seg_model_path = app_config.SEG_MODEL_PATH
            tracker_config_path = app_config.TRACKER_CONFIG_PATH

            for p in (det_model_path, seg_model_path, tracker_config_path):
                if not p.exists():
                    raise FileNotFoundError(f"必需文件缺失: {p}")

            logger.info(
                f"加载检测模型: {det_model_path} "
                f"({os.path.getsize(det_model_path) / (1024 * 1024):.2f} MB)"
            )
            self.det_model = YOLO(str(det_model_path)).to(self.device)
            logger.info(
                f"检测模型类别: {self.det_model.names}, 任务: {self.det_model.task}"
            )

            logger.info(f"加载分割模型: {seg_model_path}")
            self.seg_model = YOLO(str(seg_model_path)).to(self.device)
            self.tracker_config = str(tracker_config_path)

            try:
                test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
                self.det_model(test_img, verbose=False)
                logger.info("检测模型预热成功")
            except Exception as e:
                logger.warning(f"模型预热失败: {e}")

            logger.info("模型加载完成")
        except Exception:
            logger.exception("模型加载失败")
            raise

    def _init_tracking_state(self):
        self.seen_ids: set = set()
        self.saved_ids: set = set()
        self.id_counter: dict = {}
        self.lost_counter: dict = {}
        self.overlap_flags: dict = {}
        self.MIN_FRAMES = app_config.MIN_TRACK_FRAMES
        self.MAX_LOST_FRAMES = app_config.MAX_LOST_FRAMES
        self.should_stop = False
        self.current_frame_index = 0
        self.save_counter = 1

    def _init_business_config(self, task_id=None, task_name=None):
        info = self.task_info
        self.device_name = info.get("device_name", app_config.DEVICE_NAME)
        self.device_id = info.get("device_id", app_config.DEVICE_ID)
        self.task_name = task_name or info.get("task_name") or app_config.DEFAULT_TASK_NAME
        self.task_id = str(task_id or info.get("task_id") or app_config.DEFAULT_TASK_ID)
        self.app_name = info.get("app_name", app_config.APP_NAME)
        self.app_id = info.get("app_id", app_config.APP_ID)
        self.src_name = info.get("src_name", app_config.SRC_NAME)
        self.src_id = info.get("src_id", app_config.SRC_ID)
        self.box_id = str(info.get("boxId") or app_config.DEFAULT_BOX_ID)

    def _install_signal_handlers(self):
        """收到 SIGTERM/SIGINT 只置 should_stop，避免在信号上下文里做 IO。
        只在主线程中注册，非主线程会静默跳过。"""
        import threading
        if threading.current_thread() is not threading.main_thread():
            return

        def _handle(signum, _frame):
            logger.info(f"收到信号 {signum}，准备退出")
            self.should_stop = True

        try:
            signal.signal(signal.SIGTERM, _handle)
            signal.signal(signal.SIGINT, _handle)
        except Exception as e:
            logger.warning(f"注册信号处理失败: {e}")

    def _check_cuda(self):
        try:
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
            if cuda_available:
                logger.info(
                    f"CUDA 可用: {torch.cuda.device_count()} 设备, "
                    f"主设备 {torch.cuda.get_device_name(0)}"
                )
                try:
                    t = torch.zeros((100, 100)).cuda()
                    del t
                except Exception as e:
                    logger.warning(f"CUDA 内存测试失败: {e}")
                    cuda_available = False
            return cuda_available
        except Exception as e:
            logger.warning(f"CUDA 检查异常: {e}")
            return False

    def _connect_stream(self):
        max_retries = 3
        for i in range(max_retries):
            logger.info(f"尝试连接视频流 ({i + 1}/{max_retries}): {self.stream_url}")
            cap = cv2.VideoCapture(self.stream_url)
            if cap.isOpened():
                logger.info("视频流连接成功")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            time.sleep(1)
        raise ConnectionError(f"无法连接到视频流: {self.stream_url}")

    def _process_frame(self, frame):
        try:
            try:
                det_results = self.det_model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=app_config.DET_CONF,
                    iou=app_config.DET_IOU,
                    device=self.device,
                    verbose=False,
                )
            except RuntimeError as track_err:
                if "tensors used as indices must be long" in str(track_err):
                    logger.error(f"跟踪器索引类型错误，回退到 predict: {track_err}")
                    det_results = self.det_model.predict(
                        frame,
                        conf=app_config.DET_CONF,
                        iou=app_config.DET_IOU,
                        device=self.device,
                        verbose=False,
                    )
                else:
                    raise

            seg_results = self.seg_model.predict(
                frame,
                conf=app_config.SEG_CONF,
                iou=app_config.SEG_IOU,
                device=self.device,
                verbose=False,
            )

            has_box = (
                det_results
                and det_results[0].boxes is not None
                and len(det_results[0].boxes) > 0
            )
            plotted_frame = det_results[0].plot() if has_box else frame
            return self._analyze_results(det_results, seg_results, plotted_frame)
        except RuntimeError as e:
            if "tensors used as indices must be long" in str(e):
                logger.error(f"张量索引类型错误，跳过当前帧: {e}")
                return len(self.seen_ids)
            raise
        except cv2.error as e:
            logger.error(f"OpenCV 错误: {e}")
            return len(self.seen_ids)
        except (RuntimeError, cv2.error, OSError):
            logger.exception("处理帧时未知错误")
            return len(self.seen_ids)

    def _analyze_results(self, det_results, seg_results, plotted_frame):
        try:
            blue_boxes = []
            if seg_results[0].boxes is not None and seg_results[0].boxes.xyxy is not None:
                blue_boxes = seg_results[0].boxes.xyxy.cpu().numpy()

            current_ids: set = set()

            if det_results[0].boxes is None or det_results[0].boxes.id is None:
                self._cleanup_disappeared(current_ids)
                return len(self.seen_ids)

            try:
                boxes = det_results[0].boxes.xyxy.cpu().numpy()
                ids_tensor = det_results[0].boxes.id.cpu().to(torch.long)
                ids = ids_tensor.numpy().astype(int)
                confs = det_results[0].boxes.conf.cpu().numpy()
            except Exception as e:
                logger.error(f"读取检测结果失败: {e}")
                return len(self.seen_ids)

            min_len = min(len(boxes), len(ids), len(confs))
            if min_len < max(len(boxes), len(ids), len(confs)):
                logger.warning(
                    f"数据长度不一致 boxes={len(boxes)} ids={len(ids)} confs={len(confs)}"
                )
            boxes, ids, confs = boxes[:min_len], ids[:min_len], confs[:min_len]

            for tid, box, conf in zip(ids, boxes, confs):
                current_ids.add(int(tid))
                self._update_tracking_state(int(tid), box, blue_boxes, plotted_frame, conf)

            self._cleanup_disappeared(current_ids)
            return len(self.seen_ids)
        except Exception:
            logger.exception("分析结果时发生错误")
            return len(self.seen_ids)

    def _update_tracking_state(self, tid, box, blue_boxes, plotted_frame, conf):
        self.id_counter[tid] = self.id_counter.get(tid, 0) + 1
        self.lost_counter[tid] = 0

        overlaps = (
            any(is_overlap(box, b) for b in blue_boxes) if len(blue_boxes) > 0 else False
        )
        if tid not in self.overlap_flags:
            self.overlap_flags[tid] = overlaps
        elif overlaps and not self.overlap_flags[tid]:
            self.overlap_flags[tid] = True

        if self.id_counter[tid] >= self.MIN_FRAMES and self.overlap_flags[tid]:
            if tid not in self.seen_ids:
                self.seen_ids.add(tid)
                logger.info(f"ID {tid} 满足条件，加入 seen_ids")
            if tid not in self.saved_ids:
                self._save_detection_image(tid, box, plotted_frame, conf)
                self.saved_ids.add(tid)

    def _save_detection_image(self, tid, box, plotted_frame, conf):
        if tid is None:
            return
        if isinstance(tid, float) and (np.isnan(tid) or np.isinf(tid)):
            return
        try:
            int(tid)
        except (TypeError, ValueError):
            logger.warning(f"tid 无法转 int: {tid}")
            return

        x1, y1, x2, y2 = map(int, box)
        filename = self.output_dir / f"{self.event_id}_{self.save_counter}.jpeg"
        self.save_counter += 1
        cv2.imwrite(str(filename), plotted_frame)
        pic_num = len(self.detected_targets) + 1
        self.detected_targets.append({
            "image_path": str(filename),
            "box": [x1, y1, x2, y2],
            "tid": pic_num,
            "conf": float(conf),
            "frame_id": self.current_frame_index,
        })
        self._persist_targets_snapshot()
        logger.info(
            f"保存图片: {filename.name}, 已检测目标 {len(self.detected_targets)}"
        )

    def _persist_targets_snapshot(self):
        """每次保存图片后把 detected_targets 写到磁盘快照，
        父进程在子进程异常退出时可读快照恢复真实数据。"""
        try:
            snap_path = self.output_dir / f"{self.event_id}_targets.json"
            write_json_atomic(snap_path, {
                "event_id": self.event_id,
                "frame_id": self.current_frame_index,
                "task_info": {
                    "device_name": self.device_name,
                    "device_id": self.device_id,
                    "task_name": self.task_name,
                    "task_id": self.task_id,
                    "app_name": self.app_name,
                    "app_id": self.app_id,
                    "src_name": self.src_name,
                    "src_id": self.src_id,
                    "boxId": self.box_id,
                },
                "targets": self.detected_targets,
            })
        except Exception as e:
            logger.warning(f"快照写入失败: {e}")

    def _cleanup_disappeared(self, current_ids):
        """跟丢超过 MAX_LOST_FRAMES 帧才清理。
        id_counter 表示累计出现帧，不能因短暂跟丢而衰减。"""
        for tid in list(self.id_counter.keys()):
            if tid in current_ids:
                continue
            self.lost_counter[tid] = self.lost_counter.get(tid, 0) + 1
            if self.lost_counter[tid] >= self.MAX_LOST_FRAMES:
                self.id_counter.pop(tid, None)
                self.lost_counter.pop(tid, None)
                self.overlap_flags.pop(tid, None)

    def stop(self):
        self.should_stop = True

    def generate_summary_json(self):
        if not self.detected_targets:
            logger.warning("detected_targets 为空，跳过 summary 生成")
            return None

        try:
            json_path = self.output_dir / f"{self.event_id}.json"
            json_data = build_summary_json(
                event_id=self.event_id,
                task_info={
                    "device_name": self.device_name,
                    "device_id": self.device_id,
                    "task_name": self.task_name,
                    "task_id": self.task_id,
                    "app_name": self.app_name,
                    "app_id": self.app_id,
                    "src_name": self.src_name,
                    "src_id": self.src_id,
                    "boxId": self.box_id,
                },
                detected_targets=self.detected_targets,
                frame_id=self.current_frame_index,
            )
            write_json_atomic(json_path, json_data)
            logger.info(f"已生成 summary: {json_path.name}")

            self._link_event_to_task()

            return {
                "json_path": str(json_path),
                "image_paths": [t["image_path"] for t in self.detected_targets],
            }
        except Exception:
            logger.exception("生成 summary 异常")
            return None

    def _link_event_to_task(self):
        """通过 task_store 加锁写 event_id 回 task json。"""
        candidates = [
            app_config.TASK_DIR / f"{self.box_id}_{self.task_id}.json",
            app_config.TASK_DIR / f"{self.device_id}_{self.task_id}.json",
            app_config.TASK_DIR / f"{app_config.DEFAULT_BOX_ID}_{self.task_id}.json",
        ]
        for path in candidates:
            if path.exists():
                update_task(path, lambda d: {**d, "event_id": self.event_id})
                logger.info(f"event_id 已写入 {path.name}")
                return
        logger.warning("未找到匹配的任务文件，event_id 未写入")

    def process(self, start_frame=0, end_frame=None):
        self._reset_state()
        try:
            while not self.should_stop:
                cap = None
                frame_index = 0
                consecutive_errors = 0
                max_consecutive_errors = 5
                try:
                    cap = self._connect_stream()
                    while cap.isOpened() and not self.should_stop:
                        self.current_frame_index = frame_index
                        try:
                            success, frame = self._read_frame(cap)
                            if not success:
                                logger.warning("视频流中断，准备重连")
                                break
                            if frame_index < start_frame:
                                frame_index += 1
                                continue
                            self._process_frame(frame)
                            consecutive_errors = 0
                            if frame_index % 30 == 0:
                                logger.info(
                                    f"进度: 已处理 {frame_index} 帧 | "
                                    f"有效ID {len(self.seen_ids)} | "
                                    f"已保存 {len(self.saved_ids)}"
                                )
                            frame_index += 1
                            if end_frame and frame_index >= end_frame:
                                break
                        except Exception:
                            consecutive_errors += 1
                            logger.exception(f"处理帧 {frame_index} 失败")
                            if consecutive_errors >= max_consecutive_errors:
                                break
                            frame_index += 1
                            time.sleep(0.1)
                except (ConnectionError, cv2.error, OSError) as e:
                    logger.error(f"处理外层异常: {e}")
                    for _ in range(4):
                        if self.should_stop:
                            break
                        time.sleep(0.5)
                finally:
                    if cap and cap.isOpened():
                        cap.release()
                    if not self.should_stop:
                        for _ in range(6):
                            if self.should_stop:
                                break
                            time.sleep(0.5)
        finally:
            logger.info(f"处理结束，总计有效ID {len(self.seen_ids)}")
            if len(self.detected_targets) > 0:
                self.generate_summary_json()

        return self.seen_ids.copy()

    def _read_frame(self, cap):
        success, frame = False, None
        try:
            start = time.time()
            while time.time() - start < 2.0:
                success, frame = cap.read()
                if success or self.should_stop:
                    break
                time.sleep(0.1)
        except (cv2.error, OSError) as e:
            logger.error(f"帧读取错误: {e}")
            success = False

        if success:
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                logger.warning("帧格式无效")
                success = False
        return success, frame

    def _reset_state(self):
        self.should_stop = False
        self.seen_ids.clear()
        self.saved_ids.clear()
        self.id_counter.clear()
        self.lost_counter.clear()
        self.overlap_flags.clear()
        self.current_frame_index = 0
        self.save_counter = 1
        self.detected_targets = []


if __name__ == "__main__":
    # _configure_subprocess_logging() 已在模块导入时自动执行
    processor = VideoStreamProcessor()
    try:
        logger.info("启动本地测试")
        detected_ids = processor.process(start_frame=0)
        logger.info(f"检测到ID列表: {detected_ids}")
    except KeyboardInterrupt:
        processor.stop()
        logger.info("手动终止检测流程")
