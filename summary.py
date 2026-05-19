"""与模型无关的纯工具：bbox 重叠判定、事件 JSON 装配、targets 状态恢复。
不依赖 torch / ultralytics / cv2，方便父进程在无模型环境下做兜底。"""
from __future__ import annotations

import logging
import time

import app_config

logger = logging.getLogger(__name__)


def is_overlap(box1, box2) -> bool:
    """xyxy 两个 box 是否相交。"""
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    return (x_max - x_min) > 0 and (y_max - y_min) > 0


def build_summary_json(
    *,
    event_id: str,
    task_info: dict,
    detected_targets: list,
    frame_id: int,
) -> dict:
    """组装最终事件 JSON，纯函数。
    task_info 缺字段时回退到 app_config 默认值。"""
    targets = []
    for i, t in enumerate(detected_targets):
        try:
            x1, y1, x2, y2 = t["box"]
            targets.append({
                "angle": 0,
                "box": {
                    "left_top_x": int(x1),
                    "left_top_y": int(y1),
                    "right_bottom_x": int(x2),
                    "right_bottom_y": int(y2),
                },
                "color": [255, 0, 0, 0],
                "cross_label": "",
                "id": int(t["tid"]),
                "label": "antenna",
                "prob": round(float(t["conf"]), 5),
                "moving": True,
                "ocr": "",
                "region_label": "",
                "roi_id": 0,
                "reserved": "",
            })
        except Exception as e:
            logger.error(f"build_summary_json: target {i} 构造失败: {e}")

    return {
        "event_id": event_id,
        "event_state": 0,
        "device_name": task_info.get("device_name", app_config.DEVICE_NAME),
        "device_id": task_info.get("device_id", app_config.DEVICE_ID),
        "task_name": task_info.get("task_name", app_config.DEFAULT_TASK_NAME),
        "task_id": str(task_info.get("task_id", app_config.DEFAULT_TASK_ID)),
        "app_name": task_info.get("app_name", app_config.APP_NAME),
        "app_id": task_info.get("app_id", app_config.APP_ID),
        "src_name": task_info.get("src_name", app_config.SRC_NAME),
        "src_id": task_info.get("src_id", app_config.SRC_ID),
        "created": int(time.time()),
        "details": [{
            "frame_id": frame_id,
            "metadata": {"max_lost_time": 3},
            "model_id": "YOLO11",
            "model_name": "antenna_v1",
            "model_thres": app_config.DET_CONF,
            "model_type": 1,
            "targets": targets,
        }],
        "picNum": len(detected_targets),
    }
