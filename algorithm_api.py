"""算法服务 HTTP API。

设计要点：
- 配置从 app_config 读取，敏感凭证强制走环境变量
- 任务文件 IO 通过 task_store 加锁原子写
- 子进程使用 daemon=False + SIGTERM 优雅退出，allow suzhou 落盘
- 停止任务时优先从 targets 快照恢复真实 box/conf，不再伪造数据
- cleanup 仅删除上传成功的文件，未传 upload_log 默认不删
- 视频流状态检查改为线程池并发 + 跳过 not_started
"""
from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import multiprocessing
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Optional

import cv2
import requests
from flask import Flask, jsonify, request

import app_config
from summary import build_summary_json
from task_store import read_task, safe_remove, update_task, write_json_atomic, write_task

multiprocessing.set_start_method('spawn', force=True)

ROOT = Path(__file__).resolve().parent
TASK_DIR = app_config.TASK_DIR
LOG_DIR = app_config.LOG_DIR
RESULT_DIR = app_config.RESULT_DIR

for _d in (TASK_DIR, LOG_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(_console)

_detailed = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'algorithm_api_detailed.log',
    maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8',
)
_detailed.setLevel(logging.DEBUG)
_detailed.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(_detailed)

_err = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'algorithm_api_error.log',
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8',
)
_err.setLevel(logging.ERROR)
_err.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(_err)

stream_logger = logging.getLogger('stream_check')
stream_logger.setLevel(logging.DEBUG)
stream_logger.propagate = False
_sh = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'stream_check.log',
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8',
)
_sh.setLevel(logging.DEBUG)
_sh.setFormatter(logging.Formatter(LOG_FORMAT))
stream_logger.addHandler(_sh)


app = Flask(__name__)

processes: dict = {}
process_lock = multiprocessing.Lock()

stream_check_running = False
stream_check_thread: Optional[Thread] = None

_shutdown_called = False


def run_processor(device_sn, task_id, task_name, task_info_dict):
    """子进程入口：通过参数注入 task_info，避免再次读 task json。"""
    import logging as _logging

    # 子进程独立配置日志
    sub_logger = _logging.getLogger('run_processor')
    if not sub_logger.handlers:
        sub_logger.setLevel(_logging.INFO)
        handler = _logging.StreamHandler()
        handler.setFormatter(_logging.Formatter(LOG_FORMAT))
        sub_logger.addHandler(handler)
        sub_logger.propagate = False

    processor = None
    task_json_path: Optional[Path] = None

    try:
        sub_logger.info(f"启动子进程 device_sn={device_sn} task_id={task_id}")
        from suzhou import VideoStreamProcessor  # 模块导入时已自动配置子进程日志

        processor = VideoStreamProcessor(
            device_sn=device_sn,
            task_id=task_id,
            task_name=task_name,
            task_info=task_info_dict,
        )
        sub_logger.info(f"子进程初始化完成 event_id={processor.event_id}")

        box_id = task_info_dict.get('boxId') or app_config.DEFAULT_BOX_ID
        candidates = [
            TASK_DIR / f"{box_id}_{task_id}.json",
            TASK_DIR / f"{processor.device_id}_{task_id}.json",
        ]
        for path in candidates:
            if path.exists():
                task_json_path = path
                break

        if task_json_path:
            update_task(task_json_path, lambda d: {**d, 'event_id': processor.event_id})
            sub_logger.info(f"event_id 已写入 {task_json_path.name}")

        processor.process(start_frame=0)

        if task_json_path:
            try:
                update_task(task_json_path, lambda d: {
                    **d,
                    'detected_count': len(processor.detected_targets),
                    'last_update': int(time.time()),
                })
            except Exception as e:
                sub_logger.warning(f"更新任务文件失败: {e}")

    except Exception as e:
        sub_logger.exception(f"子进程异常: {e}")
        if processor and processor.detected_targets:
            try:
                processor.generate_summary_json()
            except Exception as e2:
                sub_logger.exception(f"异常退出生成 summary 失败: {e2}")


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = app_config.CORS_ORIGIN
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response


@app.before_request
def log_request_info():
    if request.path in ('/create', '/status', '/query', '/algorithm', '/delete', '/bis-status'):
        logger.debug('请求头: %s', dict(request.headers))
        logger.debug('请求体: %s', request.get_data(as_text=True))


@app.errorhandler(400)
def bad_request_handler(error):
    logger.error(f"400 错误: {error}")
    return jsonify({"code": 1, "error": f"请求格式错误: {error}"}), 400


def _check_content_type() -> bool:
    return request.is_json or 'application/json' in (request.content_type or '')


def _missing(data: dict, params: list) -> Optional[str]:
    for p in params:
        if p not in data:
            return p
    return None


def _filter_summary(info: dict) -> dict:
    return {
        "boxId": info.get("boxId"),
        "task_id": str(info.get("task_id", "")),
        "status": info.get("status"),
        "task_name": info.get("task_name"),
        "categoryType": info.get("app_id"),
    }


@app.route('/create', methods=['POST', 'OPTIONS'])
def create_task():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not _check_content_type():
        return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"code": 1, "error": "空请求体"}), 400

    miss = _missing(data, ['boxId', 'video_url', 'task_id', 'categoryType'])
    if miss:
        return jsonify({"code": 1, "error": f"缺少参数: {miss}"}), 400

    if data.get('categoryType') != 'antenna':
        return jsonify({"code": 1, "error": "categoryType 只能为 'antenna'"}), 400

    status = data.get('status', 'not_started')
    if status not in ('running', 'not_started', 'stream_error'):
        return jsonify({"code": 1, "error": f"无效的状态值: {status}"}), 400

    task_id = str(data['task_id'])
    box_id = str(data['boxId'])
    file_path = TASK_DIR / f"{box_id}_{task_id}.json"

    existing = read_task(file_path)
    video_url = data.get('video_url')
    task_name = data.get('task_name', '')

    if existing:
        task_name = existing.get('task_name', task_name)
        video_url = existing.get('video_url', video_url)

    src_id = video_url.split('/')[-1] if video_url else None

    task_data = {
        "boxId": box_id,
        "video_url": video_url,
        "task_id": task_id,
        "status": status,
        "task_name": task_name,
        "app_name": task_name,
        "src_name": task_name,
        "src_id": src_id,
        "app_id": data['categoryType'],
    }

    try:
        write_task(file_path, task_data)
    except Exception as e:
        logger.exception(f"写入任务文件失败: {e}")
        return jsonify({"code": 1, "error": str(e)}), 500

    logger.info(f"创建任务成功: {file_path.name}")
    return jsonify({"code": 0, "data": {"message": "任务创建成功", "task_info": task_data}}), 200


def _start_task(box_id: str, task_id: str, file_path: Path, task_info: dict):
    task_key = f"{box_id}_{task_id}"

    with process_lock:
        # 在锁内重新读取任务状态，避免并发请求的竞态条件
        fresh_info = read_task(file_path)
        if fresh_info is None:
            return jsonify({"code": 1, "error": f"任务文件不存在: {file_path.name}"}), 404

        current_status = fresh_info.get('status')
        if current_status == 'running':
            return jsonify({"code": 1, "error": "任务已在运行中"}), 400
        if current_status not in ('not_started', 'stream_error'):
            return jsonify({
                "code": 1,
                "error": f"当前任务状态为 {current_status}，不允许启动",
            }), 400

        if task_key in processes and processes[task_key].is_alive():
            return jsonify({"code": 1, "error": f"任务 {task_key} 已在运行"}), 400

        device_sn = fresh_info.get('video_url')
        if not device_sn:
            return jsonify({"code": 1, "error": f"任务 {file_path.name} 中未找到 video_url"}), 400

        task_name = fresh_info.get('task_name', app_config.DEFAULT_TASK_NAME)
        # 清理已退出的旧进程条目
        for k in list(processes.keys()):
            if not processes[k].is_alive():
                processes.pop(k, None)

        p = multiprocessing.Process(
            target=run_processor,
            args=(device_sn, task_id, task_name, dict(fresh_info)),
            daemon=False,
        )
        processes[task_key] = p
        p.start()

    logger.info(f"任务启动成功: {file_path.name} pid={p.pid}")
    info_copy = {**fresh_info, "task_id": str(fresh_info.get('task_id', task_id))}
    return jsonify({"code": 0, "data": {"status": "开始处理", "task_info": info_copy}}), 200


def _terminate_subprocess(task_key: str, file_name: str) -> bool:
    """发送 SIGTERM 优雅退出，超时后 SIGKILL 强杀。"""
    proc = processes.get(task_key)
    if not proc or not proc.is_alive():
        return False
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except Exception as e:
        logger.warning(f"发送 SIGTERM 失败: {e}")

    proc.join(timeout=15)
    if proc.is_alive():
        logger.warning(f"子进程未在 15s 内退出，强制 SIGKILL: {file_name}")
        proc.kill()
        proc.join(timeout=5)
        if proc.is_alive():
            logger.error(f"子进程强杀失败: {file_name}")
    return True


def _load_snapshot(event_id: str) -> Optional[dict]:
    snap_path = RESULT_DIR / f"{event_id}_targets.json"
    if not snap_path.exists():
        return None
    try:
        with open(snap_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取 targets 快照失败: {e}")
        return None


def _stop_task(box_id: str, task_id: str, file_path: Path, task_info: dict):
    task_key = f"{box_id}_{task_id}"

    with process_lock:
        terminated = _terminate_subprocess(task_key, file_path.name)
        processes.pop(task_key, None)
        # 清理已退出的其他进程条目，防止字典无限增长
        for k in list(processes.keys()):
            if not processes[k].is_alive():
                processes.pop(k, None)

    if not terminated:
        logger.info(f"无运行中的子进程 {task_key}，尝试基于已有数据收尾")

    event_id = task_info.get('event_id')
    if not event_id:
        return jsonify({
            "code": 0,
            "data": {"status": "处理已停止", "message": "任务文件中没有 event_id，无 summary 生成"},
        }), 200

    summary_path = RESULT_DIR / f"{event_id}.json"

    if not summary_path.exists():
        snapshot = _load_snapshot(event_id)
        if snapshot and snapshot.get('targets'):
            try:
                summary = build_summary_json(
                    event_id=event_id,
                    task_info=snapshot.get('task_info', task_info),
                    detected_targets=snapshot['targets'],
                    frame_id=snapshot.get('frame_id', 0),
                )
                write_json_atomic(summary_path, summary)
                logger.info(f"基于快照重建 summary: {len(snapshot['targets'])} 个目标")
            except Exception as e:
                logger.exception(f"重建 summary 失败: {e}")
                return jsonify({"code": 0, "data": {"status": "处理已停止", "message": "summary 重建失败"}}), 200
        else:
            # 清理空的快照文件
            safe_remove(RESULT_DIR / f"{event_id}_targets.json")
            return jsonify({
                "code": 0,
                "data": {"status": "处理已停止", "message": "未找到 summary 或 targets 快照"},
            }), 200

    image_paths = sorted(
        [str(p) for p in RESULT_DIR.glob(f"{event_id}_*.jpeg")] +
        [str(p) for p in RESULT_DIR.glob(f"{event_id}_*.jpg")],
        key=os.path.getmtime,
        reverse=True,
    )

    upload_result = upload_files({"json_path": str(summary_path), "image_paths": image_paths})
    cleanup_result = cleanup_files(
        {"json_path": str(summary_path), "image_paths": image_paths},
        upload_log=upload_result.get('details'),
    )
    safe_remove(RESULT_DIR / f"{event_id}_targets.json")

    return jsonify({
        "code": 0,
        "data": {
            "status": "处理已停止",
            "message": "检测进程已终止并已生成/推送 JSON",
            "upload_result": upload_result,
            "cleanup_result": cleanup_result,
        },
    }), 200


@app.route('/status', methods=['POST', 'OPTIONS'])
def handle_status():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not _check_content_type():
        return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

    data = request.get_json(silent=True)

    if data == {}:
        tasks = []
        for fp in TASK_DIR.glob('*.json'):
            try:
                info = read_task(fp)
                if info:
                    tasks.append(_filter_summary(info))
            except Exception as e:
                logger.error(f"读取任务文件错误 {fp}: {e}")
        return jsonify({"code": 0, "data": tasks}), 200

    if not data:
        return jsonify({"code": 1, "error": "空请求体"}), 400

    miss = _missing(data, ['boxId', 'task_id', 'status'])
    if miss:
        return jsonify({"code": 1, "error": f"缺少参数: {miss}"}), 400

    box_id = str(data['boxId'])
    task_id = str(data['task_id'])
    status = data['status']

    file_path = TASK_DIR / f"{box_id}_{task_id}.json"
    task_info = read_task(file_path)
    if task_info is None:
        return jsonify({"code": 1, "error": f"任务不存在: {file_path.name}"}), 404

    try:
        if status == 0:
            return _start_task(box_id, task_id, file_path, task_info)
        if status == 1:
            return _stop_task(box_id, task_id, file_path, task_info)
        return jsonify({"code": 1, "error": "无效的状态值，使用 0 表示开始或 1 表示停止"}), 400
    except Exception as e:
        logger.exception(f"处理 /status 异常: {e}")
        return jsonify({"code": 1, "error": str(e)}), 500


@app.route('/query', methods=['POST', 'OPTIONS'])
def query_tasks():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not _check_content_type():
        return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"code": 1, "error": "空请求体"}), 400

    tasks = []
    if data == {}:
        for fp in TASK_DIR.glob('*.json'):
            info = read_task(fp)
            if info:
                tasks.append(_filter_summary(info))
        return jsonify({"code": 0, "data": tasks}), 200

    box_id = data.get('boxId')
    task_id = data.get('task_id')
    status = data.get('status')
    task_name = data.get('task_name')
    category_type = data.get('categoryType')

    for fp in TASK_DIR.glob('*.json'):
        info = read_task(fp)
        if not info:
            continue
        if box_id and str(info.get('boxId')) != str(box_id):
            continue
        if task_id and str(info.get('task_id')) != str(task_id):
            continue
        if status and str(info.get('status')) != str(status):
            continue
        if task_name and task_name not in (info.get('task_name') or ''):
            continue
        if category_type and str(info.get('app_id', '')) != str(category_type):
            continue
        tasks.append(_filter_summary(info))

    return jsonify({"code": 0, "data": tasks}), 200


@app.route('/algorithm', methods=['POST', 'OPTIONS'])
def handle_algorithm():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not _check_content_type():
        return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"code": 1, "error": "空请求体"}), 400

    miss = _missing(data, ['boxId', 'task_id', 'algorithm_status'])
    if miss:
        return jsonify({"code": 1, "error": f"缺少参数: {miss}"}), 400

    box_id = str(data['boxId'])
    task_id = str(data['task_id'])
    algo_status = str(data['algorithm_status'])

    if algo_status == '0':
        new_status = 'running'
    elif algo_status == '1':
        new_status = 'not_started'
    else:
        return jsonify({"code": 1, "error": "无效的 algorithm_status 值，0 启动 / 1 停止"}), 400

    file_path = TASK_DIR / f"{box_id}_{task_id}.json"
    if not file_path.exists():
        return jsonify({"code": 1, "error": f"任务不存在: {file_path.name}"}), 404

    new_info = update_task(file_path, lambda d: {**d, 'status': new_status})
    logger.info(f"算法状态更新: {new_status} - {file_path.name}")
    info_copy = {**(new_info or {}), "task_id": str(task_id)}
    return jsonify({"code": 0, "data": {"message": "算法状态更新成功", "task_info": info_copy}}), 200


@app.route('/delete', methods=['POST', 'OPTIONS'])
def delete_task():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not _check_content_type():
        return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"code": 1, "error": "空请求体"}), 400

    miss = _missing(data, ['boxId', 'task_id'])
    if miss:
        return jsonify({"code": 1, "error": f"缺少参数: {miss}"}), 400

    box_id = str(data['boxId'])
    task_id = str(data['task_id'])
    file_path = TASK_DIR / f"{box_id}_{task_id}.json"

    if not file_path.exists():
        return jsonify({"code": 1, "error": f"任务文件不存在: {file_path.name}"}), 404

    task_info = read_task(file_path) or {"boxId": box_id, "task_id": task_id}
    if not safe_remove(file_path):
        return jsonify({"code": 1, "error": "删除文件失败"}), 500

    logger.info(f"成功删除任务文件: {file_path.name}")
    info_copy = {**task_info, "task_id": str(task_info.get('task_id', task_id))}
    return jsonify({
        "code": 0,
        "data": {"message": "任务删除成功", "deleted_file": file_path.name, "task_info": info_copy},
    }), 200


@app.route('/bis-status', methods=['POST', 'OPTIONS'])
def bis_status():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not _check_content_type():
        return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"code": 1, "error": "空请求体"}), 400

    miss = _missing(data, ['boxId', 'task_id'])
    if miss:
        return jsonify({"code": 1, "error": f"缺少参数: {miss}"}), 400

    box_id = str(data['boxId'])
    task_id = str(data['task_id'])
    task_key = f"{box_id}_{task_id}"

    status = "not_started"
    with process_lock:
        if task_key in processes and processes[task_key].is_alive():
            status = "running"
    return jsonify({"code": 0, "data": {"boxId": box_id, "task_id": task_id, "status": status}}), 200


def check_video_stream(video_url: str) -> bool:
    cap = None
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            stream_logger.warning(f"无法打开视频流: {video_url}")
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        success_count = 0
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
            time.sleep(0.2)
        return success_count >= 3
    except Exception as e:
        stream_logger.error(f"检查视频流异常 {video_url}: {e}", exc_info=True)
        return False
    finally:
        if cap is not None:
            cap.release()


def _check_one_task(file_path: Path):
    try:
        info = read_task(file_path)
        if not info:
            return None
        current_status = info.get('status')
        if current_status not in ('running', 'stream_error'):
            return None
        video_url = info.get('video_url')
        if not video_url:
            return None
        new_status = 'running' if check_video_stream(video_url) else 'stream_error'
        if new_status != current_status:
            return file_path, new_status
        return None
    except Exception as e:
        stream_logger.error(f"检查 {file_path.name} 异常: {e}", exc_info=True)
        return None


def stream_status_check():
    global stream_check_running
    stream_logger.info("启动视频流状态检查线程")
    while stream_check_running:
        try:
            files = list(TASK_DIR.glob('*.json'))
            if not files:
                time.sleep(app_config.STREAM_CHECK_INTERVAL)
                continue
            with ThreadPoolExecutor(max_workers=app_config.STREAM_CHECK_WORKERS) as ex:
                results = list(ex.map(_check_one_task, files))
            updated = 0
            for r in results:
                if not r:
                    continue
                fp, new_status = r
                try:
                    update_task(fp, lambda d, s=new_status: {**d, 'status': s})
                    updated += 1
                except Exception as e:
                    stream_logger.error(f"更新 {fp.name} 状态失败: {e}")
            stream_logger.info(f"本轮检查: 总任务 {len(files)} 更新 {updated}")
            time.sleep(app_config.STREAM_CHECK_INTERVAL)
        except Exception as e:
            stream_logger.exception(f"流检查线程异常: {e}")
            time.sleep(10)


def start_stream_status_check():
    global stream_check_running, stream_check_thread
    if stream_check_thread and stream_check_thread.is_alive():
        logger.warning("视频流状态检查线程已在运行")
        return
    stream_check_running = True
    stream_check_thread = Thread(target=stream_status_check, daemon=True)
    stream_check_thread.start()
    logger.info("视频流状态检查线程已启动")


def upload_files(result: dict) -> dict:
    files_to_upload = []
    json_path = result['json_path']
    if os.path.exists(json_path):
        files_to_upload.append(("json", json_path))
    for img_path in result.get('image_paths', []):
        if os.path.exists(img_path):
            files_to_upload.append(("image", img_path))

    if not app_config.upload_enabled():
        logger.warning("UPLOAD_URL/凭证未配置，跳过上传")
        return {
            "total_files": len(files_to_upload),
            "success_json": 0, "success_images": 0,
            "details": [], "skipped": True,
        }

    box_id = app_config.DEFAULT_BOX_ID
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            box_id = str(json.load(f).get('boxId', box_id))
    except Exception as e:
        logger.warning(f"读 boxId 失败，用默认 {box_id}: {e}")

    params = {
        'sysId': app_config.UPLOAD_SYS_ID,
        'boxId': box_id,
        'type': app_config.UPLOAD_TYPE,
    }
    headers = {
        "AppKeyID": app_config.UPLOAD_APP_KEY_ID,
        "AppKeySecret": app_config.UPLOAD_APP_KEY_SECRET,
    }
    upload_log = []

    def _do_upload(file_type: str, file_path: str):
        try:
            with open(file_path, 'rb') as f:
                resp = requests.post(
                    app_config.UPLOAD_URL,
                    params=params,
                    files={'file': (os.path.basename(file_path), f)},
                    headers=headers,
                    timeout=app_config.UPLOAD_TIMEOUT,
                )
            upload_log.append({
                "file": os.path.basename(file_path),
                "type": file_type,
                "status": resp.status_code,
                "response": resp.text[:100],
            })
            logger.info(f"{file_type} 上传 {os.path.basename(file_path)} status={resp.status_code}")
        except Exception as e:
            logger.error(f"{file_type} 上传失败 {os.path.basename(file_path)}: {e}")
            upload_log.append({
                "file": os.path.basename(file_path),
                "type": file_type,
                "error": str(e),
            })

    for ft, fp in files_to_upload:
        if ft == "json":
            _do_upload(ft, fp)
    time.sleep(1)
    for ft, fp in files_to_upload:
        if ft == "image":
            _do_upload(ft, fp)

    success_json = sum(1 for x in upload_log if x.get('type') == 'json' and x.get('status') == 200)
    success_images = sum(1 for x in upload_log if x.get('type') == 'image' and x.get('status') == 200)
    return {
        "total_files": len(files_to_upload),
        "success_json": success_json,
        "success_images": success_images,
        "details": upload_log,
    }


def cleanup_files(result: dict, upload_log: Optional[list] = None) -> dict:
    """只删除上传成功的文件；upload_log 为 None 时为安全起见不删。"""
    cleanup_log = []
    if upload_log is None:
        logger.warning("cleanup_files 未传 upload_log，跳过删除以避免数据丢失")
        return {"success": True, "deleted_files": 0, "details": []}

    success_set = {x.get('file') for x in upload_log if x.get('status') == 200}

    try:
        json_path = result['json_path']
        if os.path.exists(json_path) and os.path.basename(json_path) in success_set:
            if safe_remove(Path(json_path)):
                cleanup_log.append({"file": os.path.basename(json_path), "type": "json", "status": "deleted"})

        for img_path in result.get('image_paths', []):
            if os.path.exists(img_path) and os.path.basename(img_path) in success_set:
                if safe_remove(Path(img_path)):
                    cleanup_log.append({"file": os.path.basename(img_path), "type": "image", "status": "deleted"})

        return {"success": True, "deleted_files": len(cleanup_log), "details": cleanup_log}
    except Exception as e:
        logger.error(f"清理文件失败: {e}")
        return {"success": False, "error": str(e), "deleted_files": len(cleanup_log), "details": cleanup_log}


def _shutdown_subprocesses(signum=None, _frame=None):
    """主进程退出时优雅终止所有子进程，允许 suzhou 写完 summary。
    注意：在信号处理函数中，不能持有锁调用 join，否则可能死锁。
    只在主线程中注册，非主线程会静默跳过。"""
    import threading
    if threading.current_thread() is not threading.main_thread():
        return

    global _shutdown_called
    if _shutdown_called:
        return
    _shutdown_called = True
    logger.info("主进程退出中，开始终止子进程")

    procs_to_terminate = []
    with process_lock:
        for key, proc in list(processes.items()):
            if proc.is_alive():
                procs_to_terminate.append((key, proc))
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except Exception:
                    pass

    # 等待子进程退出（不持锁，避免死锁）
    for key, proc in procs_to_terminate:
        proc.join(timeout=15)
        if proc.is_alive():
            logger.warning(f"子进程 {key} 未在 15s 内退出，发送 SIGKILL")
            try:
                proc.kill()
                proc.join(timeout=5)
            except Exception:
                pass

    with process_lock:
        processes.clear()

    if signum:
        os._exit(0)


atexit.register(_shutdown_subprocesses)
try:
    signal.signal(signal.SIGTERM, _shutdown_subprocesses)
    signal.signal(signal.SIGINT, _shutdown_subprocesses)
except Exception as e:
    logger.warning(f"注册信号处理失败: {e}")


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("算法 API 服务启动")
    logger.info(f"地址: http://{app_config.API_HOST}:{app_config.API_PORT}")
    logger.info(f"上传服务: {'已启用' if app_config.upload_enabled() else '未配置（跳过上传）'}")
    logger.info(f"流检查间隔: {app_config.STREAM_CHECK_INTERVAL}s, 并发: {app_config.STREAM_CHECK_WORKERS}")
    logger.info("=" * 80)

    start_stream_status_check()
    app.run(host=app_config.API_HOST, port=app_config.API_PORT, threaded=True)
