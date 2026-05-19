"""任务 JSON 文件的并发安全读写工具。

并发风险点：
- algorithm_api 的 Flask 线程、stream_status_check 后台线程、run_processor 子进程
  都会读写 TASK_DIR 下同一份 json，未加锁会导致写入丢失或文件残缺。
- 解决方案：fcntl.flock 互斥锁 + 原子写（临时文件 + os.replace）。
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional


@contextlib.contextmanager
def _file_lock(path: Path, exclusive: bool = True):
    """对指定路径加 flock。lock 文件单独后缀，避免覆盖目标文件。"""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def read_task(path: Path) -> Optional[dict]:
    """带共享锁读取任务 json，文件不存在返回 None。"""
    if not path.exists():
        return None
    with _file_lock(path, exclusive=False):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def write_task(path: Path, data: dict) -> None:
    """原子写入：先写 .tmp，再 fsync，再 os.replace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _file_lock(path, exclusive=True):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def update_task(path: Path, mutator: Callable[[dict], dict]) -> Optional[dict]:
    """读-改-写，全程持有排他锁。文件不存在返回 None 不创建。"""
    if not path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path, exclusive=True):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        new_data = mutator(data) or data
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return new_data


def safe_remove(path: Path) -> bool:
    """安全删除文件及其 .lock 锁文件，不存在视为成功。"""
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            return False
    # 同步清理关联的锁文件
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return True


def write_json_atomic(path: Path, data: Any) -> None:
    """通用原子写入（不加锁，调用方需保证无并发）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
