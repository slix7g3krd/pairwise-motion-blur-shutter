"""运动模糊快门参考实现。

缺陷与修复一一对应：

1. 快门区间按帧时间估算，掉帧后拖影长度突变
   -> ``shutter_time``：快门与帧时间解耦，曝光窗口只由曝光时间与快门偏移决定，
      帧间隔仅用于越界判定；``frame_drop`` 在速度域补偿掉帧。
2. 相机旋转与平移分别采样导致方向错
   -> ``vector_compose``：旋转量先变换到像素空间，再与平移矢量合成单一矢量；
      ``rotation_path`` 保证快门区间内旋转采样时刻与曝光窗口对齐。
3. 历史帧重用与抗锯齿冲突
   -> ``history_split``：模糊重投影历史与 TAA 抗锯齿历史拆成两个独立缓冲，
      各自携带有效标志，一边失效不污染另一边。
4. 物体运动矢量在蒙皮后没有更新
   -> ``skin_motion``：蒙皮完成后用新的蒙皮顶点位置重算每顶点运动矢量，
      不沿用蒙皮前缓存的旧矢量。
5. 异步后处理完成时该帧已经上屏
   -> ``result_align`` / ``async_post``：结果携带源帧时间戳，只允许合入尚未
      上屏的目标帧，已上屏或错帧一律丢弃。
6. 拖影长度突变（无上限）
   -> ``trail_cap``：像素级拖影上限，超限按原方向等比压缩。

其余函数（``validation`` / ``audit`` / ``fallback`` / ``metrics`` /
``determinism``）提供配置校验、病理审计、安全降级、指标统计与确定性验证。

所有公开函数均为纯函数：不读系统时钟、不用随机数，同一输入永远得到同一输出。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

__all__ = [
    "shutter_time",
    "vector_compose",
    "history_split",
    "skin_motion",
    "result_align",
    "trail_cap",
    "rotation_path",
    "validation",
    "audit",
    "fallback",
    "frame_drop",
    "metrics",
    "camera_jitter",
    "async_post",
    "determinism",
]

_EPS = 1e-8

# 安全缺省值（60Hz 管线）
DEFAULT_EXPOSURE = 1.0 / 60.0
DEFAULT_FRAME_TIME = 1.0 / 60.0
DEFAULT_SHUTTER_OFFSET = -0.5      # 居中快门：窗口中心落在帧时间中点
DEFAULT_FOCAL_PX = 500.0           # 角位移换算像素位移用的默认焦距
DEFAULT_MAX_TRAIL_PX = 120.0       # 拖影像素上限
FALLBACK_MAX_TRAIL_PX = 64.0       # 降级时收紧的拖影上限
DEFAULT_ROTATION_STEPS = 5


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _as_float(value: Any, default: float = 0.0) -> float:
    """宽松浮点转换：无法解释时回退到 default，绝不向调用方抛类型错误。"""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick(payload: Any, key: str, explicit: Any, default: Any) -> Any:
    """关键字参数优先，其次 payload dict，最后使用 default。"""
    if explicit is not None:
        return explicit
    if isinstance(payload, dict):
        value = payload.get(key, default)
        return default if value is None else value
    return default


def _as_vector(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, ...]:
    """把 dict({x,y[,z]}) / 序列 / 标量统一成浮点元组矢量。"""
    if value is None:
        return tuple(float(c) for c in default)
    if isinstance(value, dict):
        axes = ("x", "y", "z", "w")
        present = [axis for axis in axes if axis in value]
        if not present:
            return tuple(float(c) for c in default)
        return tuple(_as_float(value[axis], 0.0) for axis in present)
    if isinstance(value, (bool, int, float)):
        # 标量 shim：约定为沿 x 轴的矢量，使退化输入也有确定语义
        return (float(value),) + tuple(0.0 for _ in range(len(default) - 1))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # 维度随输入推导，但不短于默认维度（保证下游矢量运算形状稳定）
        dim = max(len(default), len(value))
        out = []
        for i in range(dim):
            if i < len(value):
                fallback = default[i] if i < len(default) else 0.0
                out.append(_as_float(value[i], fallback))
            else:
                out.append(float(default[i] if i < len(default) else 0.0))
        return tuple(out)
    raise TypeError(f"无法把 {value!r} 解释为矢量")


def _as_point(value: Any) -> tuple[float, ...]:
    """顶点位置：支持 {"pos": (...)} / {"x","y"(,"z")} / 坐标序列。"""
    if isinstance(value, dict):
        if "pos" in value and value["pos"] is not None:
            return tuple(float(c) for c in value["pos"])
        axes = ("x", "y", "z")
        present = [a for a in axes if a in value]
        return tuple(_as_float(value[a]) for a in present) if present else (0.0, 0.0)
    return tuple(float(c) for c in value)


def _vadd(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def _vscale(a: Sequence[float], s: float) -> tuple[float, ...]:
    return tuple(float(x) * s for x in a)


def _vlen(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def _halton(index: int, base: int) -> float:
    """Halton 低差异序列第 index 项（index 从 1 开始），确定性、无随机数。"""
    fraction = 1.0
    result = 0.0
    i = max(0, int(index))
    while i > 0:
        fraction /= base
        result += fraction * (i % base)
        i //= base
    return result


# ---------------------------------------------------------------------------
# 修复 1：快门与帧时间解耦
# ---------------------------------------------------------------------------

def shutter_time(payload: Any = None, *, exposure: float | None = None,
                 frame_time: float | None = None,
                 shutter_offset: float | None = None,
                 clamp_to_frame: bool | None = None) -> dict[str, Any]:
    """计算快门曝光窗口，窗口长度与帧间隔完全解耦。

    曝光窗口只由 ``exposure``（秒）与 ``shutter_offset`` 决定：
    ``shutter_offset`` 为 -0.5 时窗口居中，0 时对齐帧首，-1 时对齐帧尾。
    ``frame_time`` 仅用于判定窗口是否越界，不参与窗口长度计算——因此掉帧
    导致 ``frame_time`` 变长时，曝光时长不变，拖影长度不会突变。

    返回窗口的 open/close（相对当前帧起始时刻的秒数）、实际曝光时长、
    越界标志与（可选的）截断结果。
    """
    if payload is not None and not isinstance(payload, dict) and frame_time is None:
        # 标量 shim：入参解释为 frame_time
        frame_time = payload

    exposure_s = _as_float(_pick(payload, "exposure", exposure, DEFAULT_EXPOSURE))
    frame_time_s = _as_float(
        _pick(payload, "frame_time", frame_time, DEFAULT_FRAME_TIME)
    )
    offset = _as_float(
        _pick(payload, "shutter_offset", shutter_offset, DEFAULT_SHUTTER_OFFSET)
    )
    do_clamp = bool(_pick(payload, "clamp_to_frame", clamp_to_frame, False))

    warnings: list[str] = []
    if exposure_s <= 0.0:
        warnings.append(f"非法曝光时间 {exposure_s}，回退默认值 {DEFAULT_EXPOSURE}")
        exposure_s = DEFAULT_EXPOSURE
    if frame_time_s <= 0.0:
        warnings.append(f"非法帧间隔 {frame_time_s}，回退默认值 {DEFAULT_FRAME_TIME}")
        frame_time_s = DEFAULT_FRAME_TIME

    # 窗口中心约定（相对当前帧起始时刻），公式 center = -offset * frame_time：
    # offset=0    -> center=0          帧首对齐
    # offset=-0.5 -> center=frame/2    居中
    # offset=-1   -> center=frame      帧尾对齐
    center = -offset * frame_time_s
    open_t = center - exposure_s * 0.5
    close_t = center + exposure_s * 0.5

    overrun = open_t < -_EPS or close_t > frame_time_s + _EPS
    if overrun:
        warnings.append(
            f"曝光窗口 [{open_t:.6f}, {close_t:.6f}] 越出帧区间 "
            f"[0, {frame_time_s:.6f}]（曝光与帧时间解耦，窗口长度保持不变）"
        )

    clamped = False
    if do_clamp:
        clamped_open = max(0.0, open_t)
        clamped_close = min(frame_time_s, close_t)
        clamped = (
            abs(clamped_open - open_t) > _EPS
            or abs(clamped_close - close_t) > _EPS
        )
        open_t, close_t = clamped_open, clamped_close

    return {
        "open": open_t,
        "close": close_t,
        "center": center,
        "exposure": close_t - open_t if do_clamp else exposure_s,
        "requested_exposure": exposure_s,
        "frame_time": frame_time_s,
        "shutter_offset": offset,
        "overrun": overrun,
        "clamped": clamped,
        "warnings": warnings,
    }


def rotation_path(payload: Any = None, *, angle_start: float | None = None,
                  angle_end: float | None = None, steps: int | None = None,
                  radius: float | None = None) -> dict[str, Any]:
    """在快门区间内对相机旋转做等步长采样。

    采样点的 ``t`` 归一化到 [0, 1]（对应快门 open -> close），每个采样点给出
    累计角度与单位圆上的旋转偏移，供 ``vector_compose`` 在同一时刻基准上
    与平移合成，避免旋转、平移各采各的造成方向错。
    """
    if payload is not None and not isinstance(payload, dict) and angle_end is None:
        angle_end = payload  # 标量 shim：解释为角位移终点（弧度）

    start = _as_float(_pick(payload, "angle_start", angle_start, 0.0))
    end = _as_float(_pick(payload, "angle_end", angle_end, 1.0))
    n_steps = int(_pick(payload, "steps", steps, DEFAULT_ROTATION_STEPS))
    rad = _as_float(_pick(payload, "radius", radius, 1.0))
    if n_steps < 1:
        n_steps = 1

    samples = []
    for i in range(n_steps + 1):
        t = i / n_steps
        angle = start + (end - start) * t
        samples.append({
            "t": t,
            "angle": angle,
            "offset": (rad * math.cos(angle), rad * math.sin(angle)),
        })

    return {
        "samples": samples,
        "angle_start": start,
        "angle_end": end,
        "angle_delta": end - start,
        "steps": n_steps,
        "radius": rad,
    }


# ---------------------------------------------------------------------------
# 修复 2：旋转与平移合成单一矢量
# ---------------------------------------------------------------------------

def vector_compose(payload: Any = None, *, translation: Any = None,
                   rotation: Any = None, focal_px: float | None = None,
                   frame_time: float | None = None) -> dict[str, Any]:
    """把相机旋转与平移合成同一像素空间中的单一运动矢量。

    ``rotation`` 支持两种形态：

    * 标量或 ``{"angle": θ[, "axis": "x"|"y"]}``：相机角位移（弧度），
      按针孔模型换算为像素位移 ``focal_px * tan(θ)``，默认沿屏幕 x 轴；
    * ``{"x","y"}`` 或坐标序列：已是像素空间的旋转矢量，直接使用。

    合成结果 ``vector = translation + rotation_px`` 是下游唯一使用的运动
    矢量，旋转和平移不再被分别采样。
    """
    if payload is not None and not isinstance(payload, dict) and rotation is None:
        rotation = payload  # 标量 shim：解释为旋转角（弧度）

    t_input = _pick(payload, "translation", translation, None)
    if isinstance(t_input, dict):
        dim = len([a for a in ("x", "y", "z", "w") if a in t_input]) or 2
    elif isinstance(t_input, Sequence) and not isinstance(t_input, (str, bytes)):
        dim = len(t_input)
    elif isinstance(t_input, (int, float)):
        dim = 1
    else:
        dim = 2
    t_vec = _as_vector(t_input, (0.0,) * dim)
    focal = _as_float(_pick(payload, "focal_px", focal_px, DEFAULT_FOCAL_PX))
    dt = _as_float(_pick(payload, "frame_time", frame_time, DEFAULT_FRAME_TIME))

    rot_input = _pick(payload, "rotation", rotation, None)
    dim = len(t_vec)
    zero_rot: tuple[float, ...] = (0.0,) * dim
    if rot_input is None:
        r_vec = zero_rot
    elif isinstance(rot_input, dict) and "angle" in rot_input:
        angle = _as_float(rot_input.get("angle"))
        axis = rot_input.get("axis", "x")
        displacement = focal * math.tan(angle)
        r_vec = tuple(displacement if axis == a else 0.0 for a in ("x", "y", "z")[:dim])
    elif isinstance(rot_input, (int, float)):
        r_vec = (focal * math.tan(float(rot_input)),) + (0.0,) * (dim - 1)
    else:
        r_vec = _as_vector(rot_input, zero_rot)
        # 维度对齐：短了补零，长了截断，保证与平移可逐分量相加
        if len(r_vec) < dim:
            r_vec = r_vec + (0.0,) * (dim - len(r_vec))
        elif len(r_vec) > dim:
            r_vec = r_vec[:dim]

    composed = _vadd(t_vec, r_vec)
    magnitude = _vlen(composed)
    if dim >= 2 and magnitude > _EPS:
        direction = math.atan2(composed[1], composed[0])
    elif dim == 1:
        direction = 0.0 if composed[0] >= 0.0 else math.pi
    else:
        direction = 0.0

    return {
        "vector": composed,
        "translation": t_vec,
        "rotation_px": r_vec,
        "magnitude": magnitude,
        "direction": direction,
        "focal_px": focal,
        "frame_time": dt,
    }


# ---------------------------------------------------------------------------
# 修复 3：模糊历史与抗锯齿历史分离
# ---------------------------------------------------------------------------

def history_split(payload: Any = None, *, entries: Sequence | None = None,
                  size: int | None = None) -> dict[str, Any]:
    """拆分历史缓冲：运动模糊重投影历史与 TAA 抗锯齿历史各自独立。

    两份缓冲拥有独立的存储与有效标志：相机跳变或 disocclusion 使模糊历史
    失效时，TAA 历史仍可正常重用，反之亦然，杜绝“历史帧重用与抗锯齿冲突”。

    入参 ``entries`` 为历史帧记录（可为帧 id 或带 ``valid`` 标志的 dict）。
    """
    if payload is not None and not isinstance(payload, dict) and size is None and entries is None:
        size = payload  # 标量 shim：解释为缓冲容量

    raw_entries = _pick(payload, "entries", entries, [])
    capacity = int(_pick(payload, "size", size, max(1, len(raw_entries))))
    if capacity < 1:
        capacity = 1

    def _buffer() -> dict[str, Any]:
        slots: list[Any] = []
        valid_flags: list[bool] = []
        for entry in raw_entries[:capacity]:
            if isinstance(entry, dict):
                slots.append(entry.get("frame", entry))
                valid_flags.append(bool(entry.get("valid", True)))
            else:
                slots.append(entry)
                valid_flags.append(True)
        return {
            "entries": slots,
            "valid_flags": valid_flags,
            "size": capacity,
            "head": len(slots) % capacity,
            "invalidated": False,
        }

    blur_buffer = _buffer()
    aa_buffer = _buffer()
    return {
        "blur_history": blur_buffer,
        "aa_history": aa_buffer,
        "separate_storage": True,
        # 两个缓冲是不同对象，写一边不会别名到另一边
        "aliased": blur_buffer["entries"] is aa_buffer["entries"],
    }


# ---------------------------------------------------------------------------
# 修复 4：蒙皮后重算运动矢量
# ---------------------------------------------------------------------------

def skin_motion(payload: Any = None, *, vertices: Sequence | None = None,
                prev_vertices: Sequence | None = None,
                frame_time: float | None = None) -> dict[str, Any]:
    """蒙皮完成后，用新的蒙皮顶点位置重算每顶点运动矢量。

    矢量 = （当前蒙皮位置 - 上一帧蒙皮位置） / frame_time。顶点按 id 配对
    （无 id 时按索引配对）；上一帧不存在的顶点视为 disocclusion，矢量置零
    并单独标记——绝不沿用蒙皮前缓存的旧矢量。
    """
    current = _pick(payload, "vertices", vertices, None)
    if current is None and isinstance(payload, dict):
        current = payload.get("current")
    previous = _pick(payload, "prev_vertices", prev_vertices, None)
    if previous is None and isinstance(payload, dict):
        previous = payload.get("previous", [])
    dt = _as_float(_pick(payload, "frame_time", frame_time, DEFAULT_FRAME_TIME))
    if dt <= 0.0:
        dt = DEFAULT_FRAME_TIME

    current = list(current or [])
    previous = list(previous or [])

    prev_by_id: dict[Any, tuple[float, ...]] = {}
    indexed_prev: list[tuple[float, ...]] = []
    for idx, vertex in enumerate(previous):
        pos = _as_point(vertex)
        indexed_prev.append(pos)
        if isinstance(vertex, dict) and "id" in vertex:
            prev_by_id[vertex["id"]] = pos

    use_id = current and isinstance(current[0], dict) and "id" in current[0]
    results: list[dict[str, Any]] = []
    disoccluded_count = 0

    for idx, vertex in enumerate(current):
        pos = _as_point(vertex)
        vertex_id = vertex.get("id") if isinstance(vertex, dict) else idx

        if use_id and vertex_id in prev_by_id:
            prev_pos = prev_by_id[vertex_id]
            disoccluded = False
        elif not use_id and idx < len(indexed_prev):
            prev_pos = indexed_prev[idx]
            disoccluded = False
        else:
            prev_pos = (0.0,) * len(pos)
            disoccluded = True
            disoccluded_count += 1

        if disoccluded:
            vector = (0.0,) * len(pos)
        else:
            delta = tuple(p - q for p, q in zip(pos, prev_pos))
            vector = tuple(d / dt for d in delta)

        results.append({
            "id": vertex_id,
            "pos": pos,
            "vector": vector,
            "magnitude": _vlen(vector),
            "disoccluded": disoccluded,
        })

    return {
        "vertices": results,
        "vectors": [item["vector"] for item in results],
        "frame_time": dt,
        "disoccluded_count": disoccluded_count,
        "updated_after_skinning": True,
    }


# ---------------------------------------------------------------------------
# 修复 5：异步后处理结果帧对齐
# ---------------------------------------------------------------------------

def result_align(payload: Any = None, *, source_frame_time: Any = None,
                 target_frame_time: Any = None, presented: bool | None = None,
                 present_time: float | None = None,
                 arrival_time: float | None = None,
                 result: Any = None) -> dict[str, Any]:
    """异步后处理结果按源帧时间戳对齐到目标帧。

    只有目标帧尚未上屏（``presented=False``，或结果到达时间早于上屏时间）
    且源帧与目标帧一致时才允许合入；目标帧已上屏或帧不匹配一律丢弃，
    防止过期帧的后处理效果被贴到当前帧（“完成时该帧已经上屏”）。
    """
    def _pick_aliased(key: str, aliases: tuple[str, ...], explicit: Any) -> Any:
        value = _pick(payload, key, explicit, None)
        if value is None and isinstance(payload, dict):
            for alias in aliases:
                if payload.get(alias) is not None:
                    return payload[alias]
        return value

    src = _pick_aliased(
        "source_frame_time", ("source_frame", "source"), source_frame_time
    )
    dst = _pick_aliased(
        "target_frame_time", ("target_frame", "target"), target_frame_time
    )
    output = _pick(payload, "result", result, None)
    is_presented = _pick(payload, "presented", presented, None)
    present_t = _pick(payload, "present_time", present_time, None)
    arrival_t = _pick(payload, "arrival_time", arrival_time, None)

    if is_presented is None:
        if present_t is not None and arrival_t is not None:
            is_presented = _as_float(arrival_t) >= _as_float(present_t)
        else:
            is_presented = False
    is_presented = bool(is_presented)

    frame_mismatch = src is not None and dst is not None and src != dst
    aligned = not is_presented and not frame_mismatch

    if frame_mismatch:
        reason = "frame_mismatch"
    elif is_presented:
        reason = "target_already_presented"
    else:
        reason = "ok"

    return {
        "status": "applied" if aligned else "discarded",
        "aligned": aligned,
        "reason": reason,
        "source_frame_time": src,
        "target_frame_time": dst,
        "presented": is_presented,
        "result": output,
    }


def async_post(payload: Any = None, *, jobs: Sequence[dict] | None = None,
               now: float | None = None,
               presented_frames: Sequence | None = None) -> dict[str, Any]:
    """异步后处理队列调度：完成的任务逐个走 ``result_align`` 对齐判定。

    每个 job 可携带 ``ready``/``complete_time``（未完成则留在 pending）、
    ``source_frame_time``、``target_frame_time``、``presented`` 与 ``result``。
    ``presented_frames`` 给出已上屏帧时间戳集合，命中即丢弃结果。
    """
    job_list = list(_pick(payload, "jobs", jobs, []) or [])
    current_time = _pick(payload, "now", now, None)
    presented_set = set(_pick(payload, "presented_frames", presented_frames, []) or [])

    applied: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for job in job_list:
        job_id = job.get("job_id", job.get("id"))
        ready = bool(job.get("ready", True))
        complete_time = job.get("complete_time")
        if not ready or (complete_time is not None and current_time is not None
                         and _as_float(complete_time) > _as_float(current_time)):
            pending.append({"job_id": job_id, "complete_time": complete_time})
            continue

        src = job.get("source_frame_time", job.get("source_frame"))
        already_presented = bool(job.get("presented", False)) or src in presented_set
        verdict = result_align({
            "source_frame_time": src,
            "target_frame_time": job.get("target_frame_time", job.get("target_frame")),
            "presented": already_presented,
            "present_time": job.get("present_time"),
            "arrival_time": job.get("arrival_time", current_time),
            "result": job.get("result"),
        })
        verdict["job_id"] = job_id
        (applied if verdict["aligned"] else discarded).append(verdict)

    return {
        "applied": applied,
        "discarded": discarded,
        "pending": pending,
        "applied_count": len(applied),
        "discarded_count": len(discarded),
        "pending_count": len(pending),
    }


# ---------------------------------------------------------------------------
# 修复 6：拖影长度上限
# ---------------------------------------------------------------------------

def trail_cap(payload: Any = None, *, vectors: Any = None,
              max_trail_px: float | None = None) -> dict[str, Any]:
    """拖影长度像素上限：超限矢量按原方向等比压缩，方向保持不变。

    入参可为单个矢量或矢量列表（dict/序列/标量均可，维度不限）。
    单矢量输入时结果放在 ``capped``；列表输入时每个元素的结果放在
    ``results``，并汇总超限数量与压缩比。
    """
    if payload is not None and not isinstance(payload, dict) and vectors is None:
        vectors = payload  # 标量 shim：沿 x 轴的 1 像素矢量

    raw = _pick(payload, "vectors", vectors, None)
    limit = _as_float(_pick(payload, "max_trail_px", max_trail_px, DEFAULT_MAX_TRAIL_PX))
    if limit <= 0.0:
        limit = DEFAULT_MAX_TRAIL_PX

    if raw is None:
        raw = {"x": 0.0, "y": 0.0}
    # dict 与数值视为单矢量；list/tuple 视为矢量集合
    single = isinstance(raw, dict) or isinstance(raw, (int, float, bool))
    vector_inputs = [raw] if single else list(raw)

    results: list[dict[str, Any]] = []
    capped_count = 0
    for item in vector_inputs:
        vec = _as_vector((item, 0.0) if isinstance(item, (int, float)) else item)
        magnitude = _vlen(vec)
        if magnitude > limit and magnitude > _EPS:
            scale = limit / magnitude
            capped_vec = _vscale(vec, scale)
            capped = True
            capped_count += 1
        else:
            scale = 1.0
            capped_vec = vec
            capped = False
        results.append({
            "vector": capped_vec,
            "original_vector": vec,
            "original_magnitude": magnitude,
            "magnitude": _vlen(capped_vec),
            "scale": scale,
            "capped": capped,
        })

    summary: dict[str, Any] = {
        "results": results,
        "max_trail_px": limit,
        "capped_count": capped_count,
        "total_count": len(results),
    }
    if single:
        summary["capped"] = results[0]
    return summary


# ---------------------------------------------------------------------------
# 掉帧：曝光不动，速度域补偿，拖影不突变
# ---------------------------------------------------------------------------

def frame_drop(payload: Any = None, *, frame_time: float | None = None,
               expected_frame_time: float | None = None,
               exposure: float | None = None,
               max_trail_px: float | None = None,
               shutter_offset: float | None = None) -> dict[str, Any]:
    """掉帧处理：快门窗口保持不变，在速度域把长帧位移均摊回标准周期。

    掉帧数 ``round(dt / expected_dt) - 1``。物体在长帧内积累的位移按
    ``velocity_scale = expected_dt / dt`` 缩回，使快门区间内积分出的
    拖影长度与稳态帧一致；曝光窗口经 ``shutter_time`` 计算，长度不随
    掉帧变化，最后由 ``trail_cap`` 做像素级兜底。
    """
    if payload is not None and not isinstance(payload, dict) and frame_time is None:
        frame_time = payload

    dt = _as_float(_pick(payload, "frame_time", frame_time, DEFAULT_FRAME_TIME))
    expected_dt = _as_float(
        _pick(payload, "expected_frame_time", expected_frame_time, DEFAULT_FRAME_TIME)
    )
    if dt <= 0.0:
        dt = DEFAULT_FRAME_TIME
    if expected_dt <= 0.0:
        expected_dt = DEFAULT_FRAME_TIME

    dropped = max(0, round(dt / expected_dt) - 1)
    velocity_scale = expected_dt / dt  # < 1：长帧速度被均摊，拖影长度不突变

    shutter = shutter_time({
        "exposure": _pick(payload, "exposure", exposure, DEFAULT_EXPOSURE),
        "frame_time": dt,
        "shutter_offset": _pick(
            payload, "shutter_offset", shutter_offset, DEFAULT_SHUTTER_OFFSET
        ),
    })
    limit = _as_float(_pick(payload, "max_trail_px", max_trail_px, DEFAULT_MAX_TRAIL_PX))

    return {
        "dropped_frames": dropped,
        "frame_time": dt,
        "expected_frame_time": expected_dt,
        "velocity_scale": velocity_scale,
        "shutter": shutter,
        # 关键不变量：曝光时长与掉帧无关
        "exposure_unchanged": abs(shutter["requested_exposure"]
                                  - _as_float(_pick(payload, "exposure", exposure,
                                                    DEFAULT_EXPOSURE))) <= _EPS,
        "max_trail_px": limit,
    }


# ---------------------------------------------------------------------------
# TAA 相机抖动：确定性序列，只进投影矩阵，不进运动矢量
# ---------------------------------------------------------------------------

def camera_jitter(payload: Any = None, *, frame_index: int | None = None,
                  scale_px: float | None = None) -> dict[str, Any]:
    """TAA 子像素抖动（Halton(2,3)），与运动模糊严格隔离。

    抖动只作用于投影采样位置；``blur_motion`` 恒为零矢量，保证拖影方向
    与长度不会被逐帧抖动带偏。
    """
    if payload is not None and not isinstance(payload, dict) and frame_index is None:
        frame_index = payload

    index = int(_pick(payload, "frame_index", frame_index, 0))
    scale = _as_float(_pick(payload, "scale_px", scale_px, 1.0))

    jitter = (
        (_halton(index + 1, 2) - 0.5) * scale,
        (_halton(index + 1, 3) - 0.5) * scale,
    )
    return {
        "frame_index": index,
        "projection_jitter": jitter,
        "blur_motion": (0.0, 0.0),
        "scale_px": scale,
        "affects_motion_vectors": False,
    }


# ---------------------------------------------------------------------------
# 配置校验 / 病理审计 / 安全降级 / 指标 / 确定性
# ---------------------------------------------------------------------------

def validation(payload: Any = None) -> dict[str, Any]:
    """校验快门与管线配置，返回错误（阻断级）与警告（可运行但有风险）。"""
    if payload is None or not isinstance(payload, dict):
        config = {} if payload is None else {"frame_time": payload}
    else:
        config = payload

    raw_exposure = config.get("exposure", DEFAULT_EXPOSURE)
    raw_frame_time = config.get("frame_time", DEFAULT_FRAME_TIME)
    raw_offset = config.get("shutter_offset", DEFAULT_SHUTTER_OFFSET)
    raw_max_trail = config.get("max_trail_px", DEFAULT_MAX_TRAIL_PX)
    exposure = _as_float(raw_exposure, DEFAULT_EXPOSURE)
    frame_time_s = _as_float(raw_frame_time, DEFAULT_FRAME_TIME)
    offset = _as_float(raw_offset, DEFAULT_SHUTTER_OFFSET)
    max_trail = _as_float(raw_max_trail, DEFAULT_MAX_TRAIL_PX)

    errors: list[str] = []
    warnings: list[str] = []

    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    if not _is_number(raw_exposure) or exposure <= 0.0:
        errors.append(f"exposure 必须为正，实际 {raw_exposure}")
    if not _is_number(raw_frame_time) or frame_time_s <= 0.0:
        errors.append(f"frame_time 必须为正，实际 {raw_frame_time}")
    if not _is_number(raw_offset) or not (-1.0 - _EPS <= offset <= _EPS):
        errors.append(f"shutter_offset 应在 [-1, 0] 内，实际 {raw_offset}")
    if not _is_number(raw_max_trail) or max_trail <= 0.0:
        errors.append(f"max_trail_px 必须为正，实际 {raw_max_trail}")
    if exposure > frame_time_s + _EPS:
        warnings.append(
            f"曝光时间 {exposure} 超过帧间隔 {frame_time_s}，快门窗口越界"
        )

    for key in ("translation", "rotation_px", "vector"):
        value = config.get(key)
        if value is None:
            continue
        vec = _as_vector(value)
        if any(not math.isfinite(c) for c in vec):
            errors.append(f"{key} 含非有限分量: {vec}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "config": {
            "exposure": exposure,
            "frame_time": frame_time_s,
            "shutter_offset": offset,
            "max_trail_px": max_trail,
        },
    }


_AUDIT_CHECKS = [
    # (快照字段, 问题 id, 描述, 修复函数)
    ("shutter_decoupled", "coupled_shutter",
     "快门区间按帧时间估算，掉帧会导致拖影长度突变", "shutter_time / frame_drop"),
    ("motion_composed", "separate_camera_sampling",
     "相机旋转与平移分别采样，运动方向错误", "vector_compose / rotation_path"),
    ("history_separated", "shared_history",
     "模糊历史与抗锯齿历史共用缓冲，互相污染", "history_split"),
    ("skinned_motion_updated", "stale_skinned_vectors",
     "蒙皮后运动矢量未重算，仍使用蒙皮前旧矢量", "skin_motion"),
    ("results_aligned", "late_async_result",
     "异步后处理结果到达时目标帧已上屏，效果贴错帧", "result_align / async_post"),
    ("trail_capped", "unbounded_trail",
     "拖影长度没有上限，极端运动下出现超长拖影", "trail_cap"),
]


def audit(payload: Any = None) -> dict[str, Any]:
    """对管线快照做六类病理审计。

    快照中对应布尔标志缺省视为健康（True）；显式置 False 则记录问题并给出
    修复入口。另对快门越界等可数值判定的风险给出警告。
    """
    snapshot = payload if isinstance(payload, dict) else {}
    issues: list[dict[str, Any]] = []

    for flag, issue_id, description, fix in _AUDIT_CHECKS:
        if snapshot.get(flag, True) is False:
            issues.append({
                "id": issue_id,
                "severity": "error",
                "description": description,
                "fix": fix,
            })

    warnings: list[str] = []
    if "exposure" in snapshot and "frame_time" in snapshot:
        if _as_float(snapshot["exposure"]) > _as_float(snapshot["frame_time"]) + _EPS:
            warnings.append("曝光窗口越出帧区间，检查 shutter_offset/exposure 配置")
    if snapshot.get("max_trail_px") is not None and _as_float(snapshot["max_trail_px"]) <= 0:
        warnings.append("max_trail_px 非正，拖影上限失效")

    return {
        "healthy": not issues,
        "issues": issues,
        "issue_count": len(issues),
        "warnings": warnings,
    }


def fallback(payload: Any = None, *, reason: str | None = None) -> dict[str, Any]:
    """安全降级：返回一套保守可用的默认管线配置。"""
    if isinstance(payload, dict):
        cause = payload.get("reason", reason or "unspecified")
    else:
        cause = reason or ("unspecified" if payload is None else str(payload))

    return {
        "reason": cause,
        "config": {
            "exposure": DEFAULT_EXPOSURE,
            "frame_time": DEFAULT_FRAME_TIME,
            "shutter_offset": DEFAULT_SHUTTER_OFFSET,
            "max_trail_px": FALLBACK_MAX_TRAIL_PX,
            "clamp_to_frame": True,
        },
        "actions": [
            "重置为居中快门，曝光时间与帧时间解耦",
            "清空并分离 blur_history 与 aa_history",
            "拖影上限收紧到 %g 像素" % FALLBACK_MAX_TRAIL_PX,
            "暂停合入未完成对齐判定的异步后处理结果",
            "蒙皮矢量全部标记待重算，disocclusion 顶点矢量置零",
        ],
        "degraded": True,
    }


def metrics(payload: Any = None, *, frames: Sequence[dict] | None = None) -> dict[str, Any]:
    """从帧记录列表统计管线运行指标。

    每帧可携带：dropped_frames、trail_magnitude（配合 max_trail_px）、
    status（applied/discarded）、jitter、presented。
    """
    frame_list = list(_pick(payload, "frames", frames, []) or [])
    n = len(frame_list)

    total_dropped = 0
    over_limit = 0
    discarded = 0
    jitter_sq_sum = 0.0
    jitter_count = 0
    trail_sum = 0.0
    trail_count = 0

    for frame in frame_list:
        total_dropped += max(0, int(frame.get("dropped_frames", 0)))

        magnitude = frame.get("trail_magnitude")
        if magnitude is not None:
            magnitude = _as_float(magnitude)
            limit = _as_float(frame.get("max_trail_px", DEFAULT_MAX_TRAIL_PX))
            trail_sum += magnitude
            trail_count += 1
            if magnitude > limit + _EPS:
                over_limit += 1

        if frame.get("status") == "discarded" or frame.get("aligned") is False:
            discarded += 1

        jitter = frame.get("jitter")
        if jitter is not None:
            jv = _as_vector(jitter)
            jitter_sq_sum += sum(c * c for c in jv)
            jitter_count += 1

    produced = n + total_dropped
    return {
        "frame_count": n,
        "drop_rate": (total_dropped / produced) if produced else 0.0,
        "total_dropped_frames": total_dropped,
        "trail_over_limit_rate": (over_limit / n) if n else 0.0,
        "result_discard_rate": (discarded / n) if n else 0.0,
        "jitter_rms": math.sqrt(jitter_sq_sum / jitter_count) if jitter_count else 0.0,
        "mean_trail_px": (trail_sum / trail_count) if trail_count else 0.0,
    }


def _run_pipeline(snapshot: dict[str, Any]) -> dict[str, Any]:
    """确定性管线：同一份快照依次经过全部纯函数阶段。"""
    return {
        "shutter": shutter_time(snapshot),
        "rotation_path": rotation_path(snapshot),
        "vector": vector_compose(snapshot),
        "jitter": camera_jitter(snapshot),
        "history": history_split(snapshot),
        "skin": skin_motion(snapshot),
        "trail": trail_cap(snapshot),
        "align": result_align(snapshot),
    }


def determinism(payload: Any = None, *, runs: int | None = None) -> dict[str, Any]:
    """纯函数管线确定性验证：同一输入执行多次，结果必须逐字段相等。"""
    snapshot = payload if isinstance(payload, dict) else {}
    times = int(runs if runs is not None else snapshot.get("runs", 2))
    times = max(2, times)

    outputs = [_run_pipeline(snapshot) for _ in range(times)]
    baseline = outputs[0]
    mismatches: list[str] = []
    for run_index, output in enumerate(outputs[1:], start=1):
        if output != baseline:
            mismatches.append(f"run {run_index} 与 run 0 结果不一致")

    return {
        "deterministic": not mismatches,
        "runs": times,
        "mismatches": mismatches,
        "run_a": baseline,
        "run_b": outputs[-1],
    }
