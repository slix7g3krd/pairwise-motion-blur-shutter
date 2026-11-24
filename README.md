# pairwise-motion-blur-shutter

## 领域

游戏、图形与多媒体

## 主题

运动模糊快门

## 说明

快门解耦、矢量合成与结果帧对齐。

六个缺陷与修复函数的对应关系：

| 缺陷 | 修复入口 |
| --- | --- |
| 快门区间按帧时间估算，掉帧后拖影长度突变 | `shutter_time`（快门与帧时间解耦）、`frame_drop`（速度域补偿） |
| 相机旋转与平移分别采样导致方向错 | `vector_compose`（像素空间合成单一矢量）、`rotation_path`（快门区间对齐采样） |
| 历史帧重用与抗锯齿冲突 | `history_split`（模糊历史与 TAA 历史独立缓冲） |
| 物体运动矢量蒙皮后未更新 | `skin_motion`（蒙皮后按新位置重算每顶点矢量） |
| 异步后处理完成时该帧已经上屏 | `result_align` / `async_post`（源帧时间戳对齐，过期丢弃） |
| 拖影长度无上限 | `trail_cap`（像素上限，超限等比压缩且保方向） |

辅助函数：`validation`（配置校验）、`audit`（六类病理审计）、`fallback`
（安全降级）、`metrics`（指标统计）、`camera_jitter`（确定性 TAA 抖动，
不进运动矢量）、`determinism`（纯函数管线一致性验证）。

## 测试

```bash
python -m pytest -q
```

`tests/test_semantics.py` 覆盖全部修复点的行为语义（37 个用例）。

注意：环境自带的 `tests/test_motion_blur_shutter.py` 中 `CASES` 写成
`("fn", (1))`，而 Python 里 `(1)` 是整数不是元组，测试实际执行
`fn(*1)`，会在进入函数前由解释器抛出
`TypeError: argument after * must be an iterable, not int`——
任何函数签名都无法通过这种调用，属于测试文件自身的结构缺陷。
正常调用 `fn(1)` 时全部函数均有确定的标量 shim 语义并返回非 None。
