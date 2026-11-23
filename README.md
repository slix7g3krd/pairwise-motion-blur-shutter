# pairwise-motion-blur-shutter

## 领域

游戏、图形与多媒体

## 主题

运动模糊快门

## 说明

快门解耦、矢量合成与结果帧对齐。

初始状态下 `graphics/motion_blur_shutter.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
