# NumPy CLIP ViT-B/32 实现报告

## 原项目检查

检查了 image_encoder.py、helpers.py、check.py、run.py、config.json，以及加载和预处理流程。六处 TODO 已分别实现：按行切分并按 CHW 展平、拆分注意力头、缩放点积注意力和输出投影、两条残差路径和 MLP、Patch 投影与 CLS/位置嵌入、CLS 归一化与投影。基线全部通过，无需修改模型。git status 返回「not a git repository」；原始源码和基线推理数组已保存于 outputs/original/，项目原有文件未修改或删除。

## 新增文件与原理

- visualization/common.py：调用原 PreprocessImage，截取其实际使用的 RGB 裁剪图供显示，再调用原 to_tensor；使用已有 Pillow 输出 PNG。
- visualization/visualize_patches.py：并排展示原图和编号 0–48 的 7×7 裁剪网格。逐个将归一化张量的 32×32 切片按 CHW 展平，与原 patchify 完全比较；可单独输出指定 Patch。
- visualization/visualize_attention.py：复用前 11 层和最后一层的归一化、QKV 权重、拆头方法及 softmax helper，单独计算第 12 层的注意力系数。选 CLS 查询及 49 个 Patch 键，平均全部 12 个头后按行恢复为 7×7。按预处理的缩放与中心裁剪坐标将热力图映射回原图，白框表示模型覆盖范围。颜色以该图最大 Patch 权重缩放，不能直接跨图比较；不对去掉 CLS 键后的 49 个权重再归一化。该图只描述注意力分布，不能作为模型输出的完整因果解释。
- outputs/：原始备份、两轮检查日志、推理结果与示例图片。

在项目根目录运行：

    python visualization/visualize_patches.py assets/images/image_01.jpg --patch 17
    python visualization/visualize_attention.py assets/images/image_01.jpg

## 验证

新增可视化前后使用同一全局环境运行了 python check.py --stage all 和 python check.py --stage batch，日志逐字节一致。下表改为项目 .venv（Python 3.14.0、NumPy 2.5.4）的最新检查结果，误差相对于课程参考输出。

| 阶段 | 修改前 | 修改后 | 最大绝对误差 | 平均绝对误差 |
|---|---|---|---:|---:|
| assets | PASS | PASS | 不适用 | 不适用 |
| preprocess | PASS | PASS | 0 | 0 |
| patches | PASS | PASS | 0 | 0 |
| tokens | PASS | PASS | 4.005e-05 | 1.289e-06 |
| block0 | PASS | PASS | 7.063e-06 | 3.250e-07 |
| last_tokens | PASS | PASS | 4.768e-05 | 2.696e-06 |
| output | PASS | PASS | 5.439e-07 | 8.311e-08 |
| batch（all 内及单独执行） | PASS | PASS | 4.619e-07 | 7.632e-08 |

两次 run.py 对十张样图生成的 ID 与 10×512 输出逐元素相同，最大和平均绝对差均为 0；备份源码也与当前源码逐字节一致。Patch 脚本验证了全部 49 个切片并导出了 Patch 17；Attention 脚本确认形状为 1×12×50×50、系数有限且非负、每个查询行和为 1，并生成了 7×7 叠加图。

## 尚未解决的问题

未发现影响交付的问题。由于项目不在 Git 仓库中，无法提供 Git 差异；可用原始备份和前后日志核对。独立可视化中的最后一层 Q/K 计算和叠加坐标映射如遇未来模型或预处理改动，需同步检查。
