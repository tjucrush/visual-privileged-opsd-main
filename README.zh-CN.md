<div align="center">

# 截图坐标工具箱

### 用于训练截图坐标预测模型的工具

**数据准备 | 模型训练 | 效果评估**

<p><a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a></p>

<img src="docs/assets/method-overview.png" alt="方法概览图" width="860">

</div>

本项目提供数据处理脚本、配置和测试，用于准备界面截图数据、训练坐标预测模型，并评估模型输出。

## 项目内容

| 模块 | 入口 | 用途 |
|---|---|---|
| 数据 | `vpopsd.prepare` | 转换数据记录并生成可复现的数据划分 |
| 训练 | `vpopsd.train` | 使用 LoRA 或全量参数微调 |
| 评估 | `vpopsd.evaluate`、`vpopsd.suite` | 评估预测结果并运行基准测试套件 |
| 配置 | `configs/` | 训练、推理和基准测试配置 |

## 快速开始

生成示例数据，然后评估坐标预测结果：

```bash
python -m vpopsd.demo
python -m vpopsd.evaluate --data outputs/demo/examples.jsonl --predictions outputs/demo/predictions.jsonl --output-dir outputs/demo/eval
```

## 安装

训练环境使用 Python 3.12 和 CUDA GPU。请根据 CUDA 版本，按照 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)安装匹配的 PyTorch 与 torchvision。

```bash
git clone https://github.com/tjucrush/visual-privileged-opsd-main.git
cd visual-privileged-opsd-main
python -m venv .venv
source .venv/bin/activate
# 安装 PyTorch 2.6–2.8 及适配当前 CUDA 版本的 torchvision。
pip install -e '.[dev,data]'
```

Windows 用户可使用 `.venv\Scripts\Activate.ps1` 激活环境。项目使用 Transformers 4.57.1；训练以单进程、单设备运行。

## 数据准备

每条 JSONL 记录包含指令、截图和边界框。坐标采用 **[0,1] 范围内归一化的 xyxy 格式**。相对图片路径以 JSONL 文件所在目录为基准。

```json
{"id":"sample-1","image":"images/1.png","instruction":"Click Save","bbox":[0.65,0.62,0.825,0.73],"benchmark":"scalecua"}
```

从 [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA) 获取界面定位数据，并将其字段和坐标格式转换为项目使用的标准格式：

```bash
python -m vpopsd.prepare convert --input data/raw/grounding.jsonl --image-root data/raw/images --image-field image --instruction-field instruction --bbox-field bbox --bbox-format xyxy --bbox-units pixels --output data/scalecua/all.jsonl
python -m vpopsd.prepare split --input data/scalecua/all.jsonl --train-size 7000 --val-size 300 --seed 42 --output-dir data/scalecua
```

请按源数据调整字段名和边界框单位。嵌套字段使用点号表示。划分时会按解码后的截图内容分组，以确保训练集和验证集截图互不重叠。可使用 `--exclude path/to/benchmark.jsonl ...` 排除评估截图。划分元数据会记录随机种子和源文件 SHA256。

## 训练

```bash
python -m vpopsd.train --config configs/qwen3_vl_8b.yaml
```

默认配置使用 LoRA rank 16、AdamW、一个训练周期和 8 步梯度累积。设置 `lora: false` 可进行全量参数微调。每次运行请使用单独的 `output_dir`。

输出包括配置与软件包版本、数据集元信息、Token 有效性、反向 KL、教师熵、Token 权重、训练时长及模型权重。训练时长涵盖生成轨迹、教师与学生前向计算、反向传播和优化器更新。

## 评估

```bash
python -m vpopsd.evaluate --data data/benchmarks/screenspot/test.jsonl --config configs/eval_base.yaml --output-dir outputs/eval-base
python -m vpopsd.suite --suite configs/benchmarks.yaml --config configs/eval_base.yaml --output-dir outputs/suite-base
```

评估 LoRA 检查点时，在推理配置中设置 `adapter: outputs/vpopsd-qwen3-vl-8b/model`。评估全量微调检查点时，将 `model_name` 指向已保存的模型目录。

测试套件覆盖 ScreenSpot、ScreenSpot-v2、ScreenSpot-Pro、GroundUI-1K、OSWorld-G 边界框任务和 UI-Vision 元素定位。适配器和任务定义见[基准测试准备说明](docs/benchmarks.md)。

评估使用原始截图和点落入边界框的准确率。报告包含各数据集准确率、等权宏平均准确率和按样本加权的微平均准确率。缺失或格式错误的预测按失败计。运行结果保存在 `metrics.json`、`predictions.jsonl` 和 `details.jsonl`。

## 消融配置

| 选项 | 配置 |
|---|---|
| 关闭数字位权重 | `position_weighting: false` |
| 关闭熵权重 | `entropy_weighting: false` |
| 关闭红色框选 | `privilege.red_box: false` |
| 关闭高斯蒙版 | `privilege.gaussian_mask: false` |
| 关闭教师辅助输入 | 将红框、高斯蒙版和 `teacher_hint` 设为 `false` |

## 测试

```bash
pytest -q
ruff check .
```

24 项测试覆盖反向 KL、教师梯度隔离、Token 对齐、坐标约定、按截图隔离的数据划分、评估指标，以及包含图像输入、生成、LoRA 和反向传播的轻量 Qwen3-VL 集成测试。已测试环境见[验证说明](docs/validation.md)。

## 项目结构

```text
src/vpopsd/     训练、评估、模型目标函数和数据集工具
configs/       训练、推理和基准测试套件配置
tests/         目标函数、对齐、数据和模型集成测试
docs/          实现说明、基准测试准备和验证说明
```

## 参考资源

- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Transformers Qwen3-VL API](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA)

## 许可

代码采用 [MIT License](LICENSE)。模型和数据集遵循各自的上游许可。

