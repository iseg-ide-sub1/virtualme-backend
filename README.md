# virtualme-backend

VirtualMe 插件后端项目，将插件收集到的动作序列数据进行预处理并用于训练，产出用户意图预测等数据。

## 模块清单

- 预处理
- 模式识别
- 日志总结
- 意图预测
- 能力分析

## 项目结构

### [/](./)

- [data_analyzer](./data_analyzer.py) 相对独立的程序，读取插件收集的数据并进行简单的统计分析

### [/dataset](./dataset)

收集的正式数据集

### [/log](./log)

收集的非正式数据，是 VirtualMe 插件的记录保存路径，其中的数据可以用于前期项目的测试

### [/base](./base)

待补充

### [/preprocess](./preprocess)

预处理

### [/intent_prediction](./intent_prediction)

意图预测

### [/log_summary](./log_summary)

日志总结

### [/pattern_recognition](./pattern_recognition)

模式识别
