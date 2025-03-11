# MAR代码阅读

# 1. 训练过程
## 1.1. 启动
从main_mar.py_277调用engine_mar.py::train_one_epoch：
数据集格式为：
```python
# samples: batch_size, 3, 256, 256
# labels: (batch_size,) 类别编号
```
先进行vae.encode