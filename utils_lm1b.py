import importlib

import importlib

# 动态导入 gopen 模块
gopen_module = importlib.import_module("webdataset.gopen")
class CustomPipe(gopen_module.Pipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.timeout=72000.0
gopen_module.Pipe=CustomPipe

gopen_module = importlib.import_module("webdataset.gopen")


import webdataset.gopen as gopen


# 替换 Pipe 为 None（或者其他自定义实现）pr

# 确保在替换后再导入 webdataset
import webdataset

# 测试调用 gopen
try:
    webdataset.gopen('1')  # 应该报错，因为 Pipe 已被替换为 None
except Exception as e:
    print(f"Error as expected: {e}")
