import importlib

import importlib

import jax.numpy

# 动态导入 gopen 模块
gopen_module = importlib.import_module("webdataset.gopen")
class CustomPipe(gopen_module.Pipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.timeout=72000.0
gopen_module.Pipe=None

# from subprocess import Popen
# x=Popen('echo 1').wait()
# print(x)
jax.numpy.ones((1,),).sharding.with_memory_kind('pinned_host')
while True:
    pass





import webdataset.gopen as gopen


# 替换 Pipe 为 None（或者其他自定义实现）pr

# 确保在替换后再导入 webdataset
import webdataset

# 测试调用 gopen
try:
    webdataset.gopen('pipe:')  # 应该报错，因为 Pipe 已被替换为 None
except Exception as e:
    print(f"Error as expected: {e}")
# webdataset.gopen('1')