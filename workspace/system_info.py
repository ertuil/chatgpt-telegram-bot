import os
import platform

# 获取Linux发行版信息
try:
    with open('/etc/os-release', 'r') as f:
        os_info = f.read()
    print('Linux 发行版:\n', os_info)
except FileNotFoundError:
    print('Linux 发行版: 未找到/etc/os-release文件')

# 获取指令集
machine = platform.machine()
print('\u6307\u4ee4\u96c6:', machine)

# 获\u53d6Python\u7248\u672c\u4fe1\u606f
python_version = platform.python_version()
print('Python \u7248\u672c:', python_version)