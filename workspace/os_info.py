import platform

# 获得操作系统信息
try:
    distribution = platform.linux_distribution()
except:
    distribution = ("Not available", "")
machine = platform.machine()
version = platform.version()

print(f"Linux 发行版: {distribution[0]} {distribution[1]}")
print(f"指令集: {machine}")
print(f"内核版本: {version}")