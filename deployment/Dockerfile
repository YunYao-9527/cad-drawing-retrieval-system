# CAD图纸检索系统 Dockerfile
# 支持GPU和CPU模式

# 使用稳定的Python 3.9基础镜像（使用Debian Bookworm稳定版）
FROM python:3.9-slim-bookworm

# 设置工作目录
WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
# 注意：git必须在pip install之前安装，因为requirements.txt中可能包含git+https://的依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git --version

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖（优先使用国内镜像源，失败则使用默认源）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ || \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录（符合交付要求的挂载路径）
RUN mkdir -p /app/logs \
    /app/data/gallery \
    /app/models \
    /app/qdrant_db

# 暴露端口
EXPOSE 5000 9090

# 健康检查 - 30秒内响应（符合交付要求）
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()" || exit 1

# 启动命令
CMD ["python", "main.py"]
