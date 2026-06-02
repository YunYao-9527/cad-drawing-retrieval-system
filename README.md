# CAD图纸智能检索系统

基于CLIP模型的工程图纸智能检索系统，使用向量数据库进行高效相似度搜索。

## 📋 项目概述

本项目是一个模块化的CAD图纸检索系统，采用微服务架构设计，支持：
- CLIP模型特征提取
- 向量数据库索引（Qdrant）
- RESTful API接口
- Web可视化界面
- 健康检查和监控
- Docker容器化部署

## 🏗️ 架构设计

### 模块结构

```
项目根目录/
├── config/              # 配置管理模块
│   ├── config.yaml      # 配置文件
│   └── config_manager.py
├── services/            # 服务模块
│   ├── feature_service.py    # 特征提取服务
│   └── retrieval_service.py  # 检索调度服务
├── database/            # 向量数据库模块
│   └── vector_db.py     # 向量数据库管理
|   └── qdrant_db.py
├── api/                 # API网关层
│   └── app.py           # FastAPI应用
├── monitoring/          # 日志与监控模块
│   ├── logger.py        # 结构化日志
│   └── metrics.py       # Prometheus指标
├── health/              # 健康检查模块
│   └── health_check.py  # 健康检查服务
├── deployment/          # 部署脚本
│   ├── Dockerfile
│   └── docker-compose.yml
├── templates/           # Web界面模板
├── main.py              # 主启动文件
└── requirements.txt     # Python依赖
```

### 核心模块说明

#### 1. 配置管理模块 (`config/`)
- **功能**: 集中式配置管理，支持YAML配置文件和环境变量覆盖
- **特性**:
  - 使用Pydantic进行配置验证
  - 支持环境变量动态覆盖
  - 类型安全的配置访问

#### 2. 特征服务模块 (`services/feature_service.py`)
- **功能**: CLIP模型加载与特征提取
- **特性**:
  - PyTorch + GPU支持
  - 异步队列处理
  - 批量特征提取
  - 模型信息查询

#### 3. 向量数据库模块 (`database/vector_db.py`)
- **功能**: 管理向量索引
- **特性**:
  - 当前支持Qdrant
  - 批量数据导入
  - 相似度搜索
  - 结果精炼
  - 启动时自动检查并初始化数据库

#### 4. 检索调度模块 (`services/retrieval_service.py`)
- **功能**: 执行检索、结果过滤与排序
- **特性**:
  - 相似度阈值过滤
  - 类别过滤
  - 结果排序
  - 高度相似结果精炼

#### 5. API网关层 (`api/app.py`)
- **功能**: 提供REST API、异常捕获、日志记录
- **特性**:
  - FastAPI框架
  - 自动API文档
  - 请求日志记录
  - 全局异常处理
  - CORS支持

#### 6. 日志与监控模块 (`monitoring/`)
- **功能**: 结构化日志与性能指标
- **特性**:
  - JSON/文本格式日志
  - 日志轮转
  - Prometheus指标导出
  - 性能指标收集

#### 7. 健康检查模块 (`health/health_check.py`)
- **功能**: 系统健康状态检查
- **特性**:
  - `/health` - 健康检查接口
  - `/metrics` - Prometheus指标接口
  - `/ready` - Kubernetes就绪检查
  - `/live` - Kubernetes存活检查

## 🚀 快速开始

### 环境要求

- Python 3.9+
- CUDA（可选，用于GPU加速）
- 8GB+ RAM
- 10GB+ 磁盘空间

### 安装步骤

1. **克隆项目并安装依赖**

```bash
pip install -r requirements.txt
```

2. **配置系统**

编辑 `config/config.yaml`，设置以下关键配置：

```yaml
model:
  finetuned_model_path: "D:\\workspace\\图纸检索项目\\数据集\\clip_model_deployment\\best_retrieval_clip.pth"

gallery:
  cad_drawing_dir: "D:\\workspace\\图纸检索项目\\数据集\\Dataset_img2\\test_processed"
```

或使用环境变量：

```bash
export MODEL_FINETUNED_MODEL_PATH="path/to/model.pth"
export GALLERY_CAD_DRAWING_DIR="path/to/gallery"
```

3. **启动服务**

```bash
python main.py
```

服务将在 `http://127.0.0.1:5000` 启动。
--------------------------------------------------------------
### Docker部署

1. **构建镜像**

```bash
cd deployment
docker-compose build
```

2. **启动服务**

```bash
docker-compose up -d
```

3. **查看日志**

```bash
docker-compose logs -f cad-retrieval
```

## 📖 API文档

启动服务后，访问以下地址查看API文档：

- **Swagger UI**: http://127.0.0.1:5000/docs
- **ReDoc**: http://127.0.0.1:5000/redoc
-------------------------------------------------------------
### 主要API端点

#### 检索接口

```http
POST /search
Content-Type: multipart/form-data

参数:
- image: 图片文件
- top_k: 返回结果数量（默认10）
```

#### 图片管理

```http
GET /api/images?page=1&per_page=20          # 获取图片列表
POST /api/images                              # 添加图片
DELETE /api/images/{image_id}                 # 删除图片
```

#### 类别管理

```http
GET /api/classes                              # 获取所有类别
GET /api/categories?page=1&per_page=20        # 获取类别列表（分页）
```

#### 数据库管理

```http
POST /api/initialize                          # 初始化数据库
POST /api/rebuild                             # 重建数据库
POST /api/cleanup                             # 清理重复记录
```

#### 健康检查

```http
GET /health?detailed=true                     # 健康检查
GET /metrics                                  # Prometheus指标
GET /ready                                    # 就绪检查
GET /live                                     # 存活检查
```

## 🔧 配置说明

### 配置文件结构

配置文件位于 `config/config.yaml`，主要配置项：

- **app**: 应用配置（主机、端口、调试模式）
- **model**: 模型配置（模型路径、设备、嵌入维度）
- **gallery**: 图库配置（图库目录路径）
- **vector_db**: 向量数据库配置（类型、持久化目录）
- **feature**: 特征提取配置（批大小、异步队列）
- **retrieval**: 检索配置（默认top_k、相似度阈值）
- **logging**: 日志配置（级别、格式、文件路径）
- **monitoring**: 监控配置（Prometheus端口）
- **health**: 健康检查配置（检查间隔）

### 环境变量覆盖

所有配置项都可以通过环境变量覆盖，命名规则：

- 使用下划线分隔，全大写
- 例如：`APP_HOST`, `MODEL_FINETUNED_MODEL_PATH`, `GALLERY_CAD_DRAWING_DIR`

## 📊 监控和日志

### 日志

日志文件默认保存在 `logs/app.log`，支持：
- JSON格式（结构化日志）
- 文本格式（可读性更好）
- 日志轮转（自动管理文件大小）

### Prometheus指标

访问 `http://127.0.0.1:5000/metrics` 查看Prometheus格式的指标，包括：

- HTTP请求统计
- 特征提取性能
- 检索性能
- 向量数据库操作统计
- 系统运行时间

### 健康检查

访问 `http://127.0.0.1:5000/health` 查看系统健康状态，包括：

- 特征服务状态
- 向量数据库状态
- 磁盘空间（详细模式）
- GPU状态（详细模式）
----------------------------------------------------------------
## 🐳 Docker部署

### 本地开发环境部署

1. **修改配置**

编辑 `deployment/docker-compose.yml`，修改卷挂载路径：

```yaml
volumes:
  - /path/to/your/gallery:/app/data/gallery:ro
  - /path/to/your/models:/app/models:ro
```

2. **启动服务**

```bash
cd deployment
docker-compose up -d
```

3. **查看服务状态**

```bash
docker-compose ps
```

4. **查看日志**

```bash
docker-compose logs -f cad-retrieval
```

### 可选：启用监控服务

```bash
docker-compose --profile monitoring up -d
```

这将启动：
- Prometheus（端口9091）
- Grafana（端口3000）

--------------------------------------------------------------

## 📦 交付方案

### 交付要求

本项目提供完整的Docker交付方案，满足以下要求：

- ✅ **Docker镜像**: 包含完整应用环境和依赖
- ✅ **docker-compose.yml**: 服务编排配置
- ✅ **健康检查**: 30秒内响应 `/health`
- ✅ **管理脚本**: `start.sh`、`stop.sh`、`rebuild_index.sh`
- ✅ **默认挂载路径**: `/app/logs/`、`/app/data/`、`/app/models/`

### 准备交付包

在Windows本地执行：

```powershell
# 运行交付准备脚本
cd D:\workspace\图纸检索项目\数据集\000_test
.\deployment\prepare_delivery.ps1
```

脚本会自动：
1. 构建Docker镜像
2. 保存镜像为tar文件
3. 准备所有交付文件（docker-compose.yml、管理脚本、文档）
4. 生成交付清单

### 交付文件清单

交付包包含以下文件：

```
delivery/
├── cad-retrieval-system.tar      # Docker镜像（必需）
├── docker-compose.yml             # Docker Compose配置（必需）
├── start.sh                       # 启动脚本（必需）
├── stop.sh                         # 停止脚本（必需）
├── rebuild_index.sh                # 重建索引脚本（必需）
├── README.md                       # 项目说明
├── DELIVERY_PACKAGE.md             # 交付说明
├── ACCEPTANCE_TEST.md              # 接收方测试指南
└── DEPLOYMENT_GUIDE.md            # 部署指南
```

### 交付方式

**方式一：文件传输**
- 将 `delivery/` 目录打包（zip或tar）
- 通过U盘、FTP、云存储等方式传输

**方式二：Docker镜像仓库**
- 推送到Docker Hub或私有仓库
- 接收方直接拉取镜像

### 接收方操作

接收方收到交付包后，按照 `ACCEPTANCE_TEST.md` 进行部署和测试：

1. **加载镜像**: `docker load -i cad-retrieval-system.tar`
2. **准备数据**: 创建目录并放置模型和图库文件
3. **启动服务**: `./start.sh`
4. **验证部署**: `curl http://localhost:5000/health`

详细步骤请参考 `deployment/ACCEPTANCE_TEST.md`。

---

## 🚀 Linux服务器Docker部署完整指南

本指南详细说明如何将CAD图纸检索系统部署到Linux服务器上。

### 前置要求

- **本地开发机**：Windows/Linux/Mac，已安装Docker和Docker Compose
- **Linux服务器**：Ubuntu 20.04+ / CentOS 7+ / Debian 10+，具有root或sudo权限
- **网络连接**：本地开发机可以访问Linux服务器（SSH）

### 第一步：本地准备（在开发机上操作）

#### 1.1 检查项目代码

确保项目代码完整，所有文件都在：

```bash
# 检查关键文件是否存在
ls -la deployment/Dockerfile
ls -la deployment/docker-compose.yml
ls -la config/config.yaml
ls -la requirements.txt
ls -la main.py
```

#### 1.2 准备模型文件

确保CLIP模型文件已准备好：

```bash
# 检查模型文件
ls -lh /path/to/your/model/best_retrieval_clip.pth
```

#### 1.3 准备图库目录

确保图库目录包含所有CAD图纸：

```bash
# 检查图库目录
ls -lh /path/to/your/gallery/
```

#### 1.4 修改docker-compose.yml配置

编辑 `deployment/docker-compose.yml`，将Windows路径改为Linux路径：

```yaml
volumes:
  # 修改为服务器上的实际路径
  - /data/cad/gallery:/app/data/gallery:ro
  - /data/cad/models:/app/models:ro
  - /data/cad/qdrant_db:/app/qdrant_db
  - /data/cad/logs:/app/logs
  - ./config:/app/config:ro
```

同时修改环境变量：

```yaml
environment:
  - MODEL_FINETUNED_MODEL_PATH=/app/models/best_retrieval_clip.pth
  - GALLERY_CAD_DRAWING_DIR=/app/data/gallery
  - APP_HOST=0.0.0.0
  - APP_PORT=5000
  - VECTOR_DB_TYPE=qdrant
```

#### 1.5 构建Docker镜像（可选）

**方式一：在本地构建镜像并导出**

```bash
# 进入项目根目录
cd /path/to/project

# 构建镜像
docker build -t cad-retrieval:latest -f deployment/Dockerfile .

# 导出镜像为tar文件
docker save cad-retrieval:latest -o cad-retrieval.tar

# 检查文件大小（可能很大，几GB）
ls -lh cad-retrieval.tar
```

**方式二：直接在服务器上构建（推荐）**

如果服务器可以直接访问代码仓库，可以在服务器上直接构建，跳过此步骤。

---

### 第二步：服务器准备（在Linux服务器上操作）

#### 2.1 安装Docker和Docker Compose

**Ubuntu/Debian系统：**

```bash
# 更新系统包
sudo apt-get update

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose插件
sudo apt-get install docker-compose-plugin

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 验证安装
docker --version
docker compose version
```

#### 2.2 配置Docker用户权限（可选）

如果不想每次都用sudo，将当前用户添加到docker组：

```bash
# 添加用户到docker组
sudo usermod -aG docker $USER

# 重新登录或执行
newgrp docker

# 验证（不需要sudo）
docker ps
```

#### 2.3 安装NVIDIA Container Toolkit（如果使用GPU）

```bash
# 添加NVIDIA仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启Docker
sudo systemctl restart docker

# 验证GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 2.4 创建项目目录结构

```bash
# 创建主目录
sudo mkdir -p /data/cad
cd /data/cad

# 创建子目录
sudo mkdir -p gallery      # 图库目录
sudo mkdir -p models       # 模型目录
sudo mkdir -p qdrant_db    # 向量数据库目录
sudo mkdir -p logs         # 日志目录
sudo mkdir -p deployment   # 部署配置目录

# 设置权限（根据实际情况调整）
sudo chown -R $USER:$USER /data/cad
chmod -R 755 /data/cad
```

---

### 第三步：传输文件到服务器

#### 3.1 传输项目代码

**方式一：使用SCP（推荐用于小项目）**

```bash
# 在本地开发机上执行
scp -r /path/to/project user@server-ip:/data/cad/

# 或者只传输必要文件
scp -r deployment/ user@server-ip:/data/cad/
scp -r config/ user@server-ip:/data/cad/
scp -r api/ user@server-ip:/data/cad/
scp -r services/ user@server-ip:/data/cad/
scp -r database/ user@server-ip:/data/cad/
scp -r monitoring/ user@server-ip:/data/cad/
scp -r health/ user@server-ip:/data/cad/
scp main.py requirements.txt user@server-ip:/data/cad/
```

**方式二：使用Git（推荐）**

```bash
# 在服务器上执行
cd /data/cad
git clone https://your-repo-url.git .
# 或者
git clone https://your-repo-url.git project
mv project/* .
rm -rf project
```

**方式三：使用rsync（推荐用于大文件）**

```bash
# 在本地开发机上执行
rsync -avz --progress /path/to/project/ user@server-ip:/data/cad/
```

#### 3.2 传输模型文件

```bash
# 在本地开发机上执行
scp /path/to/model/best_retrieval_clip.pth user@server-ip:/data/cad/models/

# 或者使用rsync（支持断点续传）
rsync -avz --progress /path/to/model/best_retrieval_clip.pth user@server-ip:/data/cad/models/
```

#### 3.3 传输图库文件

```bash
# 在本地开发机上执行（如果图库很大，建议使用rsync）
rsync -avz --progress /path/to/gallery/ user@server-ip:/data/cad/gallery/

# 或者打包后传输
cd /path/to/gallery
tar -czf gallery.tar.gz .
scp gallery.tar.gz user@server-ip:/data/cad/
# 在服务器上解压
ssh user@server-ip "cd /data/cad && tar -xzf gallery.tar.gz -C gallery && rm gallery.tar.gz"
```

#### 3.4 传输Docker镜像（如果已在本地构建）

```bash
# 在本地开发机上执行
scp cad-retrieval.tar user@server-ip:/tmp/

# 在服务器上导入镜像
ssh user@server-ip "docker load -i /tmp/cad-retrieval.tar && rm /tmp/cad-retrieval.tar"
```

---

### 第四步：服务器配置（在Linux服务器上操作）

#### 4.1 验证文件完整性

```bash
# 在服务器上检查
cd /data/cad

# 检查项目文件
ls -la
ls -la deployment/
ls -la config/

# 检查模型文件
ls -lh models/best_retrieval_clip.pth

# 检查图库文件
ls -lh gallery/ | head -20
```

#### 4.2 配置环境变量（可选）

创建 `.env` 文件：

```bash
cd /data/cad
cat > .env << EOF
# 应用配置
APP_HOST=0.0.0.0
APP_PORT=5000
APP_DEBUG=false

# 模型配置
MODEL_FINETUNED_MODEL_PATH=/app/models/best_retrieval_clip.pth
MODEL_DEVICE=auto

# 图库配置
GALLERY_CAD_DRAWING_DIR=/app/data/gallery

# 向量数据库配置
VECTOR_DB_TYPE=qdrant

# 日志配置
LOGGING_LEVEL=INFO
LOGGING_FORMAT=json
EOF
```

#### 4.3 修改docker-compose.yml（如果还没修改）

```bash
cd /data/cad/deployment
nano docker-compose.yml  # 或使用vim
```

确保volumes部分使用服务器路径：

```yaml
volumes:
  - /data/cad/gallery:/app/data/gallery:ro
  - /data/cad/models:/app/models:ro
  - /data/cad/qdrant_db:/app/qdrant_db
  - /data/cad/logs:/app/logs
  - /data/cad/config:/app/config:ro
```

#### 4.4 配置GPU支持（如果使用GPU）

编辑 `deployment/docker-compose.yml`，取消注释GPU配置：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

### 第五步：构建和启动服务（在Linux服务器上操作）

#### 5.1 构建Docker镜像

```bash
cd /data/cad

# 构建镜像（如果还没构建）
docker build -t cad-retrieval:latest -f deployment/Dockerfile .

# 查看镜像
docker images | grep cad-retrieval
```

**注意**：首次构建可能需要10-30分钟，取决于网络速度和服务器性能。

#### 5.2 启动服务

```bash
cd /data/cad/deployment

# 启动服务（后台运行）
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志（实时）
docker compose logs -f cad-retrieval
```

#### 5.3 等待服务就绪

首次启动会自动扫描图库并建立索引，这可能需要较长时间（取决于图库大小）：

```bash
# 持续查看日志，等待看到以下信息表示就绪：
# "✅ 数据库初始化完成"
# "🚀 服务启动成功，监听在 http://0.0.0.0:5000"
```

#### 5.4 验证服务运行

```bash
# 检查容器状态
docker compose ps

# 检查健康状态（30秒内应该响应）
curl http://localhost:5000/health

# 检查指标接口
curl http://localhost:5000/metrics

# 检查API文档
curl http://localhost:5000/docs
```

---

### 第六步：配置防火墙和网络（在Linux服务器上操作）

#### 6.1 配置防火墙

**Ubuntu/Debian（使用ufw）：**

```bash
# 允许5000端口（API服务）
sudo ufw allow 5000/tcp

# 允许9090端口（Prometheus指标，可选）
sudo ufw allow 9090/tcp

# 启用防火墙
sudo ufw enable

# 查看防火墙状态
sudo ufw status
```

#### 6.2 配置Nginx反向代理（可选，推荐用于生产环境）

```bash
# 安装Nginx
sudo apt-get install -y nginx  # Ubuntu/Debian
# 或
sudo yum install -y nginx      # CentOS/RHEL

# 创建配置文件
sudo nano /etc/nginx/sites-available/cad-retrieval
```

Nginx配置示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名或IP

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置：

```bash
# Ubuntu/Debian
sudo ln -s /etc/nginx/sites-available/cad-retrieval /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# CentOS/RHEL
sudo cp /etc/nginx/sites-available/cad-retrieval /etc/nginx/conf.d/
sudo nginx -t
sudo systemctl restart nginx
```

---

### 第七步：验证部署（在本地或服务器上操作）

#### 7.1 测试健康检查

```bash
# 在服务器上测试
curl http://localhost:5000/health

# 从外部测试（替换为服务器IP）
curl http://your-server-ip:5000/health
```

预期响应：

```json
{
  "status": "healthy",
  "timestamp": "2025-01-XX...",
  "uptime_seconds": 123.45,
  "version": "2.0.0",
  "checks": {
    "feature_service": {...},
    "vector_database": {...}
  }
}
```

#### 7.2 测试检索API

```bash
# 上传图片进行检索测试
curl -X POST http://your-server-ip:5000/search \
  -F "image=@/path/to/test-image.png" \
  -F "top_k=5"
```

#### 7.3 访问Web界面

在浏览器中访问：

```
http://your-server-ip:5000
```

或如果配置了Nginx：

```
http://your-domain.com
```

---

### 第八步：日常运维

#### 8.1 查看服务状态

```bash
cd /data/cad/deployment

# 查看容器状态
docker compose ps

# 查看资源使用
docker stats cad-retrieval-system

# 查看日志
docker compose logs -f cad-retrieval
```

#### 8.2 停止服务

```bash
cd /data/cad/deployment
docker compose stop
```

#### 8.3 重启服务

```bash
cd /data/cad/deployment
docker compose restart
```

#### 8.4 更新服务

```bash
cd /data/cad

# 拉取最新代码
git pull  # 或重新传输文件

# 重新构建镜像
docker build -t cad-retrieval:latest -f deployment/Dockerfile .

# 重启服务
cd deployment
docker compose down
docker compose up -d
```

#### 8.5 重建索引

```bash
# 方式一：使用API
curl -X POST http://localhost:5000/api/rebuild

# 方式二：使用脚本（在容器内）
docker compose exec cad-retrieval python -c "
from config.config_manager import init_config, get_config
from database.vector_db import init_vector_db, get_vector_db
init_config()
init_vector_db()
vector_db = get_vector_db()
count = vector_db.initialize_database()
print(f'重建完成，共处理 {count} 个图片')
"
```

#### 8.6 查看日志文件

```bash
# 查看应用日志
tail -f /data/cad/logs/app.log

# 查看Docker日志
docker compose logs -f --tail=100 cad-retrieval
```

#### 8.7 备份数据

```bash
# 备份向量数据库
tar -czf qdrant_db_backup_$(date +%Y%m%d).tar.gz /data/cad/qdrant_db

# 备份日志
tar -czf logs_backup_$(date +%Y%m%d).tar.gz /data/cad/logs
```

#### 8.8 清理资源

```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的容器
docker container prune

# 清理未使用的卷
docker volume prune
```

---

### 故障排查

#### 问题1：容器无法启动

```bash
# 查看详细错误日志
docker compose logs cad-retrieval

# 检查配置文件
docker compose config

# 检查端口占用
sudo netstat -tlnp | grep 5000
```

#### 问题2：健康检查失败

```bash
# 检查容器内部
docker compose exec cad-retrieval bash

# 在容器内测试
curl http://localhost:5000/health
python -c "from health.health_check import get_health_checker; print(get_health_checker().check_health())"
```

#### 问题3：GPU不可用

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 检查容器GPU访问
docker compose exec cad-retrieval nvidia-smi
```

#### 问题4：索引构建失败

```bash
# 检查图库目录权限
ls -la /data/cad/gallery

# 检查磁盘空间
df -h /data/cad

# 查看详细错误
docker compose logs cad-retrieval | grep -i error
```

---

### 性能优化建议

1. **GPU加速**：确保正确配置NVIDIA Container Toolkit
2. **内存优化**：根据图库大小调整Docker内存限制
3. **磁盘IO**：使用SSD存储向量数据库和图库
4. **网络优化**：使用Nginx反向代理和负载均衡
5. **监控告警**：配置Prometheus和Grafana监控

---

### 安全建议

1. **防火墙**：只开放必要端口（5000, 9090）
2. **HTTPS**：使用Nginx配置SSL证书
3. **访问控制**：配置Nginx基本认证或OAuth
4. **数据备份**：定期备份向量数据库和日志
5. **日志审计**：定期检查日志文件

-------------------------------------------------------------------

## 🛠️ 开发指南

### 项目结构

- **config/**: 配置管理，不依赖其他模块
- **monitoring/**: 日志和监控，不依赖业务模块
- **services/**: 业务服务，依赖config和monitoring
- **database/**: 数据访问，依赖config、monitoring和services
- **api/**: API层，依赖所有模块
- **health/**: 健康检查，依赖所有模块

### 添加新功能

1. **添加新API端点**

在 `api/app.py` 中添加路由：

```python
@app.get("/api/new-endpoint")
async def new_endpoint():
    # 实现逻辑
    pass
```

2. **添加新服务**

在 `services/` 目录下创建新服务文件，遵循单例模式：

```python
from services import get_service_name
```

3. **添加新配置**

在 `config/config.yaml` 中添加配置项，在 `config/config_manager.py` 中添加对应的Pydantic模型。

## 📝 常见问题

### Q: 如何启用GPU加速？

A: 确保安装了CUDA版本的PyTorch，系统会自动检测并使用GPU。也可以通过环境变量 `MODEL_DEVICE=cuda` 强制使用GPU。

### Q: 如何调整检索精度？

A: 修改 `config/config.yaml` 中的 `retrieval` 配置：
- `similarity_threshold`: 相似度阈值
- `enable_refinement`: 是否启用结果精炼

### Q: 如何查看详细日志？

A: 修改 `config/config.yaml` 中的 `logging.level` 为 `DEBUG`，或设置环境变量 `LOGGING_LEVEL=DEBUG`。

---

