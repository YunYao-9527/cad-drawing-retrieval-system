# ✅ 接收方测试指南 - CAD图纸检索系统

本文档说明如何在Linux服务器上接收、部署和测试CAD图纸检索系统。

---

## 📦 第一步：接收交付文件

### 1.1 接收文件清单

确认已收到以下文件：

- [ ] `cad-retrieval-system.tar` - Docker镜像文件
- [ ] `docker-compose.yml` - Docker Compose配置
- [ ] `start.sh` - 启动脚本
- [ ] `stop.sh` - 停止脚本
- [ ] `rebuild_index.sh` - 重建索引脚本
- [ ] `README.md` - 项目说明文档
- [ ] `ACCEPTANCE_TEST.md` - 本测试指南

### 1.2 准备测试环境

```bash
# 1. 检查Docker是否安装
docker --version
docker compose version

# 如果没有安装，执行：
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install docker-compose-plugin  # Ubuntu/Debian
# 或
sudo yum install docker-compose-plugin      # CentOS/RHEL

# 2. 创建项目目录
sudo mkdir -p /data/cad-retrieval
cd /data/cad-retrieval

# 3. 上传交付文件到此目录
# 使用FTP、SCP或其他方式上传所有交付文件
```

---

## 🚀 第二步：加载Docker镜像

```bash
# 进入项目目录
cd /data/cad-retrieval

# 加载Docker镜像
docker load -i cad-retrieval-system.tar

# 验证镜像已加载
docker images | grep cad-retrieval-system
```

**预期输出**：
```
REPOSITORY              TAG       IMAGE ID       CREATED         SIZE
cad-retrieval-system    latest    xxxxxxxxxxxx   2 hours ago     2.5GB
```

---

## 📁 第三步：准备数据目录

```bash
# 创建必要的目录结构
sudo mkdir -p /data/cad-retrieval/{logs,data/gallery,models,qdrant_db}
sudo chown -R $USER:$USER /data/cad-retrieval

# 准备模型文件（如果有）
# 将模型文件 best_retrieval_clip.pth 放到 /data/cad-retrieval/models/

# 准备图库文件（如果有）
# 将CAD图纸图片放到 /data/cad-retrieval/data/gallery/
```

**目录结构**：
```
/data/cad-retrieval/
├── cad-retrieval-system.tar
├── docker-compose.yml
├── start.sh
├── stop.sh
├── rebuild_index.sh
├── logs/              # 日志目录（自动创建）
├── data/
│   └── gallery/       # 图库目录（需要放置图片）
├── models/            # 模型目录（需要放置模型文件）
└── qdrant_db/         # 向量数据库目录（自动创建）
```

---

## ⚙️ 第四步：配置docker-compose.yml

检查并修改 `docker-compose.yml` 中的挂载路径：

```bash
# 编辑docker-compose.yml
nano docker-compose.yml
# 或
vi docker-compose.yml
```

**确保挂载路径正确**：
```yaml
volumes:
  - /data/cad-retrieval/data:/app/data:ro      # 图库目录
  - /data/cad-retrieval/models:/app/models:ro  # 模型目录
  - /data/cad-retrieval/qdrant_db:/app/qdrant_db  # 向量数据库
  - /data/cad-retrieval/logs:/app/logs        # 日志目录
```

**确保环境变量正确**：
```yaml
environment:
  - MODEL_FINETUNED_MODEL_PATH=/app/models/best_retrieval_clip.pth
  - GALLERY_CAD_DRAWING_DIR=/app/data/gallery
  - APP_HOST=0.0.0.0
  - VECTOR_DB_TYPE=qdrant
```

---

## 🎬 第五步：启动服务

### 5.1 使用启动脚本

```bash
# 给脚本添加执行权限
chmod +x start.sh stop.sh rebuild_index.sh

# 启动服务
./start.sh
```

### 5.2 或使用Docker Compose

```bash
# 启动服务
docker compose up -d

# 查看启动日志
docker compose logs -f cad-retrieval
```

**预期输出**：
```
[+] Running 2/2
 ✔ Network cad-retrieval-network    Created
 ✔ Container cad-retrieval-system   Started
```

---

## ✅ 第六步：验证部署

### 6.1 检查容器状态

```bash
# 查看容器状态
docker compose ps

# 预期输出：
# NAME                    STATUS          PORTS
# cad-retrieval-system    Up 10 seconds   0.0.0.0:5000->5000/tcp
```

### 6.2 检查健康状态（30秒内响应）

```bash
# 等待30秒后检查健康状态
sleep 30
curl http://localhost:5000/health

# 预期输出：
# {"status":"healthy","timestamp":"2025-01-XX...","version":"2.0.0"}
```

**测试脚本**：
```bash
#!/bin/bash
echo "等待服务启动..."
for i in {1..30}; do
    response=$(curl -s http://localhost:5000/health 2>/dev/null)
    if [ $? -eq 0 ] && echo "$response" | grep -q "healthy"; then
        echo "✅ 健康检查通过（耗时 ${i} 秒）"
        echo "响应: $response"
        exit 0
    fi
    sleep 1
done
echo "❌ 健康检查超时（30秒）"
exit 1
```

### 6.3 检查API端点

```bash
# 检查API文档
curl http://localhost:5000/docs

# 检查API列表
curl http://localhost:5000/api

# 检查数据库状态
curl http://localhost:5000/api/status
```

### 6.4 检查监控指标

```bash
# 检查Prometheus指标
curl http://localhost:5000/metrics

# 预期输出包含：
# http_requests_total
# http_request_duration_seconds
# system_uptime_seconds
# cpu_utilization_percent
# gpu_utilization_percent
```

---

## 🧪 第七步：功能测试

### 7.1 测试图片检索

```bash
# 准备测试图片（使用图库中的任意一张图片）
TEST_IMAGE="/data/cad-retrieval/data/gallery/test_image.png"

# 发送检索请求
curl -X POST "http://localhost:5000/search" \
  -F "image=@$TEST_IMAGE" \
  -F "top_k=10" \
  -o search_result.json

# 查看结果
cat search_result.json | python -m json.tool
```

**预期输出**：
```json
{
  "status": "success",
  "results": [
    {
      "rank": 1,
      "id": "...",
      "filename": "...",
      "similarity": 0.95,
      "distance": 0.05
    },
    ...
  ],
  "query_image": "test_image.png"
}
```

### 7.2 测试Web界面

在浏览器中访问：
```
http://your-server-ip:5000
```

应该能看到：
- 图片上传界面
- 检索结果展示
- API文档链接

### 7.3 测试管理脚本

```bash
# 测试停止脚本
./stop.sh

# 检查容器已停止
docker compose ps

# 测试启动脚本
./start.sh

# 检查容器已启动
docker compose ps
```

---

## 🔄 第八步：测试重建索引

```bash
# 停止服务
./stop.sh

# 删除现有索引（可选，用于测试）
rm -rf qdrant_db/*

# 重建索引
./rebuild_index.sh

# 预期输出：
# ✅ 索引重建完成，共处理 XXX 个图片

# 重新启动服务
./start.sh
```

---

## 📊 第九步：性能测试

### 9.1 检查响应时间

```bash
# 测试健康检查响应时间
time curl -s http://localhost:5000/health

# 预期：响应时间 < 1秒
```

### 9.2 测试并发请求

```bash
# 安装Apache Bench（如果未安装）
sudo apt-get install apache2-utils  # Ubuntu/Debian
sudo yum install httpd-tools        # CentOS/RHEL

# 并发测试（10个并发，100个请求）
ab -n 100 -c 10 http://localhost:5000/health

# 检查QPS（每秒请求数）
curl http://localhost:5000/metrics | grep http_requests_qps
```

---

## 🐛 第十步：故障排查

### 10.1 查看日志

```bash
# 查看容器日志
docker compose logs cad-retrieval

# 查看应用日志
tail -f /data/cad-retrieval/logs/app.log

# 查看错误日志
grep ERROR /data/cad-retrieval/logs/app.log
```

### 10.2 常见问题

**问题1: 容器无法启动**
```bash
# 检查镜像是否正确加载
docker images | grep cad-retrieval

# 检查端口是否被占用
netstat -tulpn | grep 5000

# 检查磁盘空间
df -h
```

**问题2: 健康检查失败**
```bash
# 检查容器内部
docker compose exec cad-retrieval bash

# 在容器内检查
curl http://localhost:5000/health
ps aux | grep python
```

**问题3: 检索返回空结果**
```bash
# 检查数据库是否初始化
curl http://localhost:5000/api/status

# 检查图库目录是否有文件
ls -la /data/cad-retrieval/data/gallery/

# 检查日志中的初始化信息
grep "初始化" /data/cad-retrieval/logs/app.log
```

---

## ✅ 测试验收标准

### 必须满足的要求

- [x] Docker镜像成功加载
- [x] 容器在30秒内启动并响应健康检查
- [x] `/health` 端点返回 `{"status":"healthy"}`
- [x] API文档可访问（`/docs`）
- [x] 图片检索功能正常
- [x] Web界面可访问
- [x] 监控指标正常（`/metrics`）
- [x] 日志文件正常生成
- [x] `start.sh`、`stop.sh`、`rebuild_index.sh` 脚本正常工作

### 性能要求

- [x] 健康检查响应时间 < 1秒
- [x] 检索响应时间 < 5秒（单张图片）
- [x] 系统稳定运行24小时无崩溃

---

## 📝 测试报告模板

```
测试日期: ___________
测试人员: ___________
服务器信息:
  - IP地址: ___________
  - 操作系统: ___________
  - Docker版本: ___________

测试结果:
  [ ] Docker镜像加载成功
  [ ] 容器启动成功（30秒内）
  [ ] 健康检查通过
  [ ] API功能正常
  [ ] 图片检索功能正常
  [ ] Web界面正常
  [ ] 监控指标正常
  [ ] 管理脚本正常

问题记录:
  _________________________________
  _________________________________

验收意见:
  [ ] 通过
  [ ] 不通过（原因：__________）
```

---

## 📞 技术支持

如遇到问题，请提供以下信息：
1. 错误日志（`docker compose logs cad-retrieval`）
2. 系统信息（`uname -a`、`docker --version`）
3. 配置文件内容（`docker-compose.yml`）
4. 测试步骤和结果

联系方式：
- 技术支持邮箱: [your-email@example.com]
- 技术支持电话: [your-phone]

