# 📦 交付方案 - CAD图纸检索系统

## 📋 交付清单

### 1. Docker镜像
- **文件**: `cad-retrieval-system.tar` (Docker镜像压缩包)
- **说明**: 包含完整的应用环境和依赖

### 2. Docker Compose配置
- **文件**: `docker-compose.yml`
- **说明**: 服务编排配置，包含应用和可选监控服务

### 3. 管理脚本
- **文件**: `start.sh` - 启动服务
- **文件**: `stop.sh` - 停止服务
- **文件**: `rebuild_index.sh` - 重建索引

### 4. 配置文件（可选）
- **文件**: `config/config.yaml` - 应用配置
- **文件**: `.env.example` - 环境变量示例

### 5. 文档
- **文件**: `README.md` - 项目说明
- **文件**: `DEPLOYMENT_GUIDE.md` - 部署指南
- **文件**: `ACCEPTANCE_TEST.md` - 接收方测试指南

---

## 🚀 交付步骤

### 步骤1: 准备交付包

在Windows本地执行：

```powershell
# 1. 构建Docker镜像
cd D:\workspace\图纸检索项目\数据集\000_test
docker build -t cad-retrieval-system:latest -f deployment/Dockerfile .

# 2. 保存镜像为tar文件
docker save cad-retrieval-system:latest -o deployment/cad-retrieval-system.tar

# 3. 验证镜像大小（通常1-3GB）
Get-Item deployment/cad-retrieval-system.tar | Select-Object Name, @{Name="Size(GB)";Expression={[math]::Round($_.Length/1GB, 2)}}
```

### 步骤2: 准备交付文件

创建交付目录结构：

```
delivery/
├── cad-retrieval-system.tar      # Docker镜像
├── docker-compose.yml             # Docker Compose配置
├── start.sh                       # 启动脚本
├── stop.sh                        # 停止脚本
├── rebuild_index.sh               # 重建索引脚本
├── config/
│   └── config.yaml                # 配置文件（可选）
├── .env.example                   # 环境变量示例（可选）
├── README.md                      # 项目说明
├── DEPLOYMENT_GUIDE.md            # 部署指南
└── ACCEPTANCE_TEST.md             # 接收方测试指南
```

### 步骤3: 打包交付

```powershell
# 创建交付包
cd D:\workspace\图纸检索项目\数据集\000_test
Compress-Archive -Path deployment\cad-retrieval-system.tar,deployment\docker-compose.yml,start.sh,stop.sh,rebuild_index.sh,README.md,deployment\DEPLOYMENT_GUIDE.md -DestinationPath delivery-package.zip

# 或使用7-Zip创建分卷压缩（如果文件很大）
# 7z a -v2g delivery-package.7z delivery/
```

### 步骤4: 交付方式

**方式A: 直接传输**
- 通过U盘、移动硬盘等物理介质
- 通过FTP/SFTP上传到服务器
- 通过云存储（百度网盘、阿里云OSS等）分享

**方式B: Docker镜像仓库**
- 推送到Docker Hub或私有仓库
- 接收方直接拉取镜像

---

## 📝 交付说明文档

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+, CentOS 7+, Debian 10+)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **磁盘空间**: 至少10GB可用空间
- **内存**: 至少4GB RAM
- **CPU**: 2核以上（推荐4核+）
- **GPU**: 可选（支持NVIDIA GPU加速）

### 挂载路径说明

系统使用以下默认挂载路径：

- `/app/logs/` - 日志文件目录
- `/app/data/` - 数据目录（包含图库）
- `/app/models/` - 模型文件目录

### 端口说明

- **5000** - 应用主服务端口（API和Web界面）
- **9090** - Prometheus监控指标端口
- **9091** - Prometheus服务端口（可选）
- **3000** - Grafana可视化端口（可选）

### 健康检查

- **端点**: `http://localhost:5000/health`
- **响应时间**: 启动后30秒内响应
- **检查间隔**: 10秒
- **超时时间**: 5秒

---

## ✅ 交付检查清单

在交付前，请确认：

- [ ] Docker镜像已构建并保存为tar文件
- [ ] docker-compose.yml配置正确，挂载路径符合要求
- [ ] start.sh、stop.sh、rebuild_index.sh脚本已准备
- [ ] 所有脚本具有执行权限（chmod +x）
- [ ] 配置文件已准备（如需要）
- [ ] 文档已更新并包含接收方测试指南
- [ ] 已测试镜像可以正常启动
- [ ] 健康检查在30秒内响应

---

## 📞 技术支持

如有问题，请联系：
- 技术支持邮箱: [your-email@example.com]
- 技术支持电话: [your-phone]

