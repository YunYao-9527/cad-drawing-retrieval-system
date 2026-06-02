# 📦 部署文档索引

本目录包含所有部署相关的文档和脚本。

## 📚 文档列表

### 1. [QUICK_START.md](QUICK_START.md) - 快速开始
- 适合有SSH访问权限的情况
- 5分钟快速部署指南
- 最常用的部署方式

### 2. [DEPLOY_WITHOUT_SSH.md](DEPLOY_WITHOUT_SSH.md) - 无SSH部署指南 ⭐
- **适合无法使用SSH的情况**
- 提供6种替代方案：
  - Web控制台部署
  - FTP/SFTP客户端部署
  - Git仓库部署
  - Docker镜像仓库部署
  - 分步骤手动部署
  - 云服务器文件管理部署

### 3. [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - 完整部署指南
- 详细的部署步骤
- 包含故障排查
- 日常运维指南

### 4. [server_commands.txt](server_commands.txt) - 服务器命令脚本
- 可直接复制到服务器执行的完整脚本
- 适合Web控制台使用
- 包含所有部署步骤

## 🛠️ 脚本列表

### 1. [prepare_delivery.ps1](prepare_delivery.ps1) - 交付准备脚本
- 在Windows本地运行
- 构建Docker镜像
- 准备所有交付文件

## 🚀 快速选择

### 我有SSH访问权限
→ 使用 [QUICK_START.md](QUICK_START.md)

### 我没有SSH访问权限
→ 使用 [DEPLOY_WITHOUT_SSH.md](DEPLOY_WITHOUT_SSH.md)

### 我需要详细说明
→ 使用 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### 我只有Web控制台
→ 使用 [server_commands.txt](server_commands.txt)，复制内容到Web控制台执行

## 📋 部署前检查清单

- [ ] 确认服务器操作系统（Ubuntu/CentOS/Debian）
- [ ] 确认服务器有root或sudo权限
- [ ] 确认服务器有至少10GB可用空间
- [ ] 准备模型文件（`best_retrieval_clip.pth`）
- [ ] 准备图库文件（CAD图纸图片）
- [ ] 确认服务器访问方式（SSH/Web控制台/FTP等）

## 🆘 需要帮助？

如果遇到问题：
1. 查看对应的部署文档
2. 检查故障排查章节
3. 查看日志文件
4. 联系技术支持

