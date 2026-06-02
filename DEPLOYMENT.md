# 工程图纸检索系统部署说明

本仓库只建议上传代码、配置模板、前端页面和部署脚本，不建议把完整数据集、模型权重、本地 Qdrant 向量库、实验缓存和日志直接提交到 GitHub。

## 本地运行

```powershell
cd D:\workspace\图纸检索项目\数据集\111_test\0version_1
$env:MODEL_DEVICE="cuda"
py -3.10 main.py
```

启动后访问：

```text
http://127.0.0.1:5000/
```

健康检查：

```text
http://127.0.0.1:5000/health
```

## 服务器部署

部署时需要额外挂载或上传以下资源：

- 模型权重：`best_retrieval_clip.pth`
- 图库数据：清理后的 `Dataset_img3`
- 向量库：已建好的 `qdrant_db`，或在服务器上重新执行 `build_index.py`

Docker 方式可以参考：

```bash
cd deployment
docker build -t cad-retrieval-system:latest -f Dockerfile ..
docker compose up -d
```

如果在云平台部署，需要把 `APP_HOST` 设置为 `0.0.0.0`，并把服务端口设置为平台分配的端口或 `5000`。

## GitHub Pages 限制

GitHub Pages 只能托管静态网页，不能运行 FastAPI、CLIP 模型、Qdrant 向量检索和 OCR 服务。因此本项目如果要得到可检索的网址，需要部署后端服务，不能只依赖 GitHub Pages。

## 临时演示网址

如果只是答辩或临时演示，可以在本地服务启动后使用 Cloudflare Tunnel、ngrok 等工具生成临时公网地址。临时地址依赖本机进程和网络，关闭终端或电脑后地址会失效。
