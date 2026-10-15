# CAD 云端部署范围

## 两种运行模式

| 模式 | 启动入口 | 是否执行 CLIP 模型推理 |
| --- | --- | --- |
| 仓库根目录 Dockerfile | `demo_main.py` | 否，使用模拟检索结果 |
| 完整服务 | `main.py` | 是，需要训练权重、CLIP、PyTorch 与向量索引 |

根目录 Dockerfile 的轻量 Demo 可以演示界面与图库管理，不能作为完整检索服务已上线的证明。

## 完整服务所需材料

1. 可信的训练权重文件 `best_retrieval_clip.pth` 或其受控下载地址。该权重不在公开仓库中。
2. 可供检索的图库；仓库 `demo_gallery` 可作为小规模验证图库。
3. 能容纳模型、检查点加载与索引的运行内存。512 MB 免费实例不适合该模型服务。
4. 需要长期保留上传图片或索引时，配置持久存储。

不要将私有权重或访问凭据上传到公开 Git 仓库。明确模型来源、资源预算与托管权限后，再创建完整模型服务。

## 完整版部署入口

`Dockerfile.full` 使用原始 `main.py`，通过 Render 环境变量 `MODEL_DOWNLOAD_URL` 下载权重，也可挂载到 `CLIP_MODEL_PATH` 指定的位置。下载 URL 只在服务端设置。

- `MODEL_DOWNLOAD_URL`：受控的 HTTPS 权重下载地址。
- `MODEL_SHA256`：可选，指定预期 SHA-256 校验值。
- `CLIP_MODEL_PATH`：默认 `./models/best_retrieval_clip.pth`。
- `GALLERY_DIR`：默认 `./demo_gallery`。
- 服务监听平台注入的 `PORT`，健康检查路径 `/health`。

完整镜像和入口已准备；由于当前缺少可用于云端的权重和足够的已授权云资源，模型推理尚未进行云端验证。
