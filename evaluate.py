import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import clip
import numpy as np
import faiss
from collections import defaultdict


# 配置参数（仅保留评估所需参数）
class Config:
    data_root = r"D:\PCproject\ShuiLi\数据集\Dataset_img2"  # 数据集根目录
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = "ViT-B/32"  # 必须与训练时使用的模型一致
    embedding_dim = 512
    num_workers = 4
    class_level = 0  # 必须与训练时保持一致


config = Config()


# 检索优化的CLIP Adapter模型（与训练时保持一致）
class RetrievalCLIPAdapter(nn.Module):
    def __init__(self, clip_model, num_classes=None):
        super().__init__()
        self.clip_model = clip_model.float()
        self.feature_dim = clip_model.visual.output_dim

        # 检索专用Adapter（与训练时保持一致）
        self.adapter = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.embedding_dim)
        )

        if num_classes:
            self.classifier = nn.Linear(config.embedding_dim, num_classes)
        else:
            self.classifier = None

        # 评估时冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, return_features=False):
        x = x.float()

        with torch.no_grad():
            base_features = self.clip_model.encode_image(x)

        retrieval_features = F.normalize(self.adapter(base_features), p=2, dim=1)

        if return_features:
            return retrieval_features
        elif self.classifier:
            return self.classifier(retrieval_features)
        else:
            return retrieval_features


# 数据加载（仅加载测试集）
def get_test_dataloader(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return test_loader, test_dataset.classes


# 完整的检索评估函数
def comprehensive_retrieval_evaluation(model, test_loader, save_suffix="evaluated", class_names=None):
    """对已训练模型进行完整的检索评估"""
    model.eval()
    all_features = []
    all_labels = []  # 保存真实标签

    print("提取测试集特征...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="特征提取"):
            features = model(images.to(config.device), return_features=True)
            all_features.append(features.cpu())
            all_labels.extend(labels.tolist())

    features = torch.cat(all_features).numpy()
    print(f"特征形状: {features.shape}")

    # 保存特征（可选）
    np.save(f"features_{save_suffix}.npy", features)
    np.save(f"labels_{save_suffix}.npy", all_labels)

    # 建立FAISS索引
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    faiss.write_index(index, f"cad_index_{save_suffix}.faiss")

    # 正确的标签获取函数
    def get_label(idx):
        return all_labels[idx]

    # 构建标签到索引的映射
    label_to_indices = defaultdict(list)
    for i, label in enumerate(all_labels):
        label_to_indices[label].append(i)

    if label_to_indices:
        # 计算k值
        max_num_relevant = max(len(v) for v in label_to_indices.values()) - 1
        k = 2 * max_num_relevant + 5
        print(f"自动设定 k = {k} (最大同类 = {max_num_relevant + 1})")

        # 评估指标
        NN_correct = 0
        FT_total, ST_total = 0, 0
        nDCG_total, MRR_total = 0, 0
        valid_samples = 0
        per_class_total = defaultdict(int)
        per_class_correct = defaultdict(int)

        print("开始全面评估...")
        for i in tqdm(range(len(features))):
            query_feat = features[i:i + 1]
            query_label = get_label(i)
            relevant_indices = set(label_to_indices[query_label]) - {i}
            num_relevant = len(relevant_indices)

            if num_relevant == 0:
                continue

            # 检索Top-k结果
            D, I = index.search(query_feat, k)
            I = I[0].tolist()

            # NN (Top-1)
            top1 = I[1] if I[0] == i else I[0]
            if get_label(top1) == query_label:
                NN_correct += 1

            # First Tier 和 Second Tier
            FT_K = num_relevant
            ST_K = 2 * num_relevant
            FT_recall = sum(1 for j in I[1:FT_K + 1] if j in relevant_indices) / num_relevant
            ST_recall = sum(1 for j in I[1:ST_K + 1] if j in relevant_indices) / num_relevant
            FT_total += FT_recall
            ST_total += ST_recall

            # nDCG
            relevance = [1 if j in relevant_indices else 0 for j in I[1:]]
            DCG = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
            IDCG = sum(1.0 / np.log2(i + 2) for i in range(min(num_relevant, len(relevance))))
            nDCG_total += DCG / IDCG if IDCG > 0 else 0

            # MRR
            for rank, j in enumerate(I[1:], 1):
                if j in relevant_indices:
                    MRR_total += 1.0 / rank
                    break

            # 按类别统计
            per_class_total[query_label] += 1
            if query_label == get_label(top1):
                per_class_correct[query_label] += 1

            valid_samples += 1

        # 打印结果
        print(f"\n📊 模型评测结果")
        print(f"✅ NN (Top-1): {NN_correct}/{valid_samples} = {NN_correct / valid_samples:.2%}")
        print(f"✅ First Tier (FT): {FT_total / valid_samples:.2%}")
        print(f"✅ Second Tier (ST): {ST_total / valid_samples:.2%}")
        print(f"✅ nDCG: {nDCG_total / valid_samples:.2%}")
        print(f"✅ ANMRR: {1 - MRR_total / valid_samples:.3f}")

        # 打印分类准确率
        print("\n📂 分类 Top-1:")
        for cls_idx in sorted(per_class_total):
            correct = per_class_correct[cls_idx]
            total = per_class_total[cls_idx]
            acc = 100 * correct / total if total > 0 else 0

            # 使用类别名称而不是索引
            if class_names and cls_idx < len(class_names):
                cls_name = class_names[cls_idx]
            else:
                cls_name = f"Class_{cls_idx}"

            print(f"  {cls_name:20s}: {acc:.2f}% ({correct}/{total})")

        return {
            'NN': NN_correct / valid_samples,
            'FT': FT_total / valid_samples,
            'ST': ST_total / valid_samples,
            'nDCG': nDCG_total / valid_samples,
            'ANMRR': 1 - MRR_total / valid_samples
        }
    else:
        print("无有效数据评估")
        return None


def evaluate_trained_model(model_path, test_dir=None):
    """评估已训练好的模型"""
    # 如果未指定测试集路径，使用默认路径
    if test_dir is None:
        test_dir = os.path.join(config.data_root, "test_processed")

    if not os.path.exists(test_dir):
        raise ValueError(f"测试集路径不存在: {test_dir}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载测试数据
    print(f"加载测试集: {test_dir}")
    test_loader, classes = get_test_dataloader(test_dir)
    num_classes = len(classes)
    print(f"类别列表: {classes} (共{num_classes}类)")

    # 加载CLIP基础模型
    print(f"加载CLIP模型: {config.clip_model}")
    clip_model, _ = clip.load(config.clip_model, device=config.device)

    # 创建模型并加载权重
    model = RetrievalCLIPAdapter(clip_model, num_classes).to(config.device)
    print(f"加载训练好的模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 确保模型处于评估模式
    model.eval()

    # 进行评估
    print("开始模型评估...")
    results = comprehensive_retrieval_evaluation(model, test_loader, "trained_model", classes)

    return results


if __name__ == "__main__":
    # 在这里指定模型路径和测试集路径
    model_path = "best_retrieval_clip.pth"  # 替换为你的模型路径
    test_dir = None  # 默认为config.data_root下的test_processed

    # 执行评估
    evaluate_trained_model(model_path, test_dir)