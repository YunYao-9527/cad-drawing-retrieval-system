# 离线视觉检索评测结果

- 时间: `2026-04-20 11:46:56`
- 数据集图片数: `200`
- 查询数: `6`
- 设备: `cuda`
- 模式: `baseline, full_model`

## 消融结果

| label | query_count | recall_at_1 | recall_at_5 | recall_at_10 | map_at_k | ndcg_at_k | first_tier | second_tier | anmrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 6 | 1.0 | 1.0 | 1.0 | 0.9301 | 0.9585 | 0.7768 | 0.8303 | 0.1498 |
| YOLO Cleaning + Structure | 6 | 1.0 | 1.0 | 1.0 | 0.9833 | 0.9894 | 0.8055 | 0.8445 | 0.1536 |

## Full Model 分类结果

| class | queries | recall_at_1 | recall_at_5 | recall_at_10 | map_at_k | ndcg_at_k | first_tier | second_tier | anmrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 人孔 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9474 | 0.9474 | 0.0455 |
| 叶轮叶片 | 2 | 1.0 | 1.0 | 1.0 | 0.95 | 0.9682 | 0.5 | 0.5862 | 0.4075 |
| 垫圈 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9692 | 1.0 | 0.0076 |