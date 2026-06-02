#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统功能测试脚本
测试所有模块的功能是否符合工程化要求
"""
import sys
import os
import time
import requests
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

def test_health_check():
    """测试健康检查接口"""
    print("=" * 60)
    print("测试1: 健康检查接口")
    print("=" * 60)
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        assert response.status_code == 200, f"健康检查失败: {response.status_code}"
        data = response.json()
        assert data.get("status") in ["healthy", "degraded"], f"状态异常: {data.get('status')}"
        print("✅ 健康检查接口正常")
        print(f"   状态: {data.get('status')}")
        print(f"   运行时间: {data.get('uptime_seconds', 0):.2f}秒")
        return True
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_metrics():
    """测试指标接口"""
    print("\n" + "=" * 60)
    print("测试2: Prometheus指标接口")
    print("=" * 60)
    try:
        response = requests.get("http://127.0.0.1:5000/metrics", timeout=5)
        assert response.status_code == 200, f"指标接口失败: {response.status_code}"
        content = response.text
        # 检查关键指标
        assert "http_requests_total" in content, "缺少HTTP请求总数指标"
        assert "http_requests_qps" in content, "缺少QPS指标"
        assert "http_request_duration_seconds_p95" in content, "缺少p95延迟指标"
        assert "cpu_utilization_percent" in content, "缺少CPU利用率指标"
        print("✅ 指标接口正常")
        print("   包含QPS、p95延迟、CPU利用率等指标")
        return True
    except Exception as e:
        print(f"❌ 指标接口测试失败: {e}")
        return False

def test_api_info():
    """测试API信息接口"""
    print("\n" + "=" * 60)
    print("测试3: API信息接口")
    print("=" * 60)
    try:
        response = requests.get("http://127.0.0.1:5000/api", timeout=5)
        assert response.status_code == 200, f"API信息接口失败: {response.status_code}"
        data = response.json()
        assert "database_info" in data, "缺少数据库信息"
        print("✅ API信息接口正常")
        print(f"   服务: {data.get('service', 'N/A')}")
        print(f"   数据库图片数: {data.get('database_info', {}).get('total_images', 0)}")
        return True
    except Exception as e:
        print(f"❌ API信息接口测试失败: {e}")
        return False

def test_stats():
    """测试统计信息接口"""
    print("\n" + "=" * 60)
    print("测试4: 统计信息接口")
    print("=" * 60)
    try:
        response = requests.get("http://127.0.0.1:5000/stats", timeout=5)
        assert response.status_code == 200, f"统计信息接口失败: {response.status_code}"
        data = response.json()
        assert "total_images" in data, "缺少图片总数"
        assert "class_distribution" in data, "缺少类别分布"
        print("✅ 统计信息接口正常")
        print(f"   总图片数: {data.get('total_images', 0)}")
        print(f"   类别数: {data.get('current_categories', 0)}")
        return True
    except Exception as e:
        print(f"❌ 统计信息接口测试失败: {e}")
        return False

def test_search():
    """测试检索接口"""
    print("\n" + "=" * 60)
    print("测试5: 检索接口")
    print("=" * 60)
    try:
        # 创建一个测试图片（使用项目中已有的图片）
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print("⚠️  配置文件不存在，跳过检索测试")
            return True
        
        # 读取配置获取图库路径
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        gallery_dir = config.get('gallery', {}).get('cad_drawing_dir', '')
        if not gallery_dir or not os.path.exists(gallery_dir):
            print("⚠️  图库目录不存在，跳过检索测试")
            return True
        
        # 查找一个测试图片
        import glob
        test_images = glob.glob(os.path.join(gallery_dir, "**", "*.png"), recursive=True)
        if not test_images:
            test_images = glob.glob(os.path.join(gallery_dir, "**", "*.jpg"), recursive=True)
        
        if not test_images:
            print("⚠️  未找到测试图片，跳过检索测试")
            return True
        
        test_image_path = test_images[0]
        print(f"   使用测试图片: {os.path.basename(test_image_path)}")
        
        # 发送检索请求
        with open(test_image_path, 'rb') as f:
            files = {'image': (os.path.basename(test_image_path), f, 'image/png')}
            data = {'top_k': 5}
            response = requests.post("http://127.0.0.1:5000/search", files=files, data=data, timeout=30)
        
        assert response.status_code == 200, f"检索接口失败: {response.status_code}"
        result = response.json()
        assert result.get("status") == "success", f"检索失败: {result.get('message', 'unknown')}"
        
        results = result.get("results", [])
        print(f"✅ 检索接口正常")
        print(f"   返回结果数: {len(results)}")
        if results:
            print(f"   第一个结果: {results[0].get('filename', 'N/A')}, 相似度: {results[0].get('similarity', 0):.4f}")
        return True
    except Exception as e:
        print(f"❌ 检索接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_images_api():
    """测试图片管理API"""
    print("\n" + "=" * 60)
    print("测试6: 图片管理API")
    print("=" * 60)
    try:
        # 测试获取图片列表
        response = requests.get("http://127.0.0.1:5000/api/images?page=1&per_page=10", timeout=5)
        assert response.status_code == 200, f"获取图片列表失败: {response.status_code}"
        data = response.json()
        assert "images" in data, "缺少图片列表"
        print("✅ 图片列表接口正常")
        print(f"   总图片数: {data.get('total', 0)}")
        
        # 测试获取类别列表
        response = requests.get("http://127.0.0.1:5000/api/classes", timeout=5)
        assert response.status_code == 200, f"获取类别列表失败: {response.status_code}"
        data = response.json()
        assert "classes" in data, "缺少类别列表"
        print("✅ 类别列表接口正常")
        print(f"   类别数: {len(data.get('classes', []))}")
        
        return True
    except Exception as e:
        print(f"❌ 图片管理API测试失败: {e}")
        return False

def test_ready_live():
    """测试Kubernetes就绪和存活检查"""
    print("\n" + "=" * 60)
    print("测试7: Kubernetes检查接口")
    print("=" * 60)
    try:
        # 测试就绪检查
        response = requests.get("http://127.0.0.1:5000/ready", timeout=5)
        assert response.status_code == 200, f"就绪检查失败: {response.status_code}"
        print("✅ 就绪检查接口正常")
        
        # 测试存活检查
        response = requests.get("http://127.0.0.1:5000/live", timeout=5)
        assert response.status_code == 200, f"存活检查失败: {response.status_code}"
        print("✅ 存活检查接口正常")
        
        return True
    except Exception as e:
        print(f"❌ Kubernetes检查接口测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("CAD图纸检索系统 - 功能测试")
    print("=" * 60)
    print("\n等待服务启动...")
    time.sleep(2)
    
    # 检查服务是否运行
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=2)
    except:
        print("❌ 服务未运行，请先启动服务: python main.py")
        return False
    
    results = []
    
    # 运行所有测试
    results.append(("健康检查", test_health_check()))
    results.append(("指标接口", test_metrics()))
    results.append(("API信息", test_api_info()))
    results.append(("统计信息", test_stats()))
    results.append(("检索接口", test_search()))
    results.append(("图片管理", test_images_api()))
    results.append(("Kubernetes检查", test_ready_live()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

