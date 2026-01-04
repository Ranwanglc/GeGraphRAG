#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行GraphRAG异常IP分类系统
自动化完成整个流程：知识图谱构建 -> GraphRAG部署 -> 异常分类评估

作者: WebSWEAgent
日期: 2025-08-10
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """
    检查系统依赖
    """
    print("=== 检查系统依赖 ===")
    
    required_packages = [
        'pandas', 'numpy', 'requests', 'pyyaml', 'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    # 检查GraphRAG
    try:
        import graphrag
        print("✅ graphrag")
    except ImportError:
        print("❌ graphrag")
        print("请运行: pip install graphrag")
        return False
    
    return True

def check_input_files():
    """
    检查输入文件
    """
    print("\n=== 检查输入文件 ===")
    
    required_files = [
        'dns_logs.xlsx',
        'abnormal_ips.csv'
    ]
    
    missing_files = []
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n缺少输入文件: {', '.join(missing_files)}")
        print("请确保以下文件存在:")
        print("- dns_logs.xlsx: DNS日志文件")
        print("- abnormal_ips.csv: 异常IP标签文件")
        return False
    
    return True

def check_ollama_service():
    """
    检查ollama服务
    """
    print("\n=== 检查ollama服务 ===")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ ollama服务运行正常")
            return True
        else:
            print("❌ ollama服务响应异常")
            return False
    except Exception as e:
        print(f"❌ 无法连接到ollama服务: {e}")
        print("请先启动ollama服务: ollama serve")
        return False

def run_step(step_name, script_name, description):
    """
    运行单个步骤（Windows编码修复版）
    """
    print(f"\n=== {step_name}: {description} ===")
    
    try:
        # Windows下需要指定编码
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=1800  # 30分钟超时
        )
        
        if result.returncode == 0:
            print(f"✅ {step_name} 完成")
            if result.stdout:
                print("输出摘要:")
                # 只显示最后几行输出
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"  {line}")
            return True
        else:
            print(f"❌ {step_name} 失败")
            print("错误信息:")
            error_msg = result.stderr or "未知错误"
            print(error_msg)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {step_name} 超时")
        return False
    except UnicodeDecodeError as e:
        print(f"❌ {step_name} 编码错误: {e}")
        print("尝试使用备用方法...")
        return run_step_fallback(step_name, script_name, description)
    except Exception as e:
        print(f"❌ {step_name} 出错: {e}")
        return False

def run_step_fallback(step_name, script_name, description):
    """
    运行步骤的备用方法（处理编码问题）
    """
    print(f"使用备用方法运行 {step_name}...")
    
    try:
        # 使用bytes模式避免编码问题
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            timeout=1800
        )
        
        if result.returncode == 0:
            print(f"✅ {step_name} 完成（备用方法）")
            
            # 尝试解码输出
            output = None
            for encoding in ['utf-8', 'gbk', 'cp936']:
                try:
                    output = result.stdout.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if output:
                print("输出摘要:")
                lines = output.strip().split('\n')
                for line in lines[-5:]:  # 只显示最后5行
                    print(f"  {line}")
            
            return True
        else:
            print(f"❌ {step_name} 失败（备用方法）")
            return False
            
    except Exception as e:
        print(f"❌ 备用方法也失败: {e}")
        return False

def create_sample_data():
    """
    创建示例数据（如果输入文件不存在）
    """
    print("\n=== 创建示例数据 ===")
    
    # 创建示例DNS日志
    if not Path('dns_logs.xlsx').exists():
        print("创建示例DNS日志文件...")
        
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_records = 100
        
        # 生成示例数据
        sample_data = {
            '配置ID': [f"config_{np.random.randint(1,10):03d}" for _ in range(n_records)],
            '发现时间': [f"2025-08-10 {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:00" 
                       for _ in range(n_records)],
            '接收时间': [f"2025-08-10 {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:00" 
                       for _ in range(n_records)],
            '客户端地址定位信息': ['示例地址'] * n_records,
            '客户端ip地址': [f"192.168.{np.random.randint(1,10)}.{np.random.randint(1,255)}" 
                          for _ in range(n_records)],
            '客户端端口': [53000 + i for i in range(n_records)],
            '协议类型': np.random.choice(['UDP', 'TCP'], n_records),
            '服务端地址定位信息': ['示例云服务'] * n_records,
            '服务端ip地址': [f"8.8.{np.random.randint(1,10)}.{np.random.randint(1,255)}" 
                          for _ in range(n_records)],
            '服务端端口': [53] * n_records,
            '查询内容': np.random.choice([
                'www.example.com', 'test.domain.com', 'suspicious-site.evil',
                'normal-website.org', 'malicious-domain.bad'
            ], n_records),
            '查询类': ['IN'] * n_records,
            '查询类型': np.random.choice(['A', 'AAAA'], n_records),
            '出入口编号': [f"{np.random.randint(1,5):03d}" for _ in range(n_records)],
            '处理机IP': [f"10.1.1.{np.random.randint(1,5)}" for _ in range(n_records)],
            '递归请求': np.random.choice(['是', '否'], n_records),
            'OPCODE': ['QUERY'] * n_records,
            '欺骗包的应答类型': [''] * n_records,
            '欺骗包RCODE': [''] * n_records,
            '欺骗策略': [''] * n_records,
            '欺骗记录': [''] * n_records,
            '业务类型': ['正常查询'] * n_records,
            '管控动作': ['允许'] * n_records,
            '嵌套地址列表': [''] * n_records,
            '传输方向': np.random.choice(['出站', '入站'], n_records)
        }
        
        df = pd.DataFrame(sample_data)
        df.to_excel('dns_logs.xlsx', index=False, engine='openpyxl')
        print("✅ 示例DNS日志文件已创建")
    
    # 创建示例异常IP标签
    if not Path('abnormal_ips.csv').exists():
        print("创建示例异常IP标签文件...")
        
        # 从DNS日志中随机选择一些IP作为异常IP
        df = pd.read_excel('dns_logs.xlsx')
        unique_ips = df['客户端ip地址'].unique()
        
        # 选择20%的IP作为异常IP
        n_anomaly = max(1, len(unique_ips) // 5)
        anomaly_ips = np.random.choice(unique_ips, n_anomaly, replace=False)
        
        label_df = pd.DataFrame({'异常IP': anomaly_ips})
        label_df.to_csv('abnormal_ips.csv', index=False, encoding='utf-8')
        print(f"✅ 示例异常IP标签文件已创建 ({len(anomaly_ips)} 个异常IP)")

def main():
    """
    主函数 - 一键运行完整流程
    """
    print("🚀 GraphRAG异常IP分类系统一键部署")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装缺少的依赖包")
        return False
    
    # 2. 检查输入文件
    if not check_input_files():
        print("\n是否创建示例数据进行测试？(y/n): ", end="")
        choice = input().lower().strip()
        if choice == 'y':
            create_sample_data()
        else:
            print("❌ 缺少输入文件，程序退出")
            return False
    
    # 3. 检查ollama服务
    if not check_ollama_service():
        print("\n❌ ollama服务检查失败")
        print("请先启动ollama服务:")
        print("1. 在终端运行: ollama serve")
        print("2. 在另一个终端运行: ollama pull qwen2.5:7b")
        print("3. 运行: ollama pull nomic-embed-text")
        return False
    
    print("\n🎯 开始执行完整流程...")
    
    # 步骤1: 构建知识图谱
    success1 = run_step(
        "步骤1", 
        "GraphRAG知识图谱构建器.py",
        "构建DNS安全知识图谱"
    )
    
    if not success1:
        print("❌ 知识图谱构建失败，程序终止")
        return False
    
    # 步骤2: 部署GraphRAG系统
    success2 = run_step(
        "步骤2",
        "GraphRAG部署器.py", 
        "部署GraphRAG系统"
    )
    
    if not success2:
        print("❌ GraphRAG部署失败，程序终止")
        return False
    
    # 步骤3: 运行异常IP分类评估
    success3 = run_step(
        "步骤3",
        "异常IP分类查询系统.py",
        "执行异常IP分类和评估"
    )
    
    if not success3:
        print("❌ 异常IP分类评估失败")
        return False
    
    # 完成
    print("\n" + "=" * 60)
    print("🎉 GraphRAG异常IP分类系统部署完成！")
    print("=" * 60)
    
    print("\n📁 生成的文件:")
    output_files = [
        "graphrag_knowledge/",
        "graphrag_workspace/", 
        "ground_truth_classifications.json",
        "ip_classification_predictions.json",
        "evaluation_report.json"
    ]
    
    for file_path in output_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print("\n📊 使用方法:")
    print("1. 查看评估报告: evaluation_report.json")
    print("2. 查看预测结果: ip_classification_predictions.json")
    print("3. 手动查询GraphRAG:")
    print("   cd graphrag_workspace")
    print("   python -m graphrag.query --root . --method global '查询内容'")
    
    print("\n🔧 系统维护:")
    print("- 更新DNS日志: 替换 dns_logs.xlsx 并重新运行步骤1-3")
    print("- 添加新异常类型: 修改 GraphRAG知识图谱构建器.py")
    print("- 调整模型参数: 修改 graphrag_workspace/settings.yaml")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ 系统部署成功，可以开始使用！")
        else:
            print("\n💥 系统部署失败，请检查错误信息")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n💥 程序执行出错: {e}")
        import traceback
        traceback.print_exc()