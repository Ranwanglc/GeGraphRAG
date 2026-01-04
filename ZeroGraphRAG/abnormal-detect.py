#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常IP分类查询系统
使用GraphRAG和大模型进行异常IP分类，并计算准确率

作者: WebSWEAgent
日期: 2025-08-10
"""

import os
import json
import pandas as pd
import subprocess
import requests
import time
from pathlib import Path
from collections import defaultdict
import re

class AnomalyIPClassifier:
    def __init__(self, graphrag_workspace="graphrag_workspace"):
        """
        初始化异常IP分类器
        """
        self.workspace = Path(graphrag_workspace)
        self.ollama_base_url = "http://localhost:11434"
        
        # 异常类型映射
        self.anomaly_type_mapping = {
            'botnet_c2': 'Botnet C&C通信',
            'dns_tunneling': 'DNS隧道',
            'malware_communication': '恶意软件通信',
            'data_exfiltration': '数据泄露',
            'reconnaissance': '网络侦察',
            'phishing': '钓鱼攻击',
            'normal': '正常',
            'unknown_anomaly': '未知异常'
        }
        
        # 查询模板
        self.query_templates = {
            'classification': """
请分析IP地址 {ip_address} 的异常行为类型。

基于以下信息进行分析：
- 该IP的DNS查询行为模式
- 网络通信特征
- 时间行为模式
- 已知的安全威胁指标

请从以下类型中选择最匹配的异常类型：
1. Botnet C&C通信 - 僵尸网络控制服务器通信
2. DNS隧道 - DNS隧道数据泄露
3. 恶意软件通信 - 恶意软件与远程服务器通信
4. 数据泄露 - 数据外泄行为
5. 网络侦察 - 网络侦察活动
6. 钓鱼攻击 - 钓鱼网站访问
7. 未知异常 - 其他类型的异常行为

请提供：
1. 异常类型分类结果
2. 置信度评分（0-1）
3. 主要判断依据
4. 建议的安全措施

格式要求：
异常类型：[具体类型]
置信度：[0-1的数值]
判断依据：[详细说明]
安全建议：[具体建议]
""",
            
            'detailed_analysis': """
请对IP地址 {ip_address} 进行详细的安全分析。

分析要点：
1. DNS查询行为是否异常
2. 网络通信模式分析
3. 时间行为特征
4. 与已知威胁的关联性
5. 潜在的安全风险

请提供全面的分析报告。
""",
            
            'batch_classification': """
请对以下异常IP地址进行批量分类分析：
{ip_list}

对每个IP，请提供：
1. 异常类型
2. 置信度
3. 主要特征

格式：
IP: [地址] | 类型: [异常类型] | 置信度: [数值] | 特征: [简要描述]
"""
        }
    
    def query_graphrag(self, query, method="global"):
        """
        查询GraphRAG系统
        """
        try:
            original_dir = os.getcwd()
            os.chdir(self.workspace)
            
            # Windows下需要指定编码
            result = subprocess.run(
                ["python", "-m", "graphrag.query", 
                 "--root", ".", 
                 "--method", method,
                 query],
                capture_output=True,
                text=True,
                encoding='utf-8',  # 明确指定UTF-8编码
                errors='ignore',   # 忽略编码错误
                timeout=120
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # 尝试不同编码读取错误信息
                error_msg = result.stderr
                if not error_msg:
                    try:
                        # 如果stderr为空，尝试用gbk解码
                        error_msg = result.stderr.encode('utf-8').decode('gbk', errors='ignore')
                    except:
                        error_msg = "编码错误，无法显示错误信息"
                
                print(f"GraphRAG查询失败: {error_msg}")
                return None
                
        except subprocess.TimeoutExpired:
            print("GraphRAG查询超时")
            return None
        except UnicodeDecodeError as e:
            print(f"编码错误: {e}")
            print("尝试使用备用方法...")
            return self._query_graphrag_fallback(query, method)
        except Exception as e:
            print(f"GraphRAG查询出错: {e}")
            return None
        finally:
            os.chdir(original_dir)
    
    def _query_graphrag_fallback(self, query, method="global"):
        """
        GraphRAG查询的备用方法（处理编码问题）
        """
        try:
            original_dir = os.getcwd()
            os.chdir(self.workspace)
            
            # 使用bytes模式避免编码问题
            result = subprocess.run(
                ["python", "-m", "graphrag.query", 
                 "--root", ".", 
                 "--method", method,
                 query],
                capture_output=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # 尝试多种编码解码输出
                output = None
                for encoding in ['utf-8', 'gbk', 'cp936', 'latin1']:
                    try:
                        output = result.stdout.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if output:
                    return output.strip()
                else:
                    print("无法解码GraphRAG输出")
                    return None
            else:
                print("GraphRAG查询失败（备用方法）")
                return None
                
        except Exception as e:
            print(f"备用查询方法也失败: {e}")
            return None
        finally:
            os.chdir(original_dir)
    
    def query_ollama_direct(self, prompt, model="qwen2.5:7b"):
        """
        直接查询ollama模型
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Ollama查询失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Ollama查询出错: {e}")
            return None
    
    def classify_single_ip(self, ip_address, use_graphrag=True):
        """
        分类单个IP地址
        """
        print(f"正在分析IP: {ip_address}")
        
        # 构建查询
        query = self.query_templates['classification'].format(ip_address=ip_address)
        
        # 使用GraphRAG查询
        if use_graphrag:
            response = self.query_graphrag(query)
        else:
            response = self.query_ollama_direct(query)
        
        if not response:
            return {
                'ip': ip_address,
                'anomaly_type': 'unknown_anomaly',
                'confidence': 0.0,
                'reasoning': '查询失败',
                'recommendations': '需要手动分析'
            }
        
        # 解析响应
        result = self.parse_classification_response(response)
        result['ip'] = ip_address
        result['raw_response'] = response
        
        return result
    
    def parse_classification_response(self, response):
        """
        解析分类响应
        """
        result = {
            'anomaly_type': 'unknown_anomaly',
            'confidence': 0.0,
            'reasoning': '',
            'recommendations': ''
        }
        
        try:
            # 提取异常类型
            type_match = re.search(r'异常类型[：:]\s*([^\n]+)', response)
            if type_match:
                type_text = type_match.group(1).strip()
                # 映射到标准类型
                for key, value in self.anomaly_type_mapping.items():
                    if value in type_text or key in type_text.lower():
                        result['anomaly_type'] = key
                        break
            
            # 提取置信度
            confidence_match = re.search(r'置信度[：:]\s*([0-9.]+)', response)
            if confidence_match:
                result['confidence'] = float(confidence_match.group(1))
            
            # 提取判断依据
            reasoning_match = re.search(r'判断依据[：:]\s*([^\n]+(?:\n[^：:]*)*)', response)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            
            # 提取安全建议
            recommendations_match = re.search(r'安全建议[：:]\s*([^\n]+(?:\n[^：:]*)*)', response)
            if recommendations_match:
                result['recommendations'] = recommendations_match.group(1).strip()
        
        except Exception as e:
            print(f"解析响应时出错: {e}")
        
        return result
    
    def classify_batch_ips(self, ip_list, use_graphrag=True, batch_size=5):
        """
        批量分类IP地址
        """
        print(f"开始批量分析 {len(ip_list)} 个IP地址")
        
        results = []
        
        # 分批处理
        for i in range(0, len(ip_list), batch_size):
            batch = ip_list[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}: {len(batch)} 个IP")
            
            for ip in batch:
                result = self.classify_single_ip(ip, use_graphrag)
                results.append(result)
                
                # 添加延迟避免过载
                time.sleep(1)
        
        return results
    
    def load_ground_truth_from_csv(self, abnormal_ips_file):
        """
        从abnormal_ips.csv文件加载真实分类标签
        第一列：IP地址，第二列：异常类型标签
        """
        try:
            df = pd.read_csv(abnormal_ips_file, encoding='utf-8')
            
            if df.shape[1] < 2:
                print("❌ abnormal_ips.csv文件缺少第二列（异常类型标签）")
                return {}
            
            ground_truth = {}
            
            # 读取第一列（IP地址）和第二列（异常类型）
            for idx, row in df.iterrows():
                ip_address = str(row.iloc[0]).strip()
                anomaly_type = str(row.iloc[1]).strip()
                
                if ip_address and ip_address != 'nan' and anomaly_type and anomaly_type != 'nan':
                    # 标准化异常类型名称
                    normalized_type = self.normalize_anomaly_type(anomaly_type)
                    
                    ground_truth[ip_address] = {
                        'type': normalized_type,
                        'description': self.anomaly_type_mapping.get(normalized_type, anomaly_type)
                    }
            
            print(f"✅ 从CSV文件加载了 {len(ground_truth)} 个IP的真实标签")
            
            # 显示标签分布
            type_counts = {}
            for gt in ground_truth.values():
                type_name = gt['type']
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            print("真实标签分布:")
            for anomaly_type, count in type_counts.items():
                type_name = self.anomaly_type_mapping.get(anomaly_type, anomaly_type)
                print(f"  {type_name}: {count}")
            
            return ground_truth
            
        except Exception as e:
            print(f"❌ 从CSV加载真实标签失败: {e}")
            return {}
    
    def normalize_anomaly_type(self, type_text):
        """
        标准化异常类型名称
        """
        type_text = type_text.lower().strip()
        
        # 映射常见的异常类型名称到标准名称
        type_mapping = {
            # Botnet C&C
            'botnet': 'botnet_c2',
            'botnet_c2': 'botnet_c2',
            'botnet_cc': 'botnet_c2',
            'c2': 'botnet_c2',
            'cc': 'botnet_c2',
            '僵尸网络': 'botnet_c2',
            
            # DNS隧道
            'dns_tunnel': 'dns_tunneling',
            'dns_tunneling': 'dns_tunneling',
            'tunnel': 'dns_tunneling',
            'dns隧道': 'dns_tunneling',
            '隧道': 'dns_tunneling',
            
            # 恶意软件通信
            'malware': 'malware_communication',
            'malware_communication': 'malware_communication',
            'malware_comm': 'malware_communication',
            '恶意软件': 'malware_communication',
            '木马': 'malware_communication',
            
            # 数据泄露
            'data_exfiltration': 'data_exfiltration',
            'exfiltration': 'data_exfiltration',
            'data_leak': 'data_exfiltration',
            '数据泄露': 'data_exfiltration',
            '数据外泄': 'data_exfiltration',
            
            # 网络侦察
            'reconnaissance': 'reconnaissance',
            'recon': 'reconnaissance',
            'scanning': 'reconnaissance',
            '侦察': 'reconnaissance',
            '扫描': 'reconnaissance',
            
            # 钓鱼攻击
            'phishing': 'phishing',
            'phish': 'phishing',
            '钓鱼': 'phishing',
            
            # 正常
            'normal': 'normal',
            'benign': 'normal',
            '正常': 'normal',
            
            # 未知异常
            'unknown': 'unknown_anomaly',
            'unknown_anomaly': 'unknown_anomaly',
            'other': 'unknown_anomaly',
            '未知': 'unknown_anomaly',
            '其他': 'unknown_anomaly'
        }
        
        return type_mapping.get(type_text, 'unknown_anomaly')
    
    def create_ground_truth_file(self, anomaly_ips, output_file="ground_truth_classifications.json"):
        """
        创建真实分类标签文件（示例）
        """
        print("创建示例真实分类标签文件...")
        
        # 示例分类（实际使用时需要根据真实情况标注）
        ground_truth = {}
        
        for i, ip in enumerate(anomaly_ips):
            # 这里是示例分类，实际使用时需要专家标注
            if i % 6 == 0:
                ground_truth[ip] = {
                    'type': 'botnet_c2',
                    'description': 'Botnet C&C通信'
                }
            elif i % 6 == 1:
                ground_truth[ip] = {
                    'type': 'dns_tunneling',
                    'description': 'DNS隧道'
                }
            elif i % 6 == 2:
                ground_truth[ip] = {
                    'type': 'malware_communication',
                    'description': '恶意软件通信'
                }
            elif i % 6 == 3:
                ground_truth[ip] = {
                    'type': 'data_exfiltration',
                    'description': '数据泄露'
                }
            elif i % 6 == 4:
                ground_truth[ip] = {
                    'type': 'reconnaissance',
                    'description': '网络侦察'
                }
            else:
                ground_truth[ip] = {
                    'type': 'phishing',
                    'description': '钓鱼攻击'
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, ensure_ascii=False, indent=2)
        
        print(f"示例真实标签文件已创建: {output_file}")
        print("注意：这是示例数据，实际使用时需要专家进行准确标注")
        
        return ground_truth
    
    def calculate_accuracy(self, predictions, ground_truth):
        """
        计算分类准确率
        """
        if not predictions or not ground_truth:
            print("预测结果或真实标签为空")
            return {}
        
        # 统计结果
        total = 0
        correct = 0
        type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions:
            ip = pred['ip']
            pred_type = pred['anomaly_type']
            
            if ip in ground_truth:
                total += 1
                true_type = ground_truth[ip]['type']
                
                type_stats[true_type]['total'] += 1
                
                if pred_type == true_type:
                    correct += 1
                    type_stats[true_type]['correct'] += 1
        
        # 计算总体准确率
        overall_accuracy = correct / total if total > 0 else 0
        
        # 计算各类型准确率
        type_accuracies = {}
        for anomaly_type, stats in type_stats.items():
            if stats['total'] > 0:
                type_accuracies[anomaly_type] = stats['correct'] / stats['total']
            else:
                type_accuracies[anomaly_type] = 0
        
        # 计算平均置信度
        confidences = [pred['confidence'] for pred in predictions if pred['confidence'] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'type_accuracies': type_accuracies,
            'average_confidence': avg_confidence,
            'type_statistics': dict(type_stats)
        }
    
    def generate_evaluation_report(self, predictions, ground_truth, output_file="evaluation_report.json"):
        """
        生成评估报告
        """
        print("生成评估报告...")
        
        # 计算准确率
        accuracy_results = self.calculate_accuracy(predictions, ground_truth)
        
        # 生成详细报告
        report = {
            'evaluation_summary': accuracy_results,
            'detailed_predictions': predictions,
            'ground_truth': ground_truth,
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_ips_evaluated': len(predictions)
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print(f"\n=== 评估报告摘要 ===")
        print(f"总体准确率: {accuracy_results['overall_accuracy']:.2%}")
        print(f"评估样本数: {accuracy_results['total_samples']}")
        print(f"正确预测数: {accuracy_results['correct_predictions']}")
        print(f"平均置信度: {accuracy_results['average_confidence']:.3f}")
        
        print(f"\n各类型准确率:")
        for anomaly_type, accuracy in accuracy_results['type_accuracies'].items():
            type_name = self.anomaly_type_mapping.get(anomaly_type, anomaly_type)
            print(f"  {type_name}: {accuracy:.2%}")
        
        print(f"\n详细报告已保存: {output_file}")
        
        return report
    
    def run_complete_evaluation(self, anomaly_ips_file, use_graphrag=True):
        """
        运行完整的评估流程
        """
        print("=== 开始异常IP分类评估 ===\n")
        
        # 1. 读取异常IP列表和真实标签
        print("1. 读取异常IP列表和真实标签...")
        try:
            df = pd.read_csv(anomaly_ips_file, encoding='utf-8')
            
            # 检查文件格式
            if df.shape[1] < 2:
                print("❌ 文件格式错误：需要至少两列（IP地址，异常类型）")
                print("当前文件列数:", df.shape[1])
                print("文件内容预览:")
                print(df.head())
                return None
            
            # 读取IP列表（第一列）
            anomaly_ips = df.iloc[:, 0].dropna().astype(str).unique().tolist()
            print(f"   发现 {len(anomaly_ips)} 个异常IP")
            
            # 读取真实标签（第二列）
            ground_truth = self.load_ground_truth_from_csv(anomaly_ips_file)
            
            if not ground_truth:
                print("❌ 无法加载真实标签，评估无法进行")
                return None
                
        except Exception as e:
            print(f"❌ 读取异常IP文件失败: {e}")
            return None
        
        # 2. 执行分类预测
        print(f"\n2. 执行分类预测（使用{'GraphRAG' if use_graphrag else 'Ollama直接查询'}）...")
        predictions = self.classify_batch_ips(anomaly_ips, use_graphrag)
        
        # 3. 计算准确率和生成报告
        print(f"\n3. 生成评估报告...")
        report = self.generate_evaluation_report(predictions, ground_truth)
        
        # 4. 保存预测结果
        predictions_file = "ip_classification_predictions.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        # 5. 保存真实标签（便于后续使用）
        ground_truth_file = "ground_truth_from_csv.json"
        with open(ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 评估完成 ===")
        print(f"预测结果文件: {predictions_file}")
        print(f"评估报告文件: evaluation_report.json")
        print(f"真实标签文件: {ground_truth_file}")
        
        return {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'evaluation_report': report
        }


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    graphrag_workspace = "graphrag_workspace"  # GraphRAG工作目录
    anomaly_ips_file = "abnormal_ips.csv"      # 异常IP文件
    
    # 创建分类器
    classifier = AnomalyIPClassifier(graphrag_workspace)
    
    # 检查GraphRAG工作目录
    if not Path(graphrag_workspace).exists():
        print(f"❌ GraphRAG工作目录不存在: {graphrag_workspace}")
        print("请先运行GraphRAG部署器")
        return
    
    # 运行完整评估
    try:
        result = classifier.run_complete_evaluation(
            anomaly_ips_file, 
            use_graphrag=True  # 设置为False可以直接使用ollama
        )
        
        if result:
            print("\n🎉 异常IP分类评估完成！")
            print("\n可以查看以下文件了解详细结果:")
            print("- ip_classification_predictions.json (预测结果)")
            print("- evaluation_report.json (评估报告)")
            print("- ground_truth_classifications.json (真实标签)")
        else:
            print("\n❌ 评估过程失败")
            
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()