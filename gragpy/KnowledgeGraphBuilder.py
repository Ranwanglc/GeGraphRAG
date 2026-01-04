#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG知识图谱构建器
将DNS日志数据和异常IP标签转化为GraphRAG可用的知识图谱

作者: WebSWEAgent
日期: 2025-08-10
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


class DNSKnowledgeGraphBuilder:
    def __init__(self):
        """
        初始化DNS知识图谱构建器
        """
        self.entities = {}  # 实体存储
        self.relationships = []  # 关系存储
        self.documents = []  # 文档存储（用于GraphRAG）

        # 异常类型定义
        self.anomaly_types = {
            'botnet_c2': {
                'name': 'Botnet C&C通信',
                'description': '僵尸网络控制服务器通信，特征包括定期连接可疑域名、DGA域名查询、非标准端口通信',
                'indicators': ['dga_domains', 'periodic_queries', 'suspicious_servers']
            },
            'dns_tunneling': {
                'name': 'DNS隧道',
                'description': 'DNS隧道数据泄露，特征包括异常长的域名查询、高频率TXT记录查询、编码数据传输',
                'indicators': ['long_domains', 'txt_queries', 'high_frequency']
            },
            'malware_communication': {
                'name': '恶意软件通信',
                'description': '恶意软件与远程服务器通信，特征包括查询已知恶意域名、异常时间模式、可疑IP通信',
                'indicators': ['malicious_domains', 'night_activity', 'suspicious_ips']
            },
            'data_exfiltration': {
                'name': '数据泄露',
                'description': '数据外泄行为，特征包括大量外部DNS查询、访问云存储服务、异常数据传输模式',
                'indicators': ['external_queries', 'cloud_services', 'large_transfers']
            },
            'reconnaissance': {
                'name': '网络侦察',
                'description': '网络侦察活动，特征包括扫描行为、多域名查询、信息收集模式',
                'indicators': ['scanning_behavior', 'multiple_domains', 'info_gathering']
            },
            'phishing': {
                'name': '钓鱼攻击',
                'description': '钓鱼网站访问，特征包括访问仿冒域名、短期域名、可疑重定向',
                'indicators': ['fake_domains', 'new_domains', 'redirections']
            }
        }

    def analyze_ip_behavior(self, ip_data):
        """
        分析IP行为模式，生成行为特征描述
        """
        behavior_desc = []

        # 查询域名分析
        domains = ip_data.get('queried_domains', set())
        if domains:
            domain_count = len(domains)
            avg_length = np.mean([len(d) for d in domains])

            behavior_desc.append(f"查询了{domain_count}个不同域名")

            if avg_length > 20:
                behavior_desc.append("查询域名平均长度异常（可能存在DGA域名）")

            # 检查可疑域名特征
            suspicious_patterns = 0
            for domain in domains:
                if len(domain) > 30:
                    suspicious_patterns += 1
                if domain.count('-') > 3:
                    suspicious_patterns += 1
                if len([c for c in domain if c.isdigit()]) > len(domain) * 0.5:
                    suspicious_patterns += 1

            if suspicious_patterns > 0:
                behavior_desc.append(f"发现{suspicious_patterns}个可疑域名模式")

        # 时间行为分析
        query_times = ip_data.get('query_times', [])
        if query_times:
            try:
                dt_list = [pd.to_datetime(t) for t in query_times if t]
                if dt_list:
                    hours = [dt.hour for dt in dt_list]
                    night_queries = len([h for h in hours if h >= 22 or h <= 6])

                    if night_queries > len(hours) * 0.3:
                        behavior_desc.append("存在大量夜间活动（可能为自动化行为）")

                    # 查询频率分析
                    if len(dt_list) > 1:
                        time_span = (max(dt_list) - min(dt_list)).total_seconds()
                        if time_span > 0:
                            frequency = len(dt_list) / (time_span / 3600)
                            if frequency > 10:
                                behavior_desc.append(f"高频查询活动（{frequency:.1f}次/小时）")
            except:
                pass

        # 服务器通信分析
        servers = ip_data.get('server_ips', set())
        if servers:
            server_count = len(servers)
            behavior_desc.append(f"与{server_count}个不同服务器通信")

            # 检查公网IP比例
            public_count = 0
            for server in servers:
                try:
                    import ipaddress
                    if not ipaddress.IPv4Address(server).is_private:
                        public_count += 1
                except:
                    pass

            if public_count > 0:
                public_ratio = public_count / server_count
                if public_ratio > 0.8:
                    behavior_desc.append("主要与公网服务器通信（可能存在外联风险）")

        return "; ".join(behavior_desc) if behavior_desc else "正常DNS查询行为"

    def classify_anomaly_type(self, ip_address, ip_data, is_anomaly=False):
        """
        基于IP行为数据分类异常类型
        """
        if not is_anomaly:
            return 'normal', '正常IP，无异常行为'

        # 特征提取
        domains = ip_data.get('queried_domains', set())
        query_times = ip_data.get('query_times', [])
        servers = ip_data.get('server_ips', set())

        scores = {}

        # Botnet C&C 特征评分
        botnet_score = 0
        if domains:
            # DGA域名特征
            dga_count = 0
            for domain in domains:
                if (len(domain) > 20 and
                        len([c for c in domain if c.isdigit()]) > len(domain) * 0.3):
                    dga_count += 1

            if dga_count > 0:
                botnet_score += 3

            # 定期查询模式
            if len(query_times) > 10:
                botnet_score += 2

        scores['botnet_c2'] = botnet_score

        # DNS隧道特征评分
        tunnel_score = 0
        if domains:
            long_domains = [d for d in domains if len(d) > 50]
            if long_domains:
                tunnel_score += 4

            # 高频查询
            if len(query_times) > 50:
                tunnel_score += 2

        scores['dns_tunneling'] = tunnel_score

        # 恶意软件通信特征评分
        malware_score = 0
        if domains:
            # 可疑域名关键词
            suspicious_keywords = ['bot', 'c2', 'evil', 'malicious', 'trojan']
            for domain in domains:
                if any(keyword in domain.lower() for keyword in suspicious_keywords):
                    malware_score += 3

        # 夜间活动
        try:
            dt_list = [pd.to_datetime(t) for t in query_times if t]
            if dt_list:
                hours = [dt.hour for dt in dt_list]
                night_ratio = len([h for h in hours if h >= 22 or h <= 6]) / len(hours)
                if night_ratio > 0.5:
                    malware_score += 2
        except:
            pass

        scores['malware_communication'] = malware_score

        # 数据泄露特征评分
        exfiltration_score = 0
        if servers:
            # 大量外部服务器
            if len(servers) > 10:
                exfiltration_score += 2

            # 云服务域名
            cloud_keywords = ['amazonaws', 'azure', 'google', 'dropbox']
            for domain in domains:
                if any(keyword in domain.lower() for keyword in cloud_keywords):
                    exfiltration_score += 3

        scores['data_exfiltration'] = exfiltration_score

        # 网络侦察特征评分
        recon_score = 0
        if domains:
            # 多样化域名查询
            if len(domains) > 20:
                recon_score += 2

            # 扫描模式
            if len(servers) > 5:
                recon_score += 1

        scores['reconnaissance'] = recon_score

        # 钓鱼攻击特征评分
        phishing_score = 0
        if domains:
            # 仿冒域名特征
            common_brands = ['google', 'microsoft', 'apple', 'amazon', 'paypal']
            for domain in domains:
                for brand in common_brands:
                    if brand in domain and brand != domain.split('.')[0]:
                        phishing_score += 3

        scores['phishing'] = phishing_score

        # 选择得分最高的异常类型
        if not scores or max(scores.values()) == 0:
            return 'unknown_anomaly', '未知异常类型，需要进一步分析'

        best_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(scores[best_type] / 5.0, 1.0)  # 归一化置信度

        type_info = self.anomaly_types[best_type]
        description = f"{type_info['name']}（置信度: {confidence:.2f}）- {type_info['description']}"

        return best_type, description

    def build_knowledge_graph(self, dns_log_path, label_path, output_dir="graphrag_data"):
        """
        构建知识图谱并生成GraphRAG所需的文档
        """
        print("=== 构建DNS安全知识图谱 ===")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取DNS日志数据
        print("1. 读取DNS日志数据...")
        file_extension = dns_log_path.lower().split('.')[-1]

        if file_extension == 'xlsx':
            df = pd.read_excel(dns_log_path, engine='openpyxl')
        else:
            df = pd.read_csv(dns_log_path, encoding='utf-8')

        print(f"   读取到 {len(df)} 条DNS查询记录")

        # 读取异常IP标签
        print("2. 读取异常IP标签...")
        label_df = pd.read_csv(label_path, encoding='utf-8')
        abnormal_ips = set(label_df.iloc[:, 0].dropna().astype(str).unique())
        print(f"   发现 {len(abnormal_ips)} 个异常IP")

        # 预处理IP行为数据
        print("3. 分析IP行为模式...")
        ip_behaviors = self._preprocess_ip_behaviors(df)

        # 构建实体和关系
        print("4. 构建知识图谱实体和关系...")
        documents = []
        anomaly_classifications = {}

        for ip_address, behavior_data in ip_behaviors.items():
            is_anomaly = ip_address in abnormal_ips

            # 分类异常类型
            anomaly_type, type_description = self.classify_anomaly_type(
                ip_address, behavior_data, is_anomaly
            )

            if is_anomaly:
                anomaly_classifications[ip_address] = {
                    'type': anomaly_type,
                    'description': type_description
                }

            # 生成IP实体描述
            behavior_desc = self.analyze_ip_behavior(behavior_data)

            # 创建文档（用于GraphRAG）
            doc_content = f"""
IP地址: {ip_address}
状态: {'异常IP' if is_anomaly else '正常IP'}
异常类型: {anomaly_type if is_anomaly else '无'}
行为描述: {behavior_desc}
详细分析: {type_description if is_anomaly else '正常DNS查询行为，无安全风险'}

查询统计:
- 查询域名数量: {len(behavior_data.get('queried_domains', set()))}
- 查询总次数: {behavior_data.get('query_count', 0)}
- 通信服务器数量: {len(behavior_data.get('server_ips', set()))}
- 配置命中数量: {len(behavior_data.get('config_hits', set()))}

时间模式:
- 查询时间记录: {len(behavior_data.get('query_times', []))}条

网络行为:
- 服务器IP列表: {', '.join(list(behavior_data.get('server_ips', set()))[:5])}
- 查询域名示例: {', '.join(list(behavior_data.get('queried_domains', set()))[:5])}
"""

            documents.append({
                'id': f"ip_{ip_address}",
                'content': doc_content.strip(),
                'metadata': {
                    'ip_address': ip_address,
                    'is_anomaly': is_anomaly,
                    'anomaly_type': anomaly_type,
                    'entity_type': 'ip_entity'
                }
            })

        # 生成域名实体文档
        print("5. 生成域名知识文档...")
        domain_stats = defaultdict(lambda: {'ips': set(), 'queries': 0})

        for idx, row in df.iterrows():
            domain = str(row['查询内容']).strip()
            client_ip = str(row['客户端ip地址']).strip()

            if domain and domain != 'nan' and client_ip and client_ip != 'nan':
                domain_stats[domain]['ips'].add(client_ip)
                domain_stats[domain]['queries'] += 1

        for domain, stats in domain_stats.items():
            if len(stats['ips']) >= 2:  # 只包含被多个IP查询的域名
                anomaly_ips = [ip for ip in stats['ips'] if ip in abnormal_ips]

                doc_content = f"""
域名: {domain}
查询统计:
- 查询总次数: {stats['queries']}
- 查询IP数量: {len(stats['ips'])}
- 异常IP查询数: {len(anomaly_ips)}

安全评估:
- 风险等级: {'高风险' if len(anomaly_ips) > 0 else '正常'}
- 异常IP列表: {', '.join(anomaly_ips) if anomaly_ips else '无'}

域名特征:
- 域名长度: {len(domain)}
- 子域名层级: {domain.count('.')}
- 顶级域名: {domain.split('.')[-1] if '.' in domain else domain}
"""

                documents.append({
                    'id': f"domain_{domain}",
                    'content': doc_content.strip(),
                    'metadata': {
                        'domain': domain,
                        'query_count': stats['queries'],
                        'ip_count': len(stats['ips']),
                        'anomaly_ip_count': len(anomaly_ips),
                        'entity_type': 'domain_entity'
                    }
                })

        # 生成异常类型知识文档
        print("6. 生成异常类型知识库...")
        for anomaly_type, type_info in self.anomaly_types.items():
            doc_content = f"""
异常类型: {type_info['name']}
类型标识: {anomaly_type}

详细描述:
{type_info['description']}

主要特征指标:
{', '.join(type_info['indicators'])}

检测要点:
- 该类型异常通常表现为特定的网络行为模式
- 需要结合多个指标进行综合判断
- 建议采取相应的安全防护措施

相关安全建议:
- 监控相关网络流量
- 加强访问控制
- 定期安全评估
"""

            documents.append({
                'id': f"anomaly_type_{anomaly_type}",
                'content': doc_content.strip(),
                'metadata': {
                    'anomaly_type': anomaly_type,
                    'type_name': type_info['name'],
                    'entity_type': 'anomaly_type'
                }
            })

        # 保存文档数据
        print("7. 保存知识图谱数据...")

        # 保存为GraphRAG输入格式
        with open(os.path.join(output_dir, "input_documents.json"), 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        # 保存为文本文件（GraphRAG的另一种输入格式）
        with open(os.path.join(output_dir, "input_documents.txt"), 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(f"# {doc['id']}\n\n")
                f.write(doc['content'])
                f.write("\n\n" + "=" * 80 + "\n\n")

        # 保存异常分类结果
        with open(os.path.join(output_dir, "anomaly_classifications.json"), 'w', encoding='utf-8') as f:
            json.dump(anomaly_classifications, f, ensure_ascii=False, indent=2)

        # 保存实体关系数据
        entities_data = {
            'ip_entities': list(ip_behaviors.keys()),
            'domain_entities': list(domain_stats.keys()),
            'anomaly_types': list(self.anomaly_types.keys())
        }

        with open(os.path.join(output_dir, "entities.json"), 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)

        print(f"\n=== 知识图谱构建完成 ===")
        print(f"输出目录: {output_dir}")
        print(f"文档数量: {len(documents)}")
        print(f"IP实体: {len(ip_behaviors)}")
        print(f"域名实体: {len(domain_stats)}")
        print(f"异常IP分类: {len(anomaly_classifications)}")

        return {
            'documents': documents,
            'anomaly_classifications': anomaly_classifications,
            'output_dir': output_dir
        }

    def _preprocess_ip_behaviors(self, df):
        """
        预处理IP行为数据
        """
        ip_behaviors = defaultdict(lambda: {
            'queried_domains': set(),
            'query_count': 0,
            'server_ips': set(),
            'config_hits': set(),
            'query_times': [],
            'query_types': [],
            'protocols': []
        })

        for idx, row in df.iterrows():
            client_ip = str(row['客户端ip地址']).strip()
            server_ip = str(row.get('服务端ip地址', '')).strip()
            domain = str(row['查询内容']).strip()
            config_id = str(row.get('配置ID', '')).strip()
            discover_time = str(row.get('发现时间', '')).strip()
            query_type = str(row.get('查询类型', '')).strip()
            protocol = str(row.get('协议类型', '')).strip()

            if client_ip and client_ip != 'nan':
                behavior = ip_behaviors[client_ip]
                behavior['query_count'] += 1

                if domain and domain != 'nan':
                    behavior['queried_domains'].add(domain)

                if server_ip and server_ip != 'nan':
                    behavior['server_ips'].add(server_ip)

                if config_id and config_id != 'nan':
                    behavior['config_hits'].add(config_id)

                if discover_time and discover_time != 'nan':
                    behavior['query_times'].append(discover_time)

                if query_type and query_type != 'nan':
                    behavior['query_types'].append(query_type)

                if protocol and protocol != 'nan':
                    behavior['protocols'].append(protocol)

        return dict(ip_behaviors)


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    dns_log_path = "./log.xlsx"  # DNS日志文件路径
    label_path = "./label.csv"  # 异常IP标签文件路径
    output_dir = "./graphrag_knowledge"  # 输出目录

    # 创建知识图谱构建器
    builder = DNSKnowledgeGraphBuilder()

    try:
        # 构建知识图谱
        result = builder.build_knowledge_graph(dns_log_path, label_path, output_dir)

        print(f"\n知识图谱构建成功！")
        print(f"可以使用以下文件进行GraphRAG部署:")
        print(f"- {output_dir}/input_documents.txt")
        print(f"- {output_dir}/anomaly_classifications.json")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保DNS日志文件和标签文件路径正确")
    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    dns_document=""


    chunks = text_splitter.split_text(dns_document)


    entities = llm_extract_entities(chunks)
    # 结果: ["192.168.1.100", "高风险", "15个域名", "夜间活动", "高频查询"]

    # 4. 关系抽取
    relationships = llm_extract_relationships(chunks)
    # 结果: [("192.168.1.100", "具有", "高风险"), ("192.168.1.100", "表现为", "夜间活动")]

    # 5. 生成多层次嵌入
    chunk_embeddings = generate_embeddings(chunks)
    entity_embeddings = generate_embeddings(entities)
    relationship_embeddings = generate_embeddings(relationships)

    # 6. 存储到向量数据库
    embedding_index.add_embeddings("chunks", chunk_embeddings)
    embedding_index.add_embeddings("entities", entity_embeddings)
    embedding_index.add_embeddings("relationships", relationship_embeddings)