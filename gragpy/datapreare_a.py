#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNS查询日志图数据集构建器 - 方案A：IP异常行为检测优化版
专注于IP异常行为检测，移除域名节点，增强IP行为特征

作者: WebSWEAgent
日期: 2025-08-10
版本: 2.0 (方案A)
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import hashlib
import ipaddress
from datetime import datetime
import re
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')

# 检查openpyxl是否安装
try:
    import openpyxl
except ImportError:
    print("警告: 未安装openpyxl库，无法读取xlsx文件")
    print("请运行: pip install openpyxl")


class DNSGraphBuilderV2:
    def __init__(self, feature_dim=512):
        """
        初始化DNS图构建器 - 方案A版本

        Args:
            feature_dim (int): 节点特征维度，默认512
        """
        self.feature_dim = feature_dim
        self.node_mapping = {}  # 节点到索引的映射
        self.node_types = {}  # 节点类型映射
        self.node_features = []  # 节点特征列表
        self.edges = []  # 边列表
        self.labels = []  # 节点标签列表

        # 节点类型定义 - 简化版本
        self.NODE_TYPES = {
            'ip': 0,  # 统一的IP节点（客户端+服务端）
            'config': 1,  # 配置节点
            'time': 2  # 时间节点（可选，简化版）
        }

        # 关系类型定义 - 重新设计
        self.EDGE_TYPES = {
            'hit': 0,  # IP-配置命中关系
            'communication': 1,  # IP-IP通信关系
            'similarity': 2,  # IP-IP相似性关系（基于共同查询域名）
            'same_subnet': 3,  # IP-IP同子网关系
            'temporal': 4  # IP-时间关系（可选）
        }

        # IP行为统计
        self.ip_behaviors = {}

    def _hash_to_vector(self, text, seed=42):
        """
        将文本哈希为固定维度的向量
        """
        hash_obj = hashlib.md5(str(text).encode())
        hash_hex = hash_obj.hexdigest()
        hash_int = int(hash_hex, 16)

        np.random.seed(hash_int % (2 ** 32) + seed)
        vector = np.random.normal(0, 1, self.feature_dim)
        vector = vector / np.linalg.norm(vector)

        return vector.astype(np.float32)

    def _ip_to_basic_vector(self, ip_str, dim=4):
        """
        将IP地址转换为基础特征向量
        """
        try:
            ip = ipaddress.IPv4Address(ip_str)
            ip_int = int(ip)

            # 提取IP地址的各个字节
            bytes_list = [(ip_int >> (8 * i)) & 0xFF for i in range(4)]

            # 归一化到[0,1]
            normalized = [b / 255.0 for b in bytes_list]

            return np.array(normalized, dtype=np.float32)

        except:
            # 如果不是有效IP，返回零向量
            return np.zeros(dim, dtype=np.float32)

    def _calculate_domain_features(self, domains):
        """
        计算域名相关的统计特征
        """
        if not domains:
            return np.zeros(15, dtype=np.float32)

        domain_list = list(domains)
        # 过滤掉空域名
        domain_list = [d for d in domain_list if d and len(d) > 0]

        if not domain_list:
            return np.zeros(15, dtype=np.float32)

        # 基础统计
        total_domains = len(domain_list)
        domain_lengths = [len(d) for d in domain_list]
        avg_length = np.mean(domain_lengths)
        max_length = max(domain_lengths)
        min_length = min(domain_lengths)

        # 顶级域名分布
        tlds = [d.split('.')[-1] if '.' in d else d for d in domain_list]
        tld_counter = Counter(tlds)

        # 计算TLD熵，避免log(0)
        tld_entropy = 0.0
        if total_domains > 1:
            for count in tld_counter.values():
                if count > 0:
                    prob = count / total_domains
                    tld_entropy -= prob * np.log2(prob)

        # 域名长度分布
        length_variance = np.var(domain_lengths) if len(domain_lengths) > 1 else 0.0

        # 可疑域名特征
        suspicious_count = 0
        for domain in domain_list:
            domain_len = len(domain)
            if domain_len == 0:
                continue

            # 简单的可疑域名检测
            digit_count = len(re.findall(r'\d', domain))
            if (domain_len > 20 or  # 过长域名
                    digit_count > domain_len * 0.5 or  # 数字过多
                    '-' in domain and domain.count('-') > 3):  # 连字符过多
                suspicious_count += 1

        suspicious_ratio = suspicious_count / total_domains if total_domains > 0 else 0.0

        # 数字和特殊字符比例，避免除零
        digit_ratios = []
        special_char_ratios = []

        for d in domain_list:
            if len(d) > 0:
                digit_ratios.append(len(re.findall(r'\d', d)) / len(d))
                special_char_ratios.append(len(re.findall(r'[-_]', d)) / len(d))

        digit_ratio = np.mean(digit_ratios) if digit_ratios else 0.0
        special_char_ratio = np.mean(special_char_ratios) if special_char_ratios else 0.0

        # 子域名深度
        subdomain_depths = [d.count('.') for d in domain_list]
        avg_subdomain_depth = np.mean(subdomain_depths)
        max_subdomain_depth = max(subdomain_depths) if subdomain_depths else 0

        # 域名唯一性（去重后的比例）
        unique_ratio = len(set(domain_list)) / total_domains if total_domains > 0 else 0.0

        return np.array([
            total_domains,
            avg_length,
            max_length,
            min_length,
            tld_entropy,
            length_variance,
            suspicious_ratio,
            digit_ratio,
            special_char_ratio,
            avg_subdomain_depth,
            max_subdomain_depth,
            unique_ratio,
            len(tld_counter),  # 唯一TLD数量
            max(tld_counter.values()) if tld_counter else 0,  # 最频繁TLD的出现次数
            min(tld_counter.values()) if tld_counter else 0  # 最少TLD的出现次数
        ], dtype=np.float32)

    def _calculate_time_features(self, query_times):
        """
        计算时间行为特征
        """
        if not query_times:
            return np.zeros(10, dtype=np.float32)

        # 转换为datetime对象
        dt_list = []
        for time_str in query_times:
            try:
                dt = pd.to_datetime(time_str)
                dt_list.append(dt)
            except:
                continue

        if not dt_list:
            return np.zeros(10, dtype=np.float32)

        # 时间统计特征
        hours = [dt.hour for dt in dt_list]
        weekdays = [dt.weekday() for dt in dt_list]

        if not hours:
            return np.zeros(10, dtype=np.float32)

        # 活跃时间段
        hour_counter = Counter(hours)
        active_hours = len(hour_counter)

        # 时间分布熵，避免log(0)
        hour_entropy = 0.0
        if len(hours) > 1:
            for count in hour_counter.values():
                if count > 0:
                    prob = count / len(hours)
                    hour_entropy -= prob * np.log2(prob)

        # 夜间活动比例 (22:00-06:00)
        night_hours = [h for h in hours if h >= 22 or h <= 6]
        night_ratio = len(night_hours) / len(hours) if len(hours) > 0 else 0.0

        # 周末活动比例
        weekend_count = len([w for w in weekdays if w >= 5])
        weekend_ratio = weekend_count / len(weekdays) if len(weekdays) > 0 else 0.0

        # 时间间隔统计
        if len(dt_list) > 1:
            try:
                intervals = [(dt_list[i + 1] - dt_list[i]).total_seconds()
                             for i in range(len(dt_list) - 1)]
                intervals = [abs(interval) for interval in intervals]  # 确保为正数
                avg_interval = np.mean(intervals) if intervals else 0.0
                interval_variance = np.var(intervals) if len(intervals) > 1 else 0.0
            except:
                avg_interval = 0.0
                interval_variance = 0.0
        else:
            avg_interval = 0.0
            interval_variance = 0.0

        # 查询频率特征
        if len(dt_list) > 1:
            try:
                total_time_span = (max(dt_list) - min(dt_list)).total_seconds()
                if total_time_span > 0:
                    query_frequency = len(dt_list) / (total_time_span / 3600)  # 每小时查询次数
                else:
                    query_frequency = 0.0
            except:
                query_frequency = 0.0
        else:
            query_frequency = 0.0

        return np.array([
            active_hours,
            hour_entropy,
            night_ratio,
            weekend_ratio,
            avg_interval,
            interval_variance,
            query_frequency,
            len(set(hours)),  # 唯一小时数
            len(set(weekdays)),  # 唯一星期数
            max(hour_counter.values()) if hour_counter else 0  # 最频繁小时的查询次数
        ], dtype=np.float32)

    def _calculate_network_features(self, server_ips, client_ips=None):
        """
        计算网络行为特征
        """
        if not server_ips:
            return np.zeros(10, dtype=np.float32)

        server_list = list(server_ips)

        # 基础统计
        unique_servers = len(server_list)

        # 服务器IP分布特征
        # 按网段分组
        subnets = set()
        for ip in server_list:
            try:
                network = ipaddress.IPv4Network(f"{ip}/24", strict=False)
                subnets.add(str(network.network_address))
            except:
                continue

        unique_subnets = len(subnets)
        subnet_diversity = unique_subnets / unique_servers if unique_servers > 0 else 0

        # 公网vs私网IP比例
        public_ips = 0
        private_ips = 0
        for ip in server_list:
            try:
                ip_obj = ipaddress.IPv4Address(ip)
                if ip_obj.is_private:
                    private_ips += 1
                else:
                    public_ips += 1
            except:
                continue

        public_ratio = public_ips / (public_ips + private_ips) if (public_ips + private_ips) > 0 else 0

        # 服务器访问频率分布
        server_counter = Counter(server_list)

        # 计算访问熵，避免log(0)
        access_entropy = 0.0
        if len(server_list) > 1:
            for count in server_counter.values():
                if count > 0:
                    prob = count / len(server_list)
                    access_entropy -= prob * np.log2(prob)

        max_access_count = max(server_counter.values()) if server_counter else 0
        min_access_count = min(server_counter.values()) if server_counter else 0

        # 计算统计值，避免空列表
        counter_values = list(server_counter.values()) if server_counter else [0]
        mean_access = np.mean(counter_values)
        var_access = np.var(counter_values) if len(counter_values) > 1 else 0.0

        return np.array([
            unique_servers,
            unique_subnets,
            subnet_diversity,
            public_ratio,
            access_entropy,
            max_access_count,
            min_access_count,
            len(server_counter),  # 唯一服务器数
            mean_access,  # 平均访问次数
            var_access  # 访问次数方差
        ], dtype=np.float32)

    def _preprocess_ip_behaviors(self, df):
        """
        预处理：统计每个IP的查询行为特征
        """
        print("正在分析IP行为模式...")
        ip_behaviors = defaultdict(lambda: {
            'queried_domains': set(),
            'query_count': 0,
            'server_ips': set(),
            'client_ips': set(),
            'config_hits': set(),
            'query_times': [],
            'failed_queries': 0,
            'query_types': [],
            'protocols': []
        })

        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"  处理进度: {idx}/{len(df)}")

            client_ip = str(row['客户端ip地址']).strip()
            server_ip = str(row.get('服务端ip地址', '')).strip()
            domain = str(row['查询内容']).strip()
            config_id = str(row.get('配置ID', '')).strip()
            discover_time = str(row.get('发现时间', '')).strip()
            query_type = str(row.get('查询类型', '')).strip()
            protocol = str(row.get('协议类型', '')).strip()

            # 统计客户端IP行为
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

            # 统计服务端IP行为
            if server_ip and server_ip != 'nan':
                behavior = ip_behaviors[server_ip]
                if client_ip and client_ip != 'nan':
                    behavior['client_ips'].add(client_ip)

        print(f"分析完成，共发现 {len(ip_behaviors)} 个唯一IP")
        return dict(ip_behaviors)

    def _generate_ip_features(self, ip_address, ip_behaviors):
        """
        为IP生成包含行为特征的512维向量
        """
        behavior = ip_behaviors.get(ip_address, {})

        # 1. 基础IP特征 (4维)
        ip_basic = self._ip_to_basic_vector(ip_address, 4)

        # 2. 查询域名特征 (15维)
        domain_features = self._calculate_domain_features(
            behavior.get('queried_domains', set())
        )

        # 3. 时间行为特征 (10维)
        time_features = self._calculate_time_features(
            behavior.get('query_times', [])
        )

        # 4. 网络行为特征 (10维)
        network_features = self._calculate_network_features(
            behavior.get('server_ips', set()),
            behavior.get('client_ips', set())
        )

        # 5. 配置交互特征 (5维)
        config_features = np.array([
            len(behavior.get('config_hits', set())),  # 命中配置数量
            behavior.get('query_count', 0),  # 总查询次数
            behavior.get('failed_queries', 0),  # 失败查询次数
            len(behavior.get('query_types', [])),  # 查询类型数量
            len(behavior.get('protocols', []))  # 协议类型数量
        ], dtype=np.float32)

        # 6. 组合所有特征
        current_dim = (len(ip_basic) + len(domain_features) +
                       len(time_features) + len(network_features) +
                       len(config_features))

        # 7. 补齐到512维
        remaining_dim = self.feature_dim - current_dim
        if remaining_dim > 0:
            # 使用IP地址的哈希向量补齐剩余维度
            hash_vector = self._hash_to_vector(ip_address, seed=42)[:remaining_dim]
        else:
            hash_vector = np.array([], dtype=np.float32)

        # 8. 拼接所有特征
        all_features = np.concatenate([
            ip_basic,
            domain_features,
            time_features,
            network_features,
            config_features,
            hash_vector
        ])

        return all_features[:self.feature_dim]  # 确保不超过指定维度

    def _add_node(self, node_id, node_type, features=None):
        """
        添加节点到图中
        """
        if node_id in self.node_mapping:
            return self.node_mapping[node_id]

        node_idx = len(self.node_mapping)
        self.node_mapping[node_id] = node_idx
        self.node_types[node_id] = self.NODE_TYPES[node_type]

        if features is not None:
            self.node_features.append(features)
        else:
            # 默认特征向量
            self.node_features.append(self._hash_to_vector(node_id))

        return node_idx

    def _add_edge(self, src_node, dst_node, edge_type):
        """
        添加边到图中
        """
        self.edges.append([src_node, dst_node])
        # 对于无向图，添加反向边
        if src_node != dst_node:
            self.edges.append([dst_node, src_node])

    def _calculate_ip_similarity(self, ip1, ip2, ip_behaviors):
        """
        计算两个IP之间的相似度（基于共同查询域名）
        """
        behavior1 = ip_behaviors.get(ip1, {})
        behavior2 = ip_behaviors.get(ip2, {})

        domains1 = behavior1.get('queried_domains', set())
        domains2 = behavior2.get('queried_domains', set())

        if not domains1 or not domains2:
            return 0.0

        # Jaccard相似度
        intersection = len(domains1.intersection(domains2))
        union = len(domains1.union(domains2))

        return intersection / union if union > 0 else 0.0

    def _is_same_subnet(self, ip1, ip2, subnet_mask=24):
        """
        判断两个IP是否在同一子网
        """
        try:
            network1 = ipaddress.IPv4Network(f"{ip1}/{subnet_mask}", strict=False)
            network2 = ipaddress.IPv4Network(f"{ip2}/{subnet_mask}", strict=False)
            return network1.network_address == network2.network_address
        except:
            return False

    def build_graph_from_csv(self, log_file_path, label_file_path=None):
        """
        从日志文件构建图数据集 - 方案A版本
        """
        print("=== DNS图构建器 V2.0 (方案A) ===")
        print("专注于IP异常行为检测的优化图结构")
        print("正在读取日志文件...")

        # 读取主要数据文件
        file_extension = log_file_path.lower().split('.')[-1]

        try:
            if file_extension == 'xlsx':
                print("检测到xlsx格式，使用read_excel读取...")
                df = pd.read_excel(log_file_path, engine='openpyxl')
            elif file_extension == 'csv':
                print("检测到csv格式，使用read_csv读取...")
                try:
                    df = pd.read_csv(log_file_path, encoding='utf-8')
                except:
                    df = pd.read_csv(log_file_path, encoding='gbk')
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
        except Exception as e:
            print(f"读取文件失败: {e}")
            raise

        print(f"读取到 {len(df)} 条DNS查询记录")

        # 读取异常IP标签文件
        abnormal_ips = set()
        if label_file_path:
            try:
                print("正在读取异常IP标签文件...")
                label_df = pd.read_csv(label_file_path, encoding='utf-8')
                abnormal_ips = set(label_df.iloc[:, 0].dropna().astype(str).unique())
                print(f"读取到 {len(abnormal_ips)} 个异常IP")
                print(f"异常IP示例: {list(abnormal_ips)[:5]}")
            except UnicodeDecodeError:
                try:
                    label_df = pd.read_csv(label_file_path, encoding='gbk')
                    abnormal_ips = set(label_df.iloc[:, 0].dropna().astype(str).unique())
                    print(f"读取到 {len(abnormal_ips)} 个异常IP (使用GBK编码)")
                except Exception as e:
                    print(f"读取标签文件失败: {e}")
            except Exception as e:
                print(f"读取标签文件失败: {e}")

        # 数据预处理
        df = df.dropna(subset=['客户端ip地址', '查询内容'])

        # 预处理IP行为
        self.ip_behaviors = self._preprocess_ip_behaviors(df)

        # 收集所有唯一IP
        all_ips = set(self.ip_behaviors.keys())
        print(f"发现 {len(all_ips)} 个唯一IP地址")

        print("正在构建IP节点...")
        ip_nodes = {}

        # 创建IP节点
        for i, ip_address in enumerate(all_ips):
            if i % 1000 == 0:
                print(f"  IP节点创建进度: {i}/{len(all_ips)}")

            ip_features = self._generate_ip_features(ip_address, self.ip_behaviors)
            node_idx = self._add_node(f"ip_{ip_address}", 'ip', ip_features)
            ip_nodes[ip_address] = node_idx

        print("正在构建配置节点...")
        config_nodes = {}
        unique_configs = set()

        for idx, row in df.iterrows():
            config_id = str(row.get('配置ID', '')).strip()
            if config_id and config_id != 'nan':
                unique_configs.add(config_id)

        for config_id in unique_configs:
            config_node = self._add_node(f"config_{config_id}", 'config')
            config_nodes[config_id] = config_node

        print(f"创建了 {len(config_nodes)} 个配置节点")

        print("正在构建边关系...")

        # 1. IP-配置命中关系
        print("  构建IP-配置关系...")
        for idx, row in df.iterrows():
            client_ip = str(row['客户端ip地址']).strip()
            config_id = str(row.get('配置ID', '')).strip()

            if (client_ip in ip_nodes and config_id in config_nodes):
                self._add_edge(ip_nodes[client_ip], config_nodes[config_id], 'hit')

        # 2. IP-IP通信关系
        print("  构建IP通信关系...")
        communication_pairs = set()
        for idx, row in df.iterrows():
            client_ip = str(row['客户端ip地址']).strip()
            server_ip = str(row.get('服务端ip地址', '')).strip()

            if (client_ip in ip_nodes and server_ip in ip_nodes and
                    client_ip != server_ip):
                pair = tuple(sorted([client_ip, server_ip]))
                if pair not in communication_pairs:
                    self._add_edge(ip_nodes[client_ip], ip_nodes[server_ip], 'communication')
                    communication_pairs.add(pair)

        print(f"    创建了 {len(communication_pairs)} 个通信关系")

        # 3. IP-IP相似性关系（基于共同查询域名）
        print("  构建IP相似性关系...")
        ip_list = list(all_ips)
        similarity_threshold = 0.1  # 相似度阈值
        similarity_count = 0

        for i in range(len(ip_list)):
            if i % 500 == 0:
                print(f"    相似性计算进度: {i}/{len(ip_list)}")

            for j in range(i + 1, len(ip_list)):
                similarity = self._calculate_ip_similarity(
                    ip_list[i], ip_list[j], self.ip_behaviors
                )

                if similarity >= similarity_threshold:
                    self._add_edge(ip_nodes[ip_list[i]], ip_nodes[ip_list[j]], 'similarity')
                    similarity_count += 1

        print(f"    创建了 {similarity_count} 个相似性关系")

        # 4. IP-IP同子网关系
        print("  构建同子网关系...")
        subnet_count = 0
        for i in range(len(ip_list)):
            for j in range(i + 1, len(ip_list)):
                if self._is_same_subnet(ip_list[i], ip_list[j]):
                    self._add_edge(ip_nodes[ip_list[i]], ip_nodes[ip_list[j]], 'same_subnet')
                    subnet_count += 1

        print(f"    创建了 {subnet_count} 个同子网关系")

        print("正在生成节点标签...")

        # 生成节点标签
        node_labels = []
        for node_id in self.node_mapping.keys():
            if node_id.startswith('ip_'):
                ip = node_id.split('_', 1)[1]
                if ip in abnormal_ips:
                    node_labels.append(1)  # 异常
                else:
                    node_labels.append(0)  # 正常
            else:
                node_labels.append(0)  # 非IP节点默认为正常

        print("正在构建PyG数据对象...")

        # 转换为PyG格式
        x = torch.tensor(np.array(self.node_features), dtype=torch.float)
        edge_index = torch.tensor(np.array(self.edges).T, dtype=torch.long)
        y = torch.tensor(node_labels, dtype=torch.long)

        # 创建PyG数据对象
        data = Data(x=x, edge_index=edge_index, y=y)

        # 添加额外信息
        data.num_nodes = len(self.node_mapping)
        data.num_edges = len(self.edges)
        data.node_types = torch.tensor([self.node_types[node_id] for node_id in self.node_mapping.keys()],
                                       dtype=torch.long)

        print(f"\n=== 图构建完成 ===")
        print(f"节点数量: {data.num_nodes}")
        print(f"边数量: {data.num_edges}")
        print(f"特征维度: {data.x.shape[1]}")
        print(f"异常节点数量: {sum(node_labels)}")
        print(f"异常节点比例: {sum(node_labels) / len(node_labels) * 100:.2f}%")

        return data

    def save_graph(self, data, output_path):
        """
        保存图数据到文件
        """
        torch.save(data, output_path)
        print(f"图数据已保存到: {output_path}")

    def get_statistics(self, data):
        """
        获取图数据统计信息
        """
        stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges // 2,  # 无向图，除以2
            'feature_dim': data.x.shape[1],
            'num_abnormal': int(data.y.sum()),
            'num_normal': int((data.y == 0).sum()),
            'abnormal_ratio': float(data.y.sum()) / len(data.y),
        }

        # 节点类型统计
        if hasattr(data, 'node_types'):
            type_counts = {}
            for type_id, type_name in self.NODE_TYPES.items():
                count = int((data.node_types == type_name).sum())
                type_counts[type_id] = count
            stats['node_type_counts'] = type_counts

        return stats


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    log_file_path = "./log.xlsx"  # DNS日志文件路径
    label_file_path = "label.csv"  # 异常IP标签文件路径
    output_path = "dns_graph_dataset_v2.pt"  # 输出文件路径

    # 创建图构建器
    builder = DNSGraphBuilderV2(feature_dim=512)

    try:
        # 构建图数据集
        graph_data = builder.build_graph_from_csv(log_file_path, label_file_path)

        # 保存数据集
        builder.save_graph(graph_data, output_path)

        # 打印统计信息
        stats = builder.get_statistics(graph_data)
        print("\n=== 图数据集统计信息 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

        print(f"\n数据集构建完成！可以使用以下代码加载:")
        print(f"import torch")
        print(f"data = torch.load('{output_path}')")
        print(f"print(data)")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()