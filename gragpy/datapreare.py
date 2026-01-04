#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNS查询日志图数据集构建器
从DNS查询日志CSV数据中构建PyTorch Geometric格式的图神经网络数据集

作者: WebSWEAgent
日期: 2025-08-10
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import hashlib
import ipaddress
from datetime import datetime
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 检查openpyxl是否安装
try:
    import openpyxl
except ImportError:
    print("警告: 未安装openpyxl库，无法读取xlsx文件")
    print("请运行: pip install openpyxl")

class DNSGraphBuilder:
    def __init__(self, feature_dim=512):
        """
        初始化DNS图构建器

        Args:
            feature_dim (int): 节点特征维度，默认512
        """
        self.feature_dim = feature_dim
        self.node_mapping = {}  # 节点到索引的映射
        self.node_types = {}    # 节点类型映射
        self.node_features = []  # 节点特征列表
        self.edges = []         # 边列表
        self.labels = []        # 节点标签列表

        # 节点类型定义
        self.NODE_TYPES = {
            'client_ip': 0,
            'server_ip': 1,
            'domain': 2,
            'time': 3,
            'config': 4
        }

        # 关系类型定义
        self.EDGE_TYPES = {
            'query': 0,      # 客户端IP与域名的查询关系
            'hit': 1,        # 客户端IP和配置的命中关系
            'request': 2,    # 客户端IP与服务器IP的请求关系
            'same_host': 3,  # 客户端IP和服务器IP的同主机关系
            'config_rel': 4  # 域名与配置的配置关系
        }

    def _hash_to_vector(self, text, seed=42):
        """
        将文本哈希为固定维度的向量

        Args:
            text (str): 输入文本
            seed (int): 随机种子

        Returns:
            np.ndarray: 特征向量
        """
        # 使用MD5哈希
        hash_obj = hashlib.md5(str(text).encode())
        hash_hex = hash_obj.hexdigest()

        # 将哈希值转换为数字序列
        hash_int = int(hash_hex, 16)

        # 设置随机种子并生成向量
        np.random.seed(hash_int % (2**32) + seed)
        vector = np.random.normal(0, 1, self.feature_dim)

        # 归一化
        vector = vector / np.linalg.norm(vector)

        return vector.astype(np.float32)

    def _ip_to_vector(self, ip_str):
        """
        将IP地址转换为特征向量

        Args:
            ip_str (str): IP地址字符串

        Returns:
            np.ndarray: 特征向量
        """
        try:
            # 尝试解析IPv4地址
            ip = ipaddress.IPv4Address(ip_str)
            ip_int = int(ip)

            # 提取IP地址的各个字节
            bytes_list = [(ip_int >> (8 * i)) & 0xFF for i in range(4)]

            # 使用IP的数值特征作为种子
            seed = sum(bytes_list)

            return self._hash_to_vector(ip_str, seed)

        except:
            # 如果不是有效IP，使用哈希方法
            return self._hash_to_vector(ip_str)

    def _time_to_vector(self, time_str):
        """
        将时间字符串转换为特征向量

        Args:
            time_str (str): 时间字符串

        Returns:
            np.ndarray: 特征向量
        """
        try:
            # 尝试解析时间
            dt = pd.to_datetime(time_str)

            # 提取时间特征
            features = [
                dt.year % 100,  # 年份后两位
                dt.month,       # 月份
                dt.day,         # 日期
                dt.hour,        # 小时
                dt.minute,      # 分钟
                dt.weekday(),   # 星期几
            ]

            seed = sum(features)
            return self._hash_to_vector(time_str, seed)

        except:
            return self._hash_to_vector(time_str)

    def _add_node(self, node_id, node_type, feature_text=None):
        """
        添加节点到图中

        Args:
            node_id (str): 节点ID
            node_type (str): 节点类型
            feature_text (str): 用于生成特征的文本

        Returns:
            int: 节点索引
        """
        if node_id in self.node_mapping:
            return self.node_mapping[node_id]

        # 分配新的节点索引
        node_idx = len(self.node_mapping)
        self.node_mapping[node_id] = node_idx
        self.node_types[node_id] = self.NODE_TYPES[node_type]

        # 生成节点特征向量
        if feature_text is None:
            feature_text = node_id

        if node_type in ['client_ip', 'server_ip']:
            feature_vector = self._ip_to_vector(feature_text)
        elif node_type == 'time':
            feature_vector = self._time_to_vector(feature_text)
        else:
            feature_vector = self._hash_to_vector(feature_text)

        self.node_features.append(feature_vector)

        return node_idx

    def _add_edge(self, src_node, dst_node, edge_type):
        """
        添加边到图中

        Args:
            src_node (int): 源节点索引
            dst_node (int): 目标节点索引
            edge_type (str): 边类型
        """
        self.edges.append([src_node, dst_node])
        # 对于无向图，添加反向边
        if src_node != dst_node:
            self.edges.append([dst_node, src_node])

    def _extract_time_node(self, time_str, granularity='hour'):
        """
        从时间字符串提取时间节点

        Args:
            time_str (str): 时间字符串
            granularity (str): 时间粒度 ('hour', 'day')

        Returns:
            str: 时间节点ID
        """
        try:
            dt = pd.to_datetime(time_str)
            if granularity == 'hour':
                return f"time_{dt.strftime('%Y%m%d_%H')}"
            elif granularity == 'day':
                return f"time_{dt.strftime('%Y%m%d')}"
            else:
                return f"time_{dt.strftime('%Y%m%d_%H%M')}"
        except:
            return f"time_unknown_{hash(time_str) % 10000}"

    def _is_same_subnet(self, ip1, ip2, subnet_mask=24):
        """
        判断两个IP是否在同一子网（用于同主机关系）

        Args:
            ip1 (str): IP地址1
            ip2 (str): IP地址2
            subnet_mask (int): 子网掩码位数

        Returns:
            bool: 是否在同一子网
        """
        try:
            network1 = ipaddress.IPv4Network(f"{ip1}/{subnet_mask}", strict=False)
            network2 = ipaddress.IPv4Network(f"{ip2}/{subnet_mask}", strict=False)
            return network1.network_address == network2.network_address
        except:
            return False

    def build_graph_from_csv(self, log_file_path, label_file_path=None):
        """
        从日志文件构建图数据集

        Args:
            log_file_path (str): DNS日志文件路径（支持xlsx和csv格式）
            label_file_path (str): 异常IP标签CSV文件路径

        Returns:
            torch_geometric.data.Data: PyG图数据对象
        """
        print("正在读取日志文件...")

        # 读取主要数据文件 - 支持xlsx和csv格式
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
                raise ValueError(f"不支持的文件格式: {file_extension}，请使用xlsx或csv格式")
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
                # 直接读取第一列作为异常IP地址列表
                abnormal_ips = set(label_df.iloc[:, 0].dropna().astype(str).unique())
                print(f"读取到 {len(abnormal_ips)} 个异常IP")
                print(f"异常IP示例: {list(abnormal_ips)[:5]}")  # 显示前5个异常IP作为验证
            except UnicodeDecodeError:
                try:
                    # 尝试GBK编码
                    label_df = pd.read_csv(label_file_path, encoding='gbk')
                    abnormal_ips = set(label_df.iloc[:, 0].dropna().astype(str).unique())
                    print(f"读取到 {len(abnormal_ips)} 个异常IP (使用GBK编码)")
                    print(f"异常IP示例: {list(abnormal_ips)[:5]}")
                except Exception as e:
                    print(f"读取标签文件失败 (GBK编码): {e}")
            except Exception as e:
                print(f"读取标签文件失败: {e}")

        # 数据预处理
        df = df.dropna(subset=['客户端ip地址', '查询内容'])

        print("正在构建图节点...")

        # 存储所有IP地址用于同主机关系判断
        all_ips = set()

        # 处理每一行数据
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"处理进度: {idx}/{len(df)}")

            # 提取基本信息
            client_ip = str(row['客户端ip地址']).strip()
            server_ip = str(row.get('服务端ip地址', '')).strip()
            domain = str(row['查询内容']).strip()
            config_id = str(row.get('配置ID', '')).strip()
            discover_time = str(row.get('发现时间', '')).strip()

            # 跳过无效数据
            if not client_ip or client_ip == 'nan':
                continue

            all_ips.add(client_ip)
            if server_ip and server_ip != 'nan':
                all_ips.add(server_ip)

            # 添加客户端IP节点
            client_node = self._add_node(f"client_{client_ip}", 'client_ip', client_ip)

            # 添加域名节点
            if domain and domain != 'nan':
                domain_node = self._add_node(f"domain_{domain}", 'domain', domain)
                # 添加查询关系
                self._add_edge(client_node, domain_node, 'query')

            # 添加服务端IP节点
            if server_ip and server_ip != 'nan':
                server_node = self._add_node(f"server_{server_ip}", 'server_ip', server_ip)
                # 添加请求关系
                self._add_edge(client_node, server_node, 'request')

            # 添加配置节点
            if config_id and config_id != 'nan':
                config_node = self._add_node(f"config_{config_id}", 'config', config_id)
                # 添加命中关系
                self._add_edge(client_node, config_node, 'hit')

                # 添加域名与配置的关系
                if domain and domain != 'nan':
                    self._add_edge(domain_node, config_node, 'config_rel')

            # 添加时间节点
            if discover_time and discover_time != 'nan':
                time_node_id = self._extract_time_node(discover_time)
                time_node = self._add_node(time_node_id, 'time', discover_time)
                # 可以添加时间相关的边（这里暂时跳过，避免图过于复杂）

        print("正在添加同主机关系...")

        # 添加同主机关系（基于子网）
        ip_list = list(all_ips)
        for i in range(len(ip_list)):
            for j in range(i + 1, len(ip_list)):
                if self._is_same_subnet(ip_list[i], ip_list[j]):
                    # 查找对应的节点
                    client_id1 = f"client_{ip_list[i]}"
                    client_id2 = f"client_{ip_list[j]}"
                    server_id1 = f"server_{ip_list[i]}"
                    server_id2 = f"server_{ip_list[j]}"

                    # 添加同主机关系
                    if client_id1 in self.node_mapping and client_id2 in self.node_mapping:
                        self._add_edge(self.node_mapping[client_id1],
                                     self.node_mapping[client_id2], 'same_host')

                    if server_id1 in self.node_mapping and server_id2 in self.node_mapping:
                        self._add_edge(self.node_mapping[server_id1],
                                     self.node_mapping[server_id2], 'same_host')

        print("正在生成节点标签...")

        # 生成节点标签
        node_labels = []
        for node_id in self.node_mapping.keys():
            # 提取IP地址
            if node_id.startswith('client_') or node_id.startswith('server_'):
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
        data.node_types = torch.tensor([self.node_types[node_id] for node_id in self.node_mapping.keys()], dtype=torch.long)

        print(f"图构建完成!")
        print(f"节点数量: {data.num_nodes}")
        print(f"边数量: {data.num_edges}")
        print(f"特征维度: {data.x.shape[1]}")
        print(f"异常节点数量: {sum(node_labels)}")

        return data

    def save_graph(self, data, output_path):
        """
        保存图数据到文件

        Args:
            data: PyG数据对象
            output_path (str): 输出文件路径
        """
        torch.save(data, output_path)
        print(f"图数据已保存到: {output_path}")

    def get_statistics(self, data):
        """
        获取图数据统计信息

        Args:
            data: PyG数据对象

        Returns:
            dict: 统计信息
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
    csv_file_path = "./log.xlsx"  # DNS日志文件路径
    label_file_path = "label.csv"  # 异常IP标签文件路径
    output_path = "dns_graph_dataset_test.pt"  # 输出文件路径

    # 创建图构建器
    builder = DNSGraphBuilder(feature_dim=512)

    try:
        # 构建图数据集
        graph_data = builder.build_graph_from_csv(csv_file_path, label_file_path)

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
        print("请确保CSV文件路径正确")
    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()