import pandas as pd
import numpy as np
import itertools
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Any
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BubbleSortNetwork:
    """冒泡排序网络类"""

    def __init__(self, n: int):
        """
        初始化n维冒泡排序网络

        Args:
            n: 排列的长度
        """
        self.n = n
        self.nodes = self._generate_nodes()
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.adj_list = self._generate_adjacency_list()
        self.edges = self._generate_edges()

    def _generate_nodes(self) -> List[str]:
        """生成所有排列节点"""
        numbers = list(range(1, self.n + 1))
        permutations = [''.join(map(str, p)) for p in itertools.permutations(numbers)]
        return permutations

    def _generate_adjacency_list(self) -> Dict[str, List[str]]:
        """生成邻接表：交换相邻位置的排列"""
        adj_list = {}
        for node in self.nodes:
            neighbors = []
            for i in range(self.n - 1):
                # 交换位置i和i+1
                neighbor = list(node)
                neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
                neighbors.append(''.join(neighbor))
            adj_list[node] = neighbors
        return adj_list

    def get_neighbors(self, node: str) -> List[str]:
        """获取节点的邻居"""
        return self.adj_list.get(node, [])

    def _generate_edges(self) -> List[Tuple[str, str]]:
        """生成所有边"""
        edges = []
        edge_set = set()
        for node, neighbors in self.adj_list.items():
            for neighbor in neighbors:
                edge_tuple = tuple(sorted([node, neighbor]))
                if edge_tuple not in edge_set:
                    edges.append((node, neighbor))
                    edge_set.add(edge_tuple)
        return edges

    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        return {
            "维度": self.n,
            "节点数": len(self.nodes),
            "边数": len(self.edges),
            "平均度": self.n - 1,  # 冒泡排序网络是(n-1)-正则图
            "是否为正则图": True,
            "正则度": self.n - 1
        }

    def visualize_network(self, highlight_nodes: Set[str] = None, save_path: str = None):
        """可视化网络（使用简化的表示）"""
        if len(self.nodes) > 100:
            print("警告：网络过大，可视化可能不清晰")

        # 创建图形
        plt.figure(figsize=(12, 10))

        if len(self.nodes) <= 30:
            # 对于小型网络，使用力导向布局
            try:
                import networkx as nx
                G = nx.Graph()
                G.add_nodes_from(self.nodes)
                G.add_edges_from(self.edges)

                # 使用spring布局
                pos = nx.spring_layout(G, seed=42)

                # 绘制节点
                node_colors = []
                for node in self.nodes:
                    if highlight_nodes and node in highlight_nodes:
                        node_colors.append('red')  # 故障节点
                    else:
                        node_colors.append('lightblue')

                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
                nx.draw_networkx_edges(G, pos, alpha=0.5)
                nx.draw_networkx_labels(G, pos, font_size=8)

                plt.title(f"{self.n}维冒泡排序网络 (节点数: {len(self.nodes)}, 边数: {len(self.edges)})")
                plt.axis('off')
            except ImportError:
                print("警告：未安装networkx库，无法进行详细可视化")
                self._simple_visualization(highlight_nodes)

        else:
            # 对于大型网络，显示度分布
            degrees = [len(self.adj_list[node]) for node in self.nodes]
            plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('节点度')
            plt.ylabel('频率')
            plt.title(f"{self.n}维冒泡排序网络度分布 (平均度: {np.mean(degrees):.2f})")
            plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"网络可视化已保存到: {save_path}")

        plt.tight_layout()
        plt.show()

    def _simple_visualization(self, highlight_nodes: Set[str] = None):
        """简单可视化（当networkx不可用时）"""
        print(f"{self.n}维冒泡排序网络")
        print(f"节点数: {len(self.nodes)}")
        print(f"边数: {len(self.edges)}")
        print(f"故障节点: {highlight_nodes if highlight_nodes else '无'}")


class PMCModel:
    """PMC模型实现"""

    def __init__(self, network):
        self.network = network

    def generate_symptoms(self, fault_nodes: Set[str]) -> Dict[Tuple[str, str], int]:
        """
        根据PMC模型生成症状集

        PMC规则：
        - 如果u和v都正常: σ(u,v)=0
        - 如果u正常但v故障: σ(u,v)=1
        - 如果u故障: σ(u,v)不可靠(0或1随机)
        """
        symptoms = {}

        for node in self.network.nodes:
            for neighbor in self.network.get_neighbors(node):
                u, v = node, neighbor

                if (v, u) in symptoms:  # 避免重复测试
                    continue

                u_fault = u in fault_nodes
                v_fault = v in fault_nodes

                if not u_fault and not v_fault:
                    # 两者都正常
                    symptoms[(u, v)] = 0
                    symptoms[(v, u)] = 0
                elif not u_fault and v_fault:
                    # u正常，v故障
                    symptoms[(u, v)] = 1
                    # v故障，测试结果不可靠
                    symptoms[(v, u)] = random.randint(0, 1)
                elif u_fault and not v_fault:
                    # u故障，测试结果不可靠
                    symptoms[(u, v)] = random.randint(0, 1)
                    # v正常，u故障
                    symptoms[(v, u)] = 1
                else:
                    # 两者都故障，测试结果都不可靠
                    symptoms[(u, v)] = random.randint(0, 1)
                    symptoms[(v, u)] = random.randint(0, 1)

        return symptoms

    def save_symptoms_to_excel(self, symptoms: Dict[Tuple[str, str], int], filepath: str):
        """保存症状集到Excel"""
        data = []
        for (u, v), sigma in symptoms.items():
            data.append({'测试者': u, '被测试者': v, '测试结果': sigma})
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False)

    def load_symptoms_from_excel(self, filepath: str) -> Dict[Tuple[str, str], int]:
        """从Excel文件加载症状集"""
        try:
            df = pd.read_excel(filepath)
            symptoms = {}
            for _, row in df.iterrows():
                u = str(row['测试者'])
                v = str(row['被测试者'])
                sigma = int(row['测试结果'])
                symptoms[(u, v)] = sigma
            return symptoms
        except Exception as e:
            print(f"加载症状集失败: {e}")
            return {}


class TRFI_PMC:
    """三轮故障识别算法"""

    def __init__(self, network, symptoms: Dict[Tuple[str, str], int]):
        self.network = network
        self.symptoms = symptoms
        self.groups = []  # 存储分组的节点列表
        self.group_of_node = {}  # 节点到组的映射
        self.faulty_groups = set()  # 故障组索引
        self.fault_free_groups = set()  # 正常组索引
        self.unclassified_groups = set()  # 未分类组索引

    def _is_00_edge(self, u: str, v: str) -> bool:
        """检查是否为(0,0)边"""
        if (u, v) in self.symptoms and (v, u) in self.symptoms:
            return self.symptoms[(u, v)] == 0 and self.symptoms[(v, u)] == 0
        return False

    def divide_groups(self):
        """分组阶段：基于(0,0)边进行分组"""
        visited = set()
        self.groups = []
        self.group_of_node = {}

        for node in self.network.nodes:
            if node in visited:
                continue

            # 开始新的组
            group = []
            queue = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue

                visited.add(current)
                group.append(current)
                self.group_of_node[current] = len(self.groups)

                # 检查邻居
                for neighbor in self.network.get_neighbors(current):
                    if neighbor not in visited and self._is_00_edge(current, neighbor):
                        queue.append(neighbor)

            if group:  # 添加非空组
                self.groups.append(group)

        # 初始化组状态
        self.unclassified_groups = set(range(len(self.groups)))
        self.faulty_groups = set()
        self.fault_free_groups = set()

    def first_round_detection(self, t: int):
        """第一轮检测：根据(0,1)或(1,0)边识别故障组"""
        to_remove = set()

        for group_idx in self.unclassified_groups:
            group = self.groups[group_idx]

            for node in group:
                for neighbor in self.network.get_neighbors(node):
                    # 检查邻居是否在同一组
                    if neighbor in group:
                        continue

                    # 获取测试结果
                    sigma_uv = self.symptoms.get((node, neighbor), -1)
                    sigma_vu = self.symptoms.get((neighbor, node), -1)

                    if sigma_uv == 0 and sigma_vu == 1:
                        # 根据Lemma 2，node是故障的
                        self.faulty_groups.add(group_idx)
                        to_remove.add(group_idx)
                        break
                    elif sigma_uv == 1 and sigma_vu == 0:
                        # 根据Lemma 2，neighbor是故障的，但neighbor可能在其他组
                        neighbor_group = self.group_of_node.get(neighbor)
                        if neighbor_group is not None:
                            self.faulty_groups.add(neighbor_group)
                            to_remove.add(neighbor_group)
                        break
                if group_idx in to_remove:
                    break

        # 更新未分类组
        self.unclassified_groups -= to_remove

    def second_round_detection(self, t: int):
        """第二轮检测：基于组大小判断"""
        to_remove = set()

        for group_idx in list(self.unclassified_groups):
            group_size = len(self.groups[group_idx])
            remaining_faults = t - len(self.faulty_groups)

            if group_size > remaining_faults:
                # 组大小超过剩余可容忍故障数，认为是正常组
                self.fault_free_groups.add(group_idx)
                to_remove.add(group_idx)

        self.unclassified_groups -= to_remove

    def third_round_detection(self, t: int):
        """第三轮检测：基于邻居组判断"""
        to_remove = set()

        for group_idx in list(self.unclassified_groups):
            # 获取未分类邻居组的数量
            unclassified_neighbor_count = 0

            for node in self.groups[group_idx]:
                for neighbor in self.network.get_neighbors(node):
                    neighbor_group = self.group_of_node.get(neighbor)
                    if (neighbor_group is not None and
                            neighbor_group in self.unclassified_groups and
                            neighbor_group != group_idx):
                        unclassified_neighbor_count += 1

            remaining_faults = t - len(self.faulty_groups)

            if unclassified_neighbor_count > remaining_faults:
                # 未分类邻居组数量超过剩余可容忍故障数，认为是故障组
                self.faulty_groups.add(group_idx)
                to_remove.add(group_idx)

        self.unclassified_groups -= to_remove

    def final_classification(self):
        """最终分类：剩余未分类组标记为正常"""
        self.fault_free_groups.update(self.unclassified_groups)
        self.unclassified_groups.clear()

    def get_faulty_nodes(self) -> List[str]:
        """获取诊断出的故障节点"""
        faulty_nodes = []
        for group_idx in self.faulty_groups:
            faulty_nodes.extend(self.groups[group_idx])
        return faulty_nodes

    def get_fault_free_nodes(self) -> List[str]:
        """获取诊断出的正常节点"""
        fault_free_nodes = []
        for group_idx in self.fault_free_groups:
            fault_free_nodes.extend(self.groups[group_idx])
        return fault_free_nodes

    def run(self, t: int) -> List[str]:
        """
        运行TRFI-PMC算法

        Args:
            t: 诊断度（最大可容忍故障数）

        Returns:
            诊断出的故障节点列表
        """
        self.divide_groups()
        self.first_round_detection(t)
        self.second_round_detection(t)
        self.third_round_detection(t)
        self.final_classification()
        return self.get_faulty_nodes()


class GeneralNetwork:
    """通用网络类（用于功能2）"""

    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]]):
        self.nodes = nodes
        self.edges = edges
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(nodes)}
        self.adj_list = self._generate_adjacency_list()

    def _generate_adjacency_list(self) -> Dict[str, List[str]]:
        """生成邻接表"""
        adj_list = {node: [] for node in self.nodes}
        for u, v in self.edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list

    def get_neighbors(self, node: str) -> List[str]:
        """获取节点的邻居"""
        return self.adj_list.get(node, [])

    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        degrees = [len(neighbors) for neighbors in self.adj_list.values()]
        return {
            "节点数": len(self.nodes),
            "边数": len(self.edges),
            "平均度": np.mean(degrees) if degrees else 0,
            "最大度": max(degrees) if degrees else 0,
            "最小度": min(degrees) if degrees else 0,
        }


class Evaluator:
    """评估器：计算各种性能指标"""

    def __init__(self, true_faulty: Set[str], predicted_faulty: Set[str], all_nodes: List[str]):
        """
        初始化评估器

        Args:
            true_faulty: 真实的故障节点集合
            predicted_faulty: 预测的故障节点集合
            all_nodes: 所有节点列表
        """
        self.true_faulty = true_faulty
        self.predicted_faulty = predicted_faulty
        self.all_nodes = set(all_nodes)

        # 计算基本统计量
        self.TP = len(true_faulty & predicted_faulty)  # 真正例
        self.FP = len(predicted_faulty - true_faulty)  # 假正例
        self.TN = len((self.all_nodes - true_faulty) - predicted_faulty)  # 真负例
        self.FN = len(true_faulty - predicted_faulty)  # 假负例

    def calculate_metrics(self) -> Dict[str, float]:
        """计算所有评估指标"""
        metrics = {}

        # 准确率
        if self.TP + self.TN + self.FP + self.FN > 0:
            metrics['准确率'] = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        else:
            metrics['准确率'] = 0.0

        # 真负率
        if self.TN + self.FP > 0:
            metrics['真负率'] = self.TN / (self.TN + self.FP)
        else:
            metrics['真负率'] = 0.0

        # 假正率
        if self.TN + self.FP > 0:
            metrics['假正率'] = self.FP / (self.TN + self.FP)
        else:
            metrics['假正率'] = 0.0

        # 真正率（召回率）
        if self.TP + self.FN > 0:
            metrics['真正率'] = self.TP / (self.TP + self.FN)
        else:
            metrics['真正率'] = 0.0

        # 精确率
        if self.TP + self.FP > 0:
            metrics['精确率'] = self.TP / (self.TP + self.FP)
        else:
            metrics['精确率'] = 0.0

        # F1分数
        if metrics['精确率'] + metrics['真正率'] > 0:
            metrics['F1分数'] = 2 * (metrics['精确率'] * metrics['真正率']) / (metrics['精确率'] + metrics['真正率'])
        else:
            metrics['F1分数'] = 0.0

        return metrics

    def plot_confusion_matrix(self, save_path: str = None):
        """绘制混淆矩阵"""
        confusion_matrix = np.array([
            [self.TP, self.FN],
            [self.FP, self.TN]
        ])

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测故障', '预测正常'],
                    yticklabels=['实际故障', '实际正常'])
        plt.title('混淆矩阵')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_metrics_radar(self, metrics: Dict[str, float], save_path: str = None):
        """绘制指标雷达图"""
        categories = list(metrics.keys())
        values = list(metrics.values())

        # 雷达图需要闭合
        categories += [categories[0]]
        values += [values[0]]

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True).tolist()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('性能指标雷达图')
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()


class TRFI_PMC_System:
    """TRFI-PMC诊断系统"""

    def __init__(self):
        self.network = None
        self.pmc_model = None
        self.diagnoser = None
        self.true_faulty_nodes = set()
        self.diagnosed_faulty_nodes = set()
        self.metrics_history = []  # 存储多次实验的指标

    def function1(self, n: int, num_faults: int, num_experiments: int = 1) -> Dict[str, Any]:
        """
        功能1：合成网络诊断

        Args:
            n: 网络维度
            num_faults: 故障节点数
            num_experiments: 实验次数

        Returns:
            包含结果的字典
        """
        print(f"功能1：合成网络诊断")
        print(f"网络维度: {n}, 故障节点数: {num_faults}, 实验次数: {num_experiments}")

        self.metrics_history = []
        all_results = []

        for exp in range(num_experiments):
            print(f"\n实验 {exp + 1}/{num_experiments}")

            # 创建冒泡排序网络
            self.network = BubbleSortNetwork(n)
            network_info = self.network.get_network_info()

            # 创建PMC模型
            self.pmc_model = PMCModel(self.network)

            # 随机选择故障节点
            if num_faults > len(self.network.nodes):
                print(f"警告：故障节点数超过总节点数，将设置为总节点数的10%")
                num_faults = max(1, int(len(self.network.nodes) * 0.1))

            self.true_faulty_nodes = set(random.sample(self.network.nodes, num_faults))

            # 生成症状集
            symptoms = self.pmc_model.generate_symptoms(self.true_faulty_nodes)

            # 运行诊断
            self.diagnoser = TRFI_PMC(self.network, symptoms)
            diagnosed_nodes = self.diagnoser.run(num_faults)
            self.diagnosed_faulty_nodes = set(diagnosed_nodes)

            # 评估结果
            evaluator = Evaluator(self.true_faulty_nodes, self.diagnosed_faulty_nodes, self.network.nodes)
            metrics = evaluator.calculate_metrics()

            # 保存结果
            result = {
                "实验编号": exp + 1,
                "网络信息": network_info,
                "真实故障节点": list(self.true_faulty_nodes),
                "诊断故障节点": list(self.diagnosed_faulty_nodes),
                "指标": metrics
            }
            all_results.append(result)
            self.metrics_history.append(metrics)

            # 显示本次实验指标
            print(f"准确率: {metrics['准确率'] * 100:.2f}%")
            print(f"真正率: {metrics['真正率'] * 100:.2f}%")
            print(f"精确率: {metrics['精确率'] * 100:.2f}%")

        # 计算平均指标（多次实验时）
        avg_metrics = {}
        if num_experiments > 1:
            print(f"\n{'=' * 60}")
            print("平均性能指标:")
            print(f"{'=' * 60}")
            for key in self.metrics_history[0].keys():
                avg_value = np.mean([m[key] for m in self.metrics_history])
                avg_metrics[key] = avg_value
                print(f"{key}: {avg_value * 100:.2f}%")

        # 保存数据
        self._save_function1_data(all_results, n, num_faults)

        return {
            "实验结果": all_results,
            "平均指标": avg_metrics if avg_metrics else self.metrics_history[0],
            "网络信息": network_info
        }

    def function2(self, nodes: List[str], edges: List[Tuple[str, str]], symptoms: Dict[Tuple[str, str], int]) -> Dict[
        str, Any]:
        """
        功能2：真实网络诊断（输入症状集）

        Args:
            nodes: 节点列表
            edges: 边列表
            symptoms: 症状集

        Returns:
            包含结果的字典
        """
        print(f"功能2：真实网络诊断")

        # 创建通用网络
        self.network = GeneralNetwork(nodes, edges)
        network_info = self.network.get_network_info()

        # 由于功能2没有提供真实故障节点，我们无法计算指标
        # 但我们可以运行诊断算法
        print(f"网络信息: 节点数={network_info['节点数']}, 边数={network_info['边数']}")

        # 运行诊断（使用节点数作为诊断度t的估计）
        t = min(10, len(nodes) // 10)  # 简单估计
        self.diagnoser = TRFI_PMC(self.network, symptoms)
        diagnosed_nodes = self.diagnoser.run(t)
        self.diagnosed_faulty_nodes = set(diagnosed_nodes)

        print(f"诊断出 {len(self.diagnosed_faulty_nodes)} 个故障节点")

        # 保存结果
        self._save_function2_data(nodes, edges, symptoms, diagnosed_nodes)

        return {
            "网络信息": network_info,
            "诊断故障节点": list(self.diagnosed_faulty_nodes),
            "诊断度t": t
        }

    def function3(self, nodes: List[str], edges: List[Tuple[str, str]], n: int, num_faults: int) -> Dict[str, Any]:
        """
        功能3：真实网络诊断（生成症状集）

        Args:
            nodes: 节点列表
            edges: 边列表
            n: 网络维度（用于PMC模型）
            num_faults: 故障节点数

        Returns:
            包含结果的字典
        """
        print(f"功能3：真实网络诊断（生成症状集）")

        # 创建通用网络
        self.network = GeneralNetwork(nodes, edges)
        network_info = self.network.get_network_info()

        # 创建PMC模型
        self.pmc_model = PMCModel(self.network)

        # 随机选择故障节点
        if num_faults > len(nodes):
            print(f"警告：故障节点数超过总节点数，将设置为总节点数的10%")
            num_faults = max(1, int(len(nodes) * 0.1))

        self.true_faulty_nodes = set(random.sample(nodes, num_faults))

        # 生成症状集
        symptoms = self.pmc_model.generate_symptoms(self.true_faulty_nodes)

        # 运行诊断
        self.diagnoser = TRFI_PMC(self.network, symptoms)
        diagnosed_nodes = self.diagnoser.run(num_faults)
        self.diagnosed_faulty_nodes = set(diagnosed_nodes)

        # 评估结果
        evaluator = Evaluator(self.true_faulty_nodes, self.diagnosed_faulty_nodes, nodes)
        metrics = evaluator.calculate_metrics()

        print(f"网络信息: 节点数={network_info['节点数']}, 边数={network_info['边数']}")
        print(f"准确率: {metrics['准确率'] * 100:.2f}%")
        print(f"真正率: {metrics['真正率'] * 100:.2f}%")
        print(f"精确率: {metrics['精确率'] * 100:.2f}%")

        # 保存结果
        self._save_function3_data(nodes, edges, symptoms, diagnosed_nodes, self.true_faulty_nodes, metrics)

        return {
            "网络信息": network_info,
            "真实故障节点": list(self.true_faulty_nodes),
            "诊断故障节点": list(self.diagnosed_faulty_nodes),
            "指标": metrics
        }

    def _save_function1_data(self, results: List[Dict], n: int, num_faults: int):
        """保存功能1的数据"""
        os.makedirs("results/function1", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存网络信息
        network_info = results[0]["网络信息"]
        info_df = pd.DataFrame(list(network_info.items()), columns=["属性", "值"])
        info_df.to_excel(f"results/function1/network_info_{timestamp}.xlsx", index=False)

        # 保存所有实验结果
        all_data = []
        for result in results:
            exp_data = {
                "实验编号": result["实验编号"],
                "准确率": result["指标"]["准确率"],
                "真正率": result["指标"]["真正率"],
                "精确率": result["指标"]["精确率"],
                "真负率": result["指标"]["真负率"],
                "假正率": result["指标"]["假正率"],
                "F1分数": result["指标"]["F1分数"]
            }
            all_data.append(exp_data)

        metrics_df = pd.DataFrame(all_data)
        metrics_df.to_excel(f"results/function1/metrics_{timestamp}.xlsx", index=False)

        # 保存最后一次实验的详细数据
        last_result = results[-1]
        faults_df = pd.DataFrame({
            "真实故障节点": list(last_result["真实故障节点"]),
            "诊断故障节点": list(last_result["诊断故障节点"])
        })
        faults_df.to_excel(f"results/function1/faults_{timestamp}.xlsx", index=False)

        print(f"数据已保存到 results/function1/ 目录")

    def _save_function2_data(self, nodes: List[str], edges: List[Tuple[str, str]],
                             symptoms: Dict[Tuple[str, str], int], diagnosed_nodes: List[str]):
        """保存功能2的数据"""
        os.makedirs("results/function2", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存网络数据
        nodes_df = pd.DataFrame({"节点": nodes})
        nodes_df.to_excel(f"results/function2/nodes_{timestamp}.xlsx", index=False)

        edges_df = pd.DataFrame(edges, columns=["起点", "终点"])
        edges_df.to_excel(f"results/function2/edges_{timestamp}.xlsx", index=False)

        # 保存症状集
        symptoms_df = pd.DataFrame([
            {"测试者": u, "被测试者": v, "测试结果": sigma}
            for (u, v), sigma in symptoms.items()
        ])
        symptoms_df.to_excel(f"results/function2/symptoms_{timestamp}.xlsx", index=False)

        # 保存诊断结果
        diagnosed_df = pd.DataFrame({"诊断故障节点": diagnosed_nodes})
        diagnosed_df.to_excel(f"results/function2/diagnosed_faults_{timestamp}.xlsx", index=False)

        print(f"数据已保存到 results/function2/ 目录")

    def _save_function3_data(self, nodes: List[str], edges: List[Tuple[str, str]],
                             symptoms: Dict[Tuple[str, str], int], diagnosed_nodes: List[str],
                             true_faults: Set[str], metrics: Dict[str, float]):
        """保存功能3的数据"""
        os.makedirs("results/function3", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存网络数据
        nodes_df = pd.DataFrame({"节点": nodes})
        nodes_df.to_excel(f"results/function3/nodes_{timestamp}.xlsx", index=False)

        edges_df = pd.DataFrame(edges, columns=["起点", "终点"])
        edges_df.to_excel(f"results/function3/edges_{timestamp}.xlsx", index=False)

        # 保存症状集
        symptoms_df = pd.DataFrame([
            {"测试者": u, "被测试者": v, "测试结果": sigma}
            for (u, v), sigma in symptoms.items()
        ])
        symptoms_df.to_excel(f"results/function3/symptoms_{timestamp}.xlsx", index=False)

        # 保存故障数据
        faults_df = pd.DataFrame({
            "真实故障节点": list(true_faults),
            "诊断故障节点": list(diagnosed_nodes)
        })
        faults_df.to_excel(f"results/function3/faults_{timestamp}.xlsx", index=False)

        # 保存指标
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["指标", "值"])
        metrics_df.to_excel(f"results/function3/metrics_{timestamp}.xlsx", index=False)

        print(f"数据已保存到 results/function3/ 目录")

    def plot_experiment_results(self, save_path: str = None):
        """绘制多次实验的结果"""
        if not self.metrics_history or len(self.metrics_history) <= 1:
            print("需要多次实验数据才能绘制结果")
            return

        # 提取指标
        experiments = list(range(1, len(self.metrics_history) + 1))
        accuracy = [m['准确率'] for m in self.metrics_history]
        recall = [m['真正率'] for m in self.metrics_history]
        precision = [m['精确率'] for m in self.metrics_history]

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 折线图
        axes[0, 0].plot(experiments, accuracy, 'o-', label='准确率', linewidth=2, markersize=8)
        axes[0, 0].plot(experiments, recall, 's-', label='真正率', linewidth=2, markersize=8)
        axes[0, 0].plot(experiments, precision, '^-', label='精确率', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('实验编号')
        axes[0, 0].set_ylabel('指标值')
        axes[0, 0].set_title('多次实验性能指标变化')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 箱线图
        data_to_plot = [accuracy, recall, precision]
        axes[0, 1].boxplot(data_to_plot, labels=['准确率', '真正率', '精确率'])
        axes[0, 1].set_ylabel('指标值')
        axes[0, 1].set_title('指标分布箱线图')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 条形图（平均指标）
        avg_accuracy = np.mean(accuracy)
        avg_recall = np.mean(recall)
        avg_precision = np.mean(precision)

        categories = ['准确率', '真正率', '精确率']
        values = [avg_accuracy, avg_recall, avg_precision]

        axes[1, 0].bar(categories, values, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('平均指标值')
        axes[1, 0].set_title('平均性能指标')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 在条形上添加数值
        for i, v in enumerate(values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 4. 散点图（准确率 vs 精确率）
        axes[1, 1].scatter(accuracy, precision, c=experiments, cmap='viridis', s=100, alpha=0.7)
        axes[1, 1].set_xlabel('准确率')
        axes[1, 1].set_ylabel('精确率')
        axes[1, 1].set_title('准确率 vs 精确率')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加颜色条
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='实验编号')

        plt.suptitle('TRFI-PMC算法多次实验性能分析', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"实验分析图已保存到: {save_path}")

        plt.show()


class TRFI_PMC_GUI:
    """TRFI-PMC系统GUI界面"""

    def __init__(self):
        self.system = TRFI_PMC_System()
        self.current_result = None

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("冒泡排序网络故障诊断系统 (TRFI-PMC)")
        self.root.geometry("1000x700")

        # 创建样式
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))

        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # 创建标题
        title_label = ttk.Label(self.main_frame, text="冒泡排序网络故障诊断系统", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # 创建功能选择区域
        self._create_function_selection()

        # 创建结果显示区域
        self._create_result_display()

        # 创建状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def _create_function_selection(self):
        """创建功能选择区域"""
        # 功能选择框架
        func_frame = ttk.LabelFrame(self.main_frame, text="功能选择", padding="10")
        func_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # 功能1：合成网络诊断
        func1_frame = ttk.Frame(func_frame)
        func1_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(func1_frame, text="功能1: 合成网络诊断", style='Subtitle.TLabel').grid(row=0, column=0, columnspan=2,
                                                                                         sticky=tk.W)

        ttk.Label(func1_frame, text="网络维度 n:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.n_var = tk.StringVar(value="4")
        n_entry = ttk.Entry(func1_frame, textvariable=self.n_var, width=10)
        n_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(func1_frame, text="故障节点数:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.faults_var1 = tk.StringVar(value="5")
        faults_entry1 = ttk.Entry(func1_frame, textvariable=self.faults_var1, width=10)
        faults_entry1.grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(func1_frame, text="实验次数:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.exp_var = tk.StringVar(value="1")
        exp_entry = ttk.Entry(func1_frame, textvariable=self.exp_var, width=10)
        exp_entry.grid(row=3, column=1, sticky=tk.W, padx=5)

        func1_btn = ttk.Button(func1_frame, text="运行功能1", command=self.run_function1)
        func1_btn.grid(row=4, column=0, columnspan=2, pady=10)

        # 功能2：真实网络诊断（输入症状集）
        func2_frame = ttk.Frame(func_frame)
        func2_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(func2_frame, text="功能2: 真实网络诊断（输入症状集）", style='Subtitle.TLabel').grid(row=0, column=0,
                                                                                                     columnspan=2,
                                                                                                     sticky=tk.W)

        # 文件选择按钮
        ttk.Label(func2_frame, text="节点文件:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.nodes_file_var = tk.StringVar()
        ttk.Entry(func2_frame, textvariable=self.nodes_file_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_nodes_file).grid(row=1, column=2, padx=5)

        ttk.Label(func2_frame, text="边文件:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.edges_file_var = tk.StringVar()
        ttk.Entry(func2_frame, textvariable=self.edges_file_var, width=30).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_edges_file).grid(row=2, column=2, padx=5)

        ttk.Label(func2_frame, text="症状集文件:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.symptoms_file_var = tk.StringVar()
        ttk.Entry(func2_frame, textvariable=self.symptoms_file_var, width=30).grid(row=3, column=1, sticky=tk.W, padx=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_symptoms_file).grid(row=3, column=2, padx=5)

        func2_btn = ttk.Button(func2_frame, text="运行功能2", command=self.run_function2)
        func2_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # 功能3：真实网络诊断（生成症状集）
        func3_frame = ttk.Frame(func_frame)
        func3_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(func3_frame, text="功能3: 真实网络诊断（生成症状集）", style='Subtitle.TLabel').grid(row=0, column=0,
                                                                                                     columnspan=2,
                                                                                                     sticky=tk.W)

        ttk.Label(func3_frame, text="节点文件:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.nodes_file_var3 = tk.StringVar()
        ttk.Entry(func3_frame, textvariable=self.nodes_file_var3, width=30).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(func3_frame, text="浏览", command=self.select_nodes_file3).grid(row=1, column=2, padx=5)

        ttk.Label(func3_frame, text="边文件:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.edges_file_var3 = tk.StringVar()
        ttk.Entry(func3_frame, textvariable=self.edges_file_var3, width=30).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(func3_frame, text="浏览", command=self.select_edges_file3).grid(row=2, column=2, padx=5)

        ttk.Label(func3_frame, text="网络维度 n:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.n_var3 = tk.StringVar(value="4")
        ttk.Entry(func3_frame, textvariable=self.n_var3, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)

        ttk.Label(func3_frame, text="故障节点数:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.faults_var3 = tk.StringVar(value="10")
        ttk.Entry(func3_frame, textvariable=self.faults_var3, width=10).grid(row=4, column=1, sticky=tk.W, padx=5)

        func3_btn = ttk.Button(func3_frame, text="运行功能3", command=self.run_function3)
        func3_btn.grid(row=5, column=0, columnspan=3, pady=10)

        # 可视化按钮
        viz_frame = ttk.Frame(func_frame)
        viz_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(viz_frame, text="可视化选项:", style='Subtitle.TLabel').grid(row=0, column=0, columnspan=2,
                                                                               sticky=tk.W)

        self.plot_btn = ttk.Button(viz_frame, text="绘制实验分析图", command=self.plot_results, state=tk.DISABLED)
        self.plot_btn.grid(row=1, column=0, pady=5, padx=5)

        self.network_viz_btn = ttk.Button(viz_frame, text="可视化网络", command=self.visualize_network,
                                          state=tk.DISABLED)
        self.network_viz_btn.grid(row=1, column=1, pady=5, padx=5)

    def _create_result_display(self):
        """创建结果显示区域"""
        # 结果显示框架
        result_frame = ttk.LabelFrame(self.main_frame, text="诊断结果", padding="10")
        result_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # 配置网格权重
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)

        # 创建标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 指标标签页
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text="性能指标")

        self.metrics_text = scrolledtext.ScrolledText(metrics_tab, width=70, height=15, font=('Consolas', 10))
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 网络信息标签页
        info_tab = ttk.Frame(self.notebook)
        self.notebook.add(info_tab, text="网络信息")

        self.info_text = scrolledtext.ScrolledText(info_tab, width=70, height=15, font=('Consolas', 10))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 故障节点标签页
        faults_tab = ttk.Frame(self.notebook)
        self.notebook.add(faults_tab, text="故障节点")

        self.faults_text = scrolledtext.ScrolledText(faults_tab, width=70, height=15, font=('Consolas', 10))
        self.faults_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 日志标签页
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="运行日志")

        self.log_text = scrolledtext.ScrolledText(log_tab, width=70, height=15, font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def select_nodes_file(self):
        """选择节点文件"""
        filename = filedialog.askopenfilename(
            title="选择节点文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.nodes_file_var.set(filename)

    def select_edges_file(self):
        """选择边文件"""
        filename = filedialog.askopenfilename(
            title="选择边文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.edges_file_var.set(filename)

    def select_symptoms_file(self):
        """选择症状集文件"""
        filename = filedialog.askopenfilename(
            title="选择症状集文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.symptoms_file_var.set(filename)

    def select_nodes_file3(self):
        """选择功能3的节点文件"""
        filename = filedialog.askopenfilename(
            title="选择节点文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.nodes_file_var3.set(filename)

    def select_edges_file3(self):
        """选择功能3的边文件"""
        filename = filedialog.askopenfilename(
            title="选择边文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.edges_file_var3.set(filename)

    def run_function1(self):
        """运行功能1"""
        try:
            self.status_var.set("正在运行功能1...")
            self.log_message("开始运行功能1: 合成网络诊断")

            # 获取参数
            n = int(self.n_var.get())
            num_faults = int(self.faults_var1.get())
            num_experiments = int(self.exp_var.get())

            self.log_message(f"参数: n={n}, 故障节点数={num_faults}, 实验次数={num_experiments}")

            # 在新线程中运行，避免界面冻结
            thread = threading.Thread(target=self._run_function1_thread, args=(n, num_faults, num_experiments))
            thread.daemon = True
            thread.start()

        except ValueError as e:
            messagebox.showerror("输入错误", f"请输入有效的数字参数: {e}")
            self.status_var.set("就绪")
        except Exception as e:
            messagebox.showerror("错误", f"运行功能1时出错: {e}")
            self.status_var.set("就绪")

    def _run_function1_thread(self, n: int, num_faults: int, num_experiments: int):
        """功能1线程函数"""
        try:
            # 运行功能1
            result = self.system.function1(n, num_faults, num_experiments)
            self.current_result = result

            # 更新UI
            self.root.after(0, self._update_function1_results, result)

            # 启用可视化按钮
            self.root.after(0, lambda: self.plot_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.network_viz_btn.config(state=tk.NORMAL))

            self.log_message("功能1运行完成")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行功能1时出错: {e}"))
            self.status_var.set("就绪")

    def run_function2(self):
        """运行功能2"""
        try:
            self.status_var.set("正在运行功能2...")
            self.log_message("开始运行功能2: 真实网络诊断（输入症状集）")

            # 检查文件
            nodes_file = self.nodes_file_var.get()
            edges_file = self.edges_file_var.get()
            symptoms_file = self.symptoms_file_var.get()

            if not nodes_file or not edges_file or not symptoms_file:
                messagebox.showerror("错误", "请选择所有必需的文件")
                self.status_var.set("就绪")
                return

            self.log_message(f"节点文件: {nodes_file}")
            self.log_message(f"边文件: {edges_file}")
            self.log_message(f"症状集文件: {symptoms_file}")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function2_thread, args=(nodes_file, edges_file, symptoms_file))
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"运行功能2时出错: {e}")
            self.status_var.set("就绪")

    def _run_function2_thread(self, nodes_file: str, edges_file: str, symptoms_file: str):
        """功能2线程函数"""
        try:
            # 读取数据
            nodes_df = pd.read_excel(nodes_file)
            nodes = nodes_df.iloc[:, 0].astype(str).tolist()

            edges_df = pd.read_excel(edges_file)
            edges = list(zip(edges_df.iloc[:, 0].astype(str), edges_df.iloc[:, 1].astype(str)))

            symptoms_df = pd.read_excel(symptoms_file)
            symptoms = {}
            for _, row in symptoms_df.iterrows():
                u = str(row.iloc[0])
                v = str(row.iloc[1])
                sigma = int(row.iloc[2])
                symptoms[(u, v)] = sigma

            self.log_message(f"读取数据完成: {len(nodes)}个节点, {len(edges)}条边, {len(symptoms)}条测试结果")

            # 运行功能2
            result = self.system.function2(nodes, edges, symptoms)
            self.current_result = result

            # 更新UI
            self.root.after(0, self._update_function2_results, result)

            self.log_message("功能2运行完成")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行功能2时出错: {e}"))
            self.status_var.set("就绪")

    def run_function3(self):
        """运行功能3"""
        try:
            self.status_var.set("正在运行功能3...")
            self.log_message("开始运行功能3: 真实网络诊断（生成症状集）")

            # 检查文件
            nodes_file = self.nodes_file_var3.get()
            edges_file = self.edges_file_var3.get()

            if not nodes_file or not edges_file:
                messagebox.showerror("错误", "请选择节点文件和边文件")
                self.status_var.set("就绪")
                return

            n = int(self.n_var3.get())
            num_faults = int(self.faults_var3.get())

            self.log_message(f"节点文件: {nodes_file}")
            self.log_message(f"边文件: {edges_file}")
            self.log_message(f"参数: n={n}, 故障节点数={num_faults}")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function3_thread, args=(nodes_file, edges_file, n, num_faults))
            thread.daemon = True
            thread.start()

        except ValueError as e:
            messagebox.showerror("输入错误", f"请输入有效的数字参数: {e}")
            self.status_var.set("就绪")
        except Exception as e:
            messagebox.showerror("错误", f"运行功能3时出错: {e}")
            self.status_var.set("就绪")

    def _run_function3_thread(self, nodes_file: str, edges_file: str, n: int, num_faults: int):
        """功能3线程函数"""
        try:
            # 读取数据
            nodes_df = pd.read_excel(nodes_file)
            nodes = nodes_df.iloc[:, 0].astype(str).tolist()

            edges_df = pd.read_excel(edges_file)
            edges = list(zip(edges_df.iloc[:, 0].astype(str), edges_df.iloc[:, 1].astype(str)))

            self.log_message(f"读取数据完成: {len(nodes)}个节点, {len(edges)}条边")

            # 运行功能3
            result = self.system.function3(nodes, edges, n, num_faults)
            self.current_result = result

            # 更新UI
            self.root.after(0, self._update_function3_results, result)

            # 启用网络可视化按钮
            self.root.after(0, lambda: self.network_viz_btn.config(state=tk.NORMAL))

            self.log_message("功能3运行完成")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行功能3时出错: {e}"))
            self.status_var.set("就绪")

    def _update_function1_results(self, result: Dict[str, Any]):
        """更新功能1的结果显示"""
        # 清空文本框
        self.metrics_text.delete(1.0, tk.END)
        self.info_text.delete(1.0, tk.END)
        self.faults_text.delete(1.0, tk.END)

        # 显示网络信息
        network_info = result["网络信息"]
        info_str = "网络信息:\n"
        for key, value in network_info.items():
            info_str += f"  {key}: {value}\n"
        self.info_text.insert(tk.END, info_str)

        # 显示性能指标
        if len(result["实验结果"]) > 1:
            # 多次实验，显示平均指标
            avg_metrics = result["平均指标"]
            metrics_str = "平均性能指标:\n"
            for key, value in avg_metrics.items():
                metrics_str += f"  {key}: {value * 100:.2f}%\n"

            # 显示每次实验的指标
            metrics_str += "\n每次实验指标:\n"
            for exp_result in result["实验结果"]:
                metrics_str += f"  实验{exp_result['实验编号']}: "
                metrics_str += f"准确率={exp_result['指标']['准确率'] * 100:.2f}%, "
                metrics_str += f"真正率={exp_result['指标']['真正率'] * 100:.2f}%, "
                metrics_str += f"精确率={exp_result['指标']['精确率'] * 100:.2f}%\n"
        else:
            # 单次实验
            exp_result = result["实验结果"][0]
            metrics = exp_result["指标"]
            metrics_str = "性能指标:\n"
            for key, value in metrics.items():
                metrics_str += f"  {key}: {value * 100:.2f}%\n"

        self.metrics_text.insert(tk.END, metrics_str)

        # 显示故障节点
        if result["实验结果"]:
            last_exp = result["实验结果"][-1]
            true_faults = last_exp["真实故障节点"]
            diagnosed_faults = last_exp["诊断故障节点"]

            faults_str = "真实故障节点:\n"
            for i, node in enumerate(true_faults, 1):
                faults_str += f"  {i}. {node}\n"

            faults_str += "\n诊断故障节点:\n"
            for i, node in enumerate(diagnosed_faults, 1):
                status = "✓" if node in true_faults else "✗"
                faults_str += f"  {i}. {node} {status}\n"

            # 计算匹配度
            correct = len(set(true_faults) & set(diagnosed_faults))
            total_true = len(true_faults)
            total_diagnosed = len(diagnosed_faults)

            faults_str += f"\n诊断统计:\n"
            faults_str += f"  真实故障节点数: {total_true}\n"
            faults_str += f"  诊断故障节点数: {total_diagnosed}\n"
            faults_str += f"  正确诊断数: {correct}\n"
            faults_str += f"  诊断准确率: {correct / total_true * 100:.2f}%\n"

            self.faults_text.insert(tk.END, faults_str)

    def _update_function2_results(self, result: Dict[str, Any]):
        """更新功能2的结果显示"""
        # 清空文本框
        self.metrics_text.delete(1.0, tk.END)
        self.info_text.delete(1.0, tk.END)
        self.faults_text.delete(1.0, tk.END)

        # 显示网络信息
        network_info = result["网络信息"]
        info_str = "网络信息:\n"
        for key, value in network_info.items():
            info_str += f"  {key}: {value}\n"
        self.info_text.insert(tk.END, info_str)

        # 功能2没有性能指标
        self.metrics_text.insert(tk.END, "功能2: 真实网络诊断（输入症状集）\n\n")
        self.metrics_text.insert(tk.END, "注：功能2需要真实故障节点才能计算性能指标\n")
        self.metrics_text.insert(tk.END, f"诊断度 t = {result['诊断度t']}\n")

        # 显示诊断结果
        diagnosed_faults = result["诊断故障节点"]
        faults_str = f"诊断故障节点 ({len(diagnosed_faults)}个):\n"
        for i, node in enumerate(diagnosed_faults, 1):
            faults_str += f"  {i}. {node}\n"

        self.faults_text.insert(tk.END, faults_str)

    def _update_function3_results(self, result: Dict[str, Any]):
        """更新功能3的结果显示"""
        # 清空文本框
        self.metrics_text.delete(1.0, tk.END)
        self.info_text.delete(1.0, tk.END)
        self.faults_text.delete(1.0, tk.END)

        # 显示网络信息
        network_info = result["网络信息"]
        info_str = "网络信息:\n"
        for key, value in network_info.items():
            info_str += f"  {key}: {value}\n"
        self.info_text.insert(tk.END, info_str)

        # 显示性能指标
        metrics = result["指标"]
        metrics_str = "性能指标:\n"
        for key, value in metrics.items():
            metrics_str += f"  {key}: {value * 100:.2f}%\n"
        self.metrics_text.insert(tk.END, metrics_str)

        # 显示故障节点
        true_faults = result["真实故障节点"]
        diagnosed_faults = result["诊断故障节点"]

        faults_str = "真实故障节点:\n"
        for i, node in enumerate(true_faults, 1):
            faults_str += f"  {i}. {node}\n"

        faults_str += "\n诊断故障节点:\n"
        for i, node in enumerate(diagnosed_faults, 1):
            status = "✓" if node in true_faults else "✗"
            faults_str += f"  {i}. {node} {status}\n"

        # 计算匹配度
        correct = len(set(true_faults) & set(diagnosed_faults))
        total_true = len(true_faults)
        total_diagnosed = len(diagnosed_faults)

        faults_str += f"\n诊断统计:\n"
        faults_str += f"  真实故障节点数: {total_true}\n"
        faults_str += f"  诊断故障节点数: {total_diagnosed}\n"
        faults_str += f"  正确诊断数: {correct}\n"
        faults_str += f"  诊断准确率: {correct / total_true * 100:.2f}%\n"

        self.faults_text.insert(tk.END, faults_str)

    def plot_results(self):
        """绘制实验分析图"""
        if self.system.metrics_history and len(self.system.metrics_history) > 1:
            self.system.plot_experiment_results()
        else:
            messagebox.showinfo("提示", "需要多次实验数据才能绘制分析图")

    def visualize_network(self):
        """可视化网络"""
        if hasattr(self.system, 'network') and self.system.network:
            if isinstance(self.system.network, BubbleSortNetwork):
                # 冒泡排序网络可视化
                if hasattr(self.system, 'true_faulty_nodes'):
                    self.system.network.visualize_network(self.system.true_faulty_nodes)
                else:
                    self.system.network.visualize_network()
            else:
                messagebox.showinfo("提示", "通用网络的可视化功能正在开发中")
        else:
            messagebox.showinfo("提示", "请先运行诊断功能")

    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    print("=" * 60)
    print("冒泡排序网络故障诊断系统 (TRFI-PMC)")
    print("=" * 60)

    # 设置随机种子
    random.seed(42)

    # 创建并运行GUI
    app = TRFI_PMC_GUI()
    app.run()


if __name__ == "__main__":
    main()