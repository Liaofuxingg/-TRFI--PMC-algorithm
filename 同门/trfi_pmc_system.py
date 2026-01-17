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
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class BubbleSortNetwork:
    """冒泡排序网络类"""

    def __init__(self, n: int):
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
        """生成邻接表"""
        adj_list = {}
        for node in self.nodes:
            neighbors = []
            for i in range(self.n - 1):
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
            "平均度": self.n - 1,
            "是否为正则图": True,
            "正则度": self.n - 1
        }


class PMCModel:
    """PMC模型实现"""

    def __init__(self, network):
        self.network = network

    def generate_symptoms(self, fault_nodes: Set[str]) -> Dict[Tuple[str, str], int]:
        """根据PMC模型生成症状集"""
        symptoms = {}

        for node in self.network.nodes:
            for neighbor in self.network.get_neighbors(node):
                u, v = node, neighbor

                if (v, u) in symptoms:
                    continue

                u_fault = u in fault_nodes
                v_fault = v in fault_nodes

                if not u_fault and not v_fault:
                    symptoms[(u, v)] = 0
                    symptoms[(v, u)] = 0
                elif not u_fault and v_fault:
                    symptoms[(u, v)] = 1
                    symptoms[(v, u)] = random.randint(0, 1)
                elif u_fault and not v_fault:
                    symptoms[(u, v)] = random.randint(0, 1)
                    symptoms[(v, u)] = 1
                else:
                    symptoms[(u, v)] = random.randint(0, 1)
                    symptoms[(v, u)] = random.randint(0, 1)

        return symptoms


class TRFI_PMC:
    """三轮故障识别算法"""

    def __init__(self, network, symptoms: Dict[Tuple[str, str], int]):
        self.network = network
        self.symptoms = symptoms

    def _is_00_edge(self, u: str, v: str) -> bool:
        """检查是否为(0,0)边"""
        if (u, v) in self.symptoms and (v, u) in self.symptoms:
            return self.symptoms[(u, v)] == 0 and self.symptoms[(v, u)] == 0
        return False

    def run(self, t: int) -> List[str]:
        """运行TRFI-PMC算法"""
        # 分组阶段
        groups = []
        group_of_node = {}
        visited = set()

        for node in self.network.nodes:
            if node in visited:
                continue

            group = []
            queue = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue

                visited.add(current)
                group.append(current)
                group_of_node[current] = len(groups)

                for neighbor in self.network.get_neighbors(current):
                    if neighbor not in visited and self._is_00_edge(current, neighbor):
                        queue.append(neighbor)

            if group:
                groups.append(group)

        # 初始化组状态
        unclassified = set(range(len(groups)))
        faulty_groups = set()
        fault_free_groups = set()

        # 第一轮检测
        to_remove = set()
        for group_idx in unclassified:
            group = groups[group_idx]
            for node in group:
                for neighbor in self.network.get_neighbors(node):
                    if neighbor in group:
                        continue

                    sigma_uv = self.symptoms.get((node, neighbor), -1)
                    sigma_vu = self.symptoms.get((neighbor, node), -1)

                    if sigma_uv == 0 and sigma_vu == 1:
                        faulty_groups.add(group_idx)
                        to_remove.add(group_idx)
                        break
                    elif sigma_uv == 1 and sigma_vu == 0:
                        neighbor_group = group_of_node.get(neighbor)
                        if neighbor_group is not None:
                            faulty_groups.add(neighbor_group)
                            to_remove.add(neighbor_group)
                        break
                if group_idx in to_remove:
                    break

        unclassified -= to_remove

        # 第二轮检测
        to_remove = set()
        for group_idx in list(unclassified):
            group_size = len(groups[group_idx])
            remaining_faults = t - len(faulty_groups)

            if group_size > remaining_faults:
                fault_free_groups.add(group_idx)
                to_remove.add(group_idx)

        unclassified -= to_remove

        # 第三轮检测
        to_remove = set()
        for group_idx in list(unclassified):
            unclassified_neighbor_count = 0
            for node in groups[group_idx]:
                for neighbor in self.network.get_neighbors(node):
                    neighbor_group = group_of_node.get(neighbor)
                    if (neighbor_group is not None and
                            neighbor_group in unclassified and
                            neighbor_group != group_idx):
                        unclassified_neighbor_count += 1

            remaining_faults = t - len(faulty_groups)

            if unclassified_neighbor_count > remaining_faults:
                faulty_groups.add(group_idx)
                to_remove.add(group_idx)

        unclassified -= to_remove
        fault_free_groups.update(unclassified)

        # 收集故障节点
        faulty_nodes = []
        for group_idx in faulty_groups:
            faulty_nodes.extend(groups[group_idx])

        return faulty_nodes


class GeneralNetwork:
    """通用网络类"""

    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]]):
        self.nodes = nodes
        self.edges = edges
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


class Evaluator:
    """评估器：计算5个核心指标"""

    def __init__(self, true_faulty: Set[str], predicted_faulty: Set[str], all_nodes: List[str]):
        self.true_faulty = true_faulty
        self.predicted_faulty = predicted_faulty
        self.all_nodes = set(all_nodes)

        # 计算混淆矩阵
        self.TP = len(true_faulty & predicted_faulty)  # 真正例
        self.FP = len(predicted_faulty - true_faulty)  # 假正例
        self.TN = len((self.all_nodes - true_faulty) - predicted_faulty)  # 真负例
        self.FN = len(true_faulty - predicted_faulty)  # 假负例

    def calculate_all_metrics(self) -> Dict[str, float]:
        """计算所有5个指标"""
        metrics = {}

        # 准确率 (Accuracy)
        if self.TP + self.TN + self.FP + self.FN > 0:
            metrics['ACCR'] = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        else:
            metrics['ACCR'] = 0.0

        # 真负率 (True Negative Rate)
        if self.TN + self.FP > 0:
            metrics['TNR'] = self.TN / (self.TN + self.FP)
        else:
            metrics['TNR'] = 0.0

        # 假正率 (False Positive Rate)
        if self.TN + self.FP > 0:
            metrics['FPR'] = self.FP / (self.TN + self.FP)
        else:
            metrics['FPR'] = 0.0

        # 真正率 (True Positive Rate / Recall)
        if self.TP + self.FN > 0:
            metrics['TPR'] = self.TP / (self.TP + self.FN)
        else:
            metrics['TPR'] = 0.0

        # 精确率 (Precision)
        if self.TP + self.FP > 0:
            metrics['Precision'] = self.TP / (self.TP + self.FP)
        else:
            metrics['Precision'] = 0.0

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return np.array([
            [self.TP, self.FN],
            [self.FP, self.TN]
        ])


class TRFI_PMC_System:
    """TRFI-PMC诊断系统"""

    def __init__(self):
        self.network = None
        self.true_faulty_nodes = set()
        self.diagnosed_faulty_nodes = set()
        self.metrics_history = []
        self.experiment_times = []

    def function1(self, n: int, num_faults: int, num_experiments: int = 1) -> Dict[str, Any]:
        """功能1：合成网络诊断"""
        print(f"功能1：合成网络诊断")
        print(f"网络维度: {n}, 故障节点数: {num_faults}, 实验次数: {num_experiments}")

        self.metrics_history = []
        self.experiment_times = []
        all_results = []

        for exp in range(num_experiments):
            print(f"\n实验 {exp + 1}/{num_experiments}")
            start_time = time.time()

            # 创建冒泡排序网络
            self.network = BubbleSortNetwork(n)
            network_info = self.network.get_network_info()

            # 随机选择故障节点
            if num_faults > len(self.network.nodes):
                num_faults = max(1, int(len(self.network.nodes) * 0.1))

            self.true_faulty_nodes = set(random.sample(self.network.nodes, num_faults))

            # 生成症状集
            pmc_model = PMCModel(self.network)
            symptoms = pmc_model.generate_symptoms(self.true_faulty_nodes)

            # 运行诊断
            diagnoser = TRFI_PMC(self.network, symptoms)
            diagnosed_nodes = diagnoser.run(num_faults)
            self.diagnosed_faulty_nodes = set(diagnosed_nodes)

            # 评估结果
            evaluator = Evaluator(self.true_faulty_nodes, self.diagnosed_faulty_nodes, self.network.nodes)
            metrics = evaluator.calculate_all_metrics()

            end_time = time.time()
            exp_time = end_time - start_time
            self.experiment_times.append(exp_time)

            # 保存结果
            result = {
                "实验编号": exp + 1,
                "网络信息": network_info,
                "真实故障节点": list(self.true_faulty_nodes),
                "诊断故障节点": list(self.diagnosed_faulty_nodes),
                "指标": metrics,
                "运行时间": exp_time
            }
            all_results.append(result)
            self.metrics_history.append(metrics)

            # 显示本次实验指标
            print(f"ACCR: {metrics['ACCR'] * 100:.2f}%")
            print(f"TNR: {metrics['TNR'] * 100:.2f}%")
            print(f"FPR: {metrics['FPR'] * 100:.2f}%")
            print(f"TPR: {metrics['TPR'] * 100:.2f}%")
            print(f"Precision: {metrics['Precision'] * 100:.2f}%")
            print(f"运行时间: {exp_time:.2f}秒")

        # 计算平均指标
        avg_metrics = {}
        if num_experiments > 1:
            print(f"\n{'=' * 60}")
            print("平均性能指标:")
            print(f"{'=' * 60}")
            for key in self.metrics_history[0].keys():
                avg_value = np.mean([m[key] for m in self.metrics_history])
                avg_metrics[key] = avg_value
                print(f"{key}: {avg_value * 100:.2f}%")
            print(f"平均运行时间: {np.mean(self.experiment_times):.2f}秒")

        # 保存数据
        self._save_function1_data(all_results, n, num_faults)

        return {
            "实验结果": all_results,
            "平均指标": avg_metrics if avg_metrics else self.metrics_history[0],
            "网络信息": network_info
        }

    def function2(self, nodes: List[str], edges: List[Tuple[str, str]],
                  symptoms: Dict[Tuple[str, str], int]) -> Dict[str, Any]:
        """功能2：真实网络诊断（输入症状集）"""
        print(f"功能2：真实网络诊断")

        # 创建通用网络
        self.network = GeneralNetwork(nodes, edges)

        # 运行诊断（使用节点数作为诊断度t的估计）
        t = min(10, len(nodes) // 10)
        diagnoser = TRFI_PMC(self.network, symptoms)
        diagnosed_nodes = diagnoser.run(t)
        self.diagnosed_faulty_nodes = set(diagnosed_nodes)

        print(f"诊断出 {len(self.diagnosed_faulty_nodes)} 个故障节点")

        return {
            "诊断故障节点": list(self.diagnosed_faulty_nodes),
            "诊断度t": t
        }

    def function3(self, nodes: List[str], edges: List[Tuple[str, str]],
                  n: int, num_faults: int) -> Dict[str, Any]:
        """功能3：真实网络诊断（生成症状集）"""
        print(f"功能3：真实网络诊断（生成症状集）")

        # 创建通用网络
        self.network = GeneralNetwork(nodes, edges)

        # 随机选择故障节点
        if num_faults > len(nodes):
            num_faults = max(1, int(len(nodes) * 0.1))

        self.true_faulty_nodes = set(random.sample(nodes, num_faults))

        # 生成症状集
        pmc_model = PMCModel(self.network)
        symptoms = pmc_model.generate_symptoms(self.true_faulty_nodes)

        # 运行诊断
        diagnoser = TRFI_PMC(self.network, symptoms)
        diagnosed_nodes = diagnoser.run(num_faults)
        self.diagnosed_faulty_nodes = set(diagnosed_nodes)

        # 评估结果
        evaluator = Evaluator(self.true_faulty_nodes, self.diagnosed_faulty_nodes, nodes)
        metrics = evaluator.calculate_all_metrics()

        print(f"ACCR: {metrics['ACCR'] * 100:.2f}%")
        print(f"TNR: {metrics['TNR'] * 100:.2f}%")
        print(f"FPR: {metrics['FPR'] * 100:.2f}%")
        print(f"TPR: {metrics['TPR'] * 100:.2f}%")
        print(f"Precision: {metrics['Precision'] * 100:.2f}%")

        return {
            "真实故障节点": list(self.true_faulty_nodes),
            "诊断故障节点": list(self.diagnosed_faulty_nodes),
            "指标": metrics
        }

    def _save_function1_data(self, results: List[Dict], n: int, num_faults: int):
        """保存功能1的数据"""
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存指标数据
        metrics_data = []
        for result in results:
            metrics = result["指标"]
            metrics_data.append({
                "实验编号": result["实验编号"],
                "ACCR": metrics['ACCR'],
                "TNR": metrics['TNR'],
                "FPR": metrics['FPR'],
                "TPR": metrics['TPR'],
                "Precision": metrics['Precision'],
                "运行时间": result["运行时间"]
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(f"results/function1_metrics_{timestamp}.xlsx", index=False)

        # 保存最后一次实验的详细数据
        if results:
            last_result = results[-1]
            faults_df = pd.DataFrame({
                "节点": list(last_result["真实故障节点"]) + list(last_result["诊断故障节点"]),
                "类型": ["真实故障"] * len(last_result["真实故障节点"]) + ["诊断故障"] * len(
                    last_result["诊断故障节点"])
            })
            faults_df.to_excel(f"results/function1_faults_{timestamp}.xlsx", index=False)

        print(f"数据已保存到 results/ 目录")

    def plot_performance_metrics(self, save_path: str = None):
        """绘制5个指标的性能变化图"""
        if not self.metrics_history or len(self.metrics_history) <= 1:
            print("需要多次实验数据才能绘制性能变化图")
            return

        # 提取数据
        experiments = list(range(1, len(self.metrics_history) + 1))
        metrics_names = ['ACCR', 'TNR', 'FPR', 'TPR', 'Precision']
        metrics_data = {name: [] for name in metrics_names}

        for metrics in self.metrics_history:
            for name in metrics_names:
                metrics_data[name].append(metrics[name] * 100)  # 转换为百分比

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        # 定义颜色和线型
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'D', 'v']

        # 1. 所有指标折线图
        ax = axes[0]
        for i, name in enumerate(metrics_names):
            ax.plot(experiments, metrics_data[name],
                    color=colors[i], linestyle=line_styles[i],
                    marker=markers[i], linewidth=2, markersize=6,
                    label=name, alpha=0.8)

        ax.set_xlabel('实验编号', fontsize=12)
        ax.set_ylabel('指标值 (%)', fontsize=12)
        ax.set_title('5个性能指标变化趋势', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # 2-6. 单个指标详细图
        for i, name in enumerate(metrics_names[1:], 1):
            ax = axes[i]
            ax.plot(experiments, metrics_data[name],
                    color=colors[i], linestyle=line_styles[i],
                    marker=markers[i], linewidth=2.5, markersize=8)

            # 添加平均值线
            mean_value = np.mean(metrics_data[name])
            ax.axhline(y=mean_value, color=colors[i], linestyle='--',
                       alpha=0.6, linewidth=1.5, label=f'平均: {mean_value:.2f}%')

            # 填充区域
            ax.fill_between(experiments, 0, metrics_data[name],
                            color=colors[i], alpha=0.2)

            ax.set_xlabel('实验编号', fontsize=10)
            ax.set_ylabel('指标值 (%)', fontsize=10)
            ax.set_title(f'{name}指标变化', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)

        # 7. 运行时间图
        ax = axes[5]
        ax.bar(experiments, self.experiment_times,
               color='#6C5CE7', alpha=0.7, edgecolor='black')

        # 添加数值标签
        for i, (exp, time_val) in enumerate(zip(experiments, self.experiment_times)):
            ax.text(exp, time_val + 0.01, f'{time_val:.2f}s',
                    ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('实验编号', fontsize=10)
        ax.set_ylabel('运行时间 (秒)', fontsize=10)
        ax.set_title('实验运行时间', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('TRFI-PMC算法多次实验性能分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"性能分析图已保存到: {save_path}")

        plt.show()

    def plot_metrics_comparison(self, save_path: str = None):
        """绘制5个指标的对比图"""
        if not self.metrics_history:
            print("需要实验数据才能绘制对比图")
            return

        # 计算平均指标
        metrics_names = ['ACCR', 'TNR', 'FPR', 'TPR', 'Precision']
        avg_metrics = {}

        for name in metrics_names:
            values = [m[name] * 100 for m in self.metrics_history]  # 转换为百分比
            avg_metrics[name] = np.mean(values)

        # 创建雷达图
        fig = plt.figure(figsize=(14, 6))

        # 1. 雷达图
        ax1 = plt.subplot(1, 3, 1, projection='polar')

        # 数据准备
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        values = [avg_metrics[name] for name in metrics_names]
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]

        # 绘制雷达图
        ax1.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax1.fill(angles, values, alpha=0.25, color='#FF6B6B')

        # 设置标签
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics_names, fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.set_yticks([20, 40, 60, 80, 100])
        ax1.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax1.set_title('平均性能指标雷达图', fontsize=13, fontweight='bold', pad=20)

        # 2. 条形图
        ax2 = plt.subplot(1, 3, 2)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = ax2.bar(range(len(metrics_names)), list(avg_metrics.values()),
                       color=colors, alpha=0.8, edgecolor='black')

        ax2.set_xlabel('性能指标', fontsize=11)
        ax2.set_ylabel('平均值 (%)', fontsize=11)
        ax2.set_title('平均性能指标对比', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(metrics_names)))
        ax2.set_xticklabels(metrics_names, fontsize=10, rotation=0)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 105)

        # 添加数值标签
        for bar, value in zip(bars, avg_metrics.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{value:.2f}%', ha='center', va='bottom', fontsize=9)

        # 3. 箱线图
        ax3 = plt.subplot(1, 3, 3)

        data_to_plot = []
        for name in metrics_names:
            values = [m[name] * 100 for m in self.metrics_history]
            data_to_plot.append(values)

        box = ax3.boxplot(data_to_plot, labels=metrics_names, patch_artist=True)

        # 设置箱线图颜色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_xlabel('性能指标', fontsize=11)
        ax3.set_ylabel('指标值 (%)', fontsize=11)
        ax3.set_title('性能指标分布箱线图', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 105)

        plt.suptitle('TRFI-PMC算法性能指标综合分析', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"指标对比图已保存到: {save_path}")

        plt.show()


class TRFI_PMC_GUI:
    """TRFI-PMC系统GUI界面"""

    def __init__(self):
        self.system = TRFI_PMC_System()
        self.current_result = None

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("冒泡排序网络故障诊断系统 (TRFI-PMC)")
        self.root.geometry("1200x800")

        # 设置图标
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass

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
        title_label = ttk.Label(self.main_frame, text="冒泡排序网络故障诊断系统 (TRFI-PMC)",
                                style='Title.TLabel', foreground='#2C3E50')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 创建功能选择区域
        self._create_function_selection()

        # 创建结果显示区域
        self._create_result_display()

        # 创建状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, foreground='#34495E')
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_function_selection(self):
        """创建功能选择区域"""
        # 功能选择框架
        func_frame = ttk.LabelFrame(self.main_frame, text="功能选择", padding="15")
        func_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # 功能1：合成网络诊断
        func1_frame = ttk.LabelFrame(func_frame, text="功能1: 合成网络诊断", padding="10")
        func1_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 参数输入
        ttk.Label(func1_frame, text="网络维度 n:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.n_var = tk.StringVar(value="4")
        ttk.Entry(func1_frame, textvariable=self.n_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(func1_frame, text="故障节点数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.faults_var1 = tk.StringVar(value="5")
        ttk.Entry(func1_frame, textvariable=self.faults_var1, width=12).grid(row=1, column=1, sticky=tk.W, padx=5,
                                                                             pady=5)

        ttk.Label(func1_frame, text="实验次数:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.exp_var = tk.StringVar(value="5")
        ttk.Entry(func1_frame, textvariable=self.exp_var, width=12).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        func1_btn = ttk.Button(func1_frame, text="运行功能1", command=self.run_function1,
                               style='Accent.TButton')
        func1_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # 功能2：真实网络诊断（输入症状集）
        func2_frame = ttk.LabelFrame(func_frame, text="功能2: 真实网络诊断（输入症状集）", padding="10")
        func2_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 文件选择
        ttk.Label(func2_frame, text="节点文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.nodes_file_var = tk.StringVar()
        ttk.Entry(func2_frame, textvariable=self.nodes_file_var, width=25).grid(row=0, column=1, sticky=tk.W, padx=5,
                                                                                pady=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_nodes_file, width=8).grid(row=0, column=2, padx=5,
                                                                                           pady=5)

        ttk.Label(func2_frame, text="边文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.edges_file_var = tk.StringVar()
        ttk.Entry(func2_frame, textvariable=self.edges_file_var, width=25).grid(row=1, column=1, sticky=tk.W, padx=5,
                                                                                pady=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_edges_file, width=8).grid(row=1, column=2, padx=5,
                                                                                           pady=5)

        ttk.Label(func2_frame, text="症状集文件:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.symptoms_file_var = tk.StringVar()
        ttk.Entry(func2_frame, textvariable=self.symptoms_file_var, width=25).grid(row=2, column=1, sticky=tk.W, padx=5,
                                                                                   pady=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_symptoms_file, width=8).grid(row=2, column=2, padx=5,
                                                                                              pady=5)

        func2_btn = ttk.Button(func2_frame, text="运行功能2", command=self.run_function2,
                               style='Accent.TButton')
        func2_btn.grid(row=3, column=0, columnspan=3, pady=(10, 0))

        # 功能3：真实网络诊断（生成症状集）
        func3_frame = ttk.LabelFrame(func_frame, text="功能3: 真实网络诊断（生成症状集）", padding="10")
        func3_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(func3_frame, text="节点文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.nodes_file_var3 = tk.StringVar()
        ttk.Entry(func3_frame, textvariable=self.nodes_file_var3, width=25).grid(row=0, column=1, sticky=tk.W, padx=5,
                                                                                 pady=5)
        ttk.Button(func3_frame, text="浏览", command=self.select_nodes_file3, width=8).grid(row=0, column=2, padx=5,
                                                                                            pady=5)

        ttk.Label(func3_frame, text="边文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.edges_file_var3 = tk.StringVar()
        ttk.Entry(func3_frame, textvariable=self.edges_file_var3, width=25).grid(row=1, column=1, sticky=tk.W, padx=5,
                                                                                 pady=5)
        ttk.Button(func3_frame, text="浏览", command=self.select_edges_file3, width=8).grid(row=1, column=2, padx=5,
                                                                                            pady=5)

        ttk.Label(func3_frame, text="网络维度 n:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.n_var3 = tk.StringVar(value="4")
        ttk.Entry(func3_frame, textvariable=self.n_var3, width=12).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(func3_frame, text="故障节点数:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.faults_var3 = tk.StringVar(value="10")
        ttk.Entry(func3_frame, textvariable=self.faults_var3, width=12).grid(row=3, column=1, sticky=tk.W, padx=5,
                                                                             pady=5)

        func3_btn = ttk.Button(func3_frame, text="运行功能3", command=self.run_function3,
                               style='Accent.TButton')
        func3_btn.grid(row=4, column=0, columnspan=3, pady=(10, 0))

        # 可视化按钮框架
        viz_frame = ttk.LabelFrame(func_frame, text="可视化选项", padding="10")
        viz_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.plot_btn = ttk.Button(viz_frame, text="绘制性能变化图",
                                   command=self.plot_performance, state=tk.DISABLED, width=20)
        self.plot_btn.grid(row=0, column=0, pady=5, padx=5)

        self.comparison_btn = ttk.Button(viz_frame, text="绘制指标对比图",
                                         command=self.plot_comparison, state=tk.DISABLED, width=20)
        self.comparison_btn.grid(row=0, column=1, pady=5, padx=5)

        self.save_btn = ttk.Button(viz_frame, text="保存当前图表",
                                   command=self.save_current_plot, state=tk.DISABLED, width=20)
        self.save_btn.grid(row=1, column=0, pady=5, padx=5)

        self.export_btn = ttk.Button(viz_frame, text="导出实验数据",
                                     command=self.export_data, state=tk.DISABLED, width=20)
        self.export_btn.grid(row=1, column=1, pady=5, padx=5)

    def _create_result_display(self):
        """创建结果显示区域"""
        # 结果显示框架
        result_frame = ttk.LabelFrame(self.main_frame, text="诊断结果", padding="10")
        result_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S),
                          padx=(15, 5), pady=5)

        # 配置网格权重
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)

        # 创建标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 指标标签页
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text="性能指标")

        # 创建指标表格
        columns = ("实验", "ACCR", "TNR", "FPR", "TPR", "Precision")
        self.metrics_tree = ttk.Treeview(metrics_tab, columns=columns, show="headings", height=12)

        # 设置列标题
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=100, anchor="center")

        # 添加滚动条
        scrollbar = ttk.Scrollbar(metrics_tab, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        self.metrics_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        # 平均指标框架
        avg_frame = ttk.Frame(metrics_tab)
        avg_frame.pack(fill="x", padx=5, pady=(0, 5))

        ttk.Label(avg_frame, text="平均指标:", font=('Arial', 10, 'bold')).pack(side="left", padx=5)
        self.avg_labels = {}
        for metric in ["ACCR", "TNR", "FPR", "TPR", "Precision"]:
            frame = ttk.Frame(avg_frame)
            frame.pack(side="left", padx=10)
            ttk.Label(frame, text=f"{metric}:").pack(side="left")
            self.avg_labels[metric] = ttk.Label(frame, text="0.00%", foreground="blue", font=('Arial', 9, 'bold'))
            self.avg_labels[metric].pack(side="left")

        # 网络信息标签页
        info_tab = ttk.Frame(self.notebook)
        self.notebook.add(info_tab, text="网络信息")

        self.info_text = scrolledtext.ScrolledText(info_tab, width=70, height=15,
                                                   font=('Consolas', 10))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 故障节点标签页
        faults_tab = ttk.Frame(self.notebook)
        self.notebook.add(faults_tab, text="故障节点")

        # 创建两个文本区域
        frame_left = ttk.Frame(faults_tab)
        frame_left.pack(side="left", fill="both", expand=True, padx=(5, 2), pady=5)

        ttk.Label(frame_left, text="真实故障节点:", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.true_faults_text = scrolledtext.ScrolledText(frame_left, height=12,
                                                          font=('Consolas', 9))
        self.true_faults_text.pack(fill="both", expand=True)

        frame_right = ttk.Frame(faults_tab)
        frame_right.pack(side="right", fill="both", expand=True, padx=(2, 5), pady=5)

        ttk.Label(frame_right, text="诊断故障节点:", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.diagnosed_faults_text = scrolledtext.ScrolledText(frame_right, height=12,
                                                               font=('Consolas', 9))
        self.diagnosed_faults_text.pack(fill="both", expand=True)

        # 日志标签页
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="运行日志")

        self.log_text = scrolledtext.ScrolledText(log_tab, width=70, height=15,
                                                  font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加日志控制按钮
        log_control_frame = ttk.Frame(log_tab)
        log_control_frame.pack(fill="x", padx=5, pady=(0, 5))

        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log, width=10).pack(side="right", padx=2)
        ttk.Button(log_control_frame, text="保存日志", command=self.save_log, width=10).pack(side="right", padx=2)

    def log_message(self, message: str, level: str = "INFO"):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 设置颜色
        colors = {
            "INFO": "black",
            "SUCCESS": "green",
            "WARNING": "orange",
            "ERROR": "red"
        }

        color = colors.get(level, "black")

        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.tag_add(f"color_{level}", f"end-2l linestart", f"end-2l lineend")
        self.log_text.tag_config(f"color_{level}", foreground=color)
        self.log_text.see(tk.END)
        self.root.update()

    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)

    def save_log(self):
        """保存日志"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"日志已保存到: {file_path}", "SUCCESS")

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
            self.log_message("开始运行功能1: 合成网络诊断", "INFO")

            # 获取参数
            n = int(self.n_var.get())
            num_faults = int(self.faults_var1.get())
            num_experiments = int(self.exp_var.get())

            self.log_message(f"参数: n={n}, 故障节点数={num_faults}, 实验次数={num_experiments}", "INFO")

            # 清空结果
            self.metrics_tree.delete(*self.metrics_tree.get_children())
            for metric in self.avg_labels:
                self.avg_labels[metric].config(text="0.00%")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function1_thread,
                                      args=(n, num_faults, num_experiments))
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
            self.root.after(0, lambda: self.comparison_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL))

            self.log_message("功能1运行完成", "SUCCESS")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行功能1时出错: {e}"))
            self.log_message(f"功能1运行出错: {e}", "ERROR")
            self.status_var.set("就绪")

    def run_function2(self):
        """运行功能2"""
        try:
            self.status_var.set("正在运行功能2...")
            self.log_message("开始运行功能2: 真实网络诊断（输入症状集）", "INFO")

            # 检查文件
            nodes_file = self.nodes_file_var.get()
            edges_file = self.edges_file_var.get()
            symptoms_file = self.symptoms_file_var.get()

            if not nodes_file or not edges_file or not symptoms_file:
                messagebox.showerror("错误", "请选择所有必需的文件")
                self.status_var.set("就绪")
                return

            self.log_message(f"节点文件: {nodes_file}", "INFO")
            self.log_message(f"边文件: {edges_file}", "INFO")
            self.log_message(f"症状集文件: {symptoms_file}", "INFO")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function2_thread,
                                      args=(nodes_file, edges_file, symptoms_file))
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

            self.log_message(f"读取数据完成: {len(nodes)}个节点, {len(edges)}条边, {len(symptoms)}条测试结果", "INFO")

            # 运行功能2
            result = self.system.function2(nodes, edges, symptoms)
            self.current_result = result

            # 更新UI
            self.root.after(0, self._update_function2_results, result)

            self.log_message("功能2运行完成", "SUCCESS")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行功能2时出错: {e}"))
            self.log_message(f"功能2运行出错: {e}", "ERROR")
            self.status_var.set("就绪")

    def run_function3(self):
        """运行功能3"""
        try:
            self.status_var.set("正在运行功能3...")
            self.log_message("开始运行功能3: 真实网络诊断（生成症状集）", "INFO")

            # 检查文件
            nodes_file = self.nodes_file_var3.get()
            edges_file = self.edges_file_var3.get()

            if not nodes_file or not edges_file:
                messagebox.showerror("错误", "请选择节点文件和边文件")
                self.status_var.set("就绪")
                return

            n = int(self.n_var3.get())
            num_faults = int(self.faults_var3.get())

            self.log_message(f"节点文件: {nodes_file}", "INFO")
            self.log_message(f"边文件: {edges_file}", "INFO")
            self.log_message(f"参数: n={n}, 故障节点数={num_faults}", "INFO")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function3_thread,
                                      args=(nodes_file, edges_file, n, num_faults))
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

            self.log_message(f"读取数据完成: {len(nodes)}个节点, {len(edges)}条边", "INFO")

            # 运行功能3
            result = self.system.function3(nodes, edges, n, num_faults)
            self.current_result = result

            # 更新UI
            self.root.after(0, self._update_function3_results, result)

            self.log_message("功能3运行完成", "SUCCESS")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行功能3时出错: {e}"))
            self.log_message(f"功能3运行出错: {e}", "ERROR")
            self.status_var.set("就绪")

    def _update_function1_results(self, result: Dict[str, Any]):
        """更新功能1的结果显示"""
        # 清空文本区域
        self.info_text.delete(1.0, tk.END)
        self.true_faults_text.delete(1.0, tk.END)
        self.diagnosed_faults_text.delete(1.0, tk.END)

        # 显示网络信息
        network_info = result["网络信息"]
        info_str = "网络信息:\n"
        for key, value in network_info.items():
            info_str += f"  {key}: {value}\n"
        self.info_text.insert(tk.END, info_str)

        # 更新指标表格
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        for exp_result in result["实验结果"]:
            exp_num = exp_result["实验编号"]
            metrics = exp_result["指标"]

            values = (exp_num,
                      f"{metrics['ACCR'] * 100:.2f}%",
                      f"{metrics['TNR'] * 100:.2f}%",
                      f"{metrics['FPR'] * 100:.2f}%",
                      f"{metrics['TPR'] * 100:.2f}%",
                      f"{metrics['Precision'] * 100:.2f}%")

            self.metrics_tree.insert("", "end", values=values)

        # 更新平均指标
        if "平均指标" in result and result["平均指标"]:
            avg_metrics = result["平均指标"]
            for metric, value in avg_metrics.items():
                if metric in self.avg_labels:
                    self.avg_labels[metric].config(text=f"{value * 100:.2f}%")
        elif result["实验结果"]:
            # 计算平均指标
            avg_metrics = {}
            for metric in ["ACCR", "TNR", "FPR", "TPR", "Precision"]:
                values = [r["指标"][metric] for r in result["实验结果"]]
                avg_metrics[metric] = np.mean(values)

            for metric, value in avg_metrics.items():
                if metric in self.avg_labels:
                    self.avg_labels[metric].config(text=f"{value * 100:.2f}%")

        # 显示故障节点
        if result["实验结果"]:
            last_exp = result["实验结果"][-1]
            true_faults = last_exp["真实故障节点"]
            diagnosed_faults = last_exp["诊断故障节点"]

            # 真实故障节点
            self.true_faults_text.insert(tk.END, f"总数: {len(true_faults)}\n")
            self.true_faults_text.insert(tk.END, "-" * 30 + "\n")
            for i, node in enumerate(true_faults, 1):
                self.true_faults_text.insert(tk.END, f"{i}. {node}\n")

            # 诊断故障节点
            self.diagnosed_faults_text.insert(tk.END, f"总数: {len(diagnosed_faults)}\n")
            self.diagnosed_faults_text.insert(tk.END, "-" * 30 + "\n")
            for i, node in enumerate(diagnosed_faults, 1):
                status = "✓" if node in true_faults else "✗"
                self.diagnosed_faults_text.insert(tk.END, f"{i}. {node} {status}\n")

    def _update_function2_results(self, result: Dict[str, Any]):
        """更新功能2的结果显示"""
        # 清空文本区域
        self.info_text.delete(1.0, tk.END)
        self.true_faults_text.delete(1.0, tk.END)
        self.diagnosed_faults_text.delete(1.0, tk.END)

        # 功能2没有网络信息
        self.info_text.insert(tk.END, "功能2: 真实网络诊断（输入症状集）\n\n")
        self.info_text.insert(tk.END, "注：功能2需要真实故障节点才能计算性能指标\n")
        self.info_text.insert(tk.END, f"诊断度 t = {result['诊断度t']}\n")

        # 清空指标表格
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for metric in self.avg_labels:
            self.avg_labels[metric].config(text="N/A")

        # 显示诊断结果
        diagnosed_faults = result["诊断故障节点"]
        self.diagnosed_faults_text.insert(tk.END, f"诊断故障节点 ({len(diagnosed_faults)}个):\n")
        self.diagnosed_faults_text.insert(tk.END, "-" * 30 + "\n")
        for i, node in enumerate(diagnosed_faults, 1):
            self.diagnosed_faults_text.insert(tk.END, f"{i}. {node}\n")

    def _update_function3_results(self, result: Dict[str, Any]):
        """更新功能3的结果显示"""
        # 清空文本区域
        self.info_text.delete(1.0, tk.END)
        self.true_faults_text.delete(1.0, tk.END)
        self.diagnosed_faults_text.delete(1.0, tk.END)

        # 清空指标表格
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        # 添加单行指标
        metrics = result["指标"]
        values = (1,
                  f"{metrics['ACCR'] * 100:.2f}%",
                  f"{metrics['TNR'] * 100:.2f}%",
                  f"{metrics['FPR'] * 100:.2f}%",
                  f"{metrics['TPR'] * 100:.2f}%",
                  f"{metrics['Precision'] * 100:.2f}%")

        self.metrics_tree.insert("", "end", values=values)

        # 更新平均指标
        for metric, value in metrics.items():
            if metric in self.avg_labels:
                self.avg_labels[metric].config(text=f"{value * 100:.2f}%")

        # 显示故障节点
        true_faults = result["真实故障节点"]
        diagnosed_faults = result["诊断故障节点"]

        # 真实故障节点
        self.true_faults_text.insert(tk.END, f"总数: {len(true_faults)}\n")
        self.true_faults_text.insert(tk.END, "-" * 30 + "\n")
        for i, node in enumerate(true_faults, 1):
            self.true_faults_text.insert(tk.END, f"{i}. {node}\n")

        # 诊断故障节点
        self.diagnosed_faults_text.insert(tk.END, f"总数: {len(diagnosed_faults)}\n")
        self.diagnosed_faults_text.insert(tk.END, "-" * 30 + "\n")
        for i, node in enumerate(diagnosed_faults, 1):
            status = "✓" if node in true_faults else "✗"
            self.diagnosed_faults_text.insert(tk.END, f"{i}. {node} {status}\n")

    def plot_performance(self):
        """绘制性能变化图"""
        if not self.system.metrics_history or len(self.system.metrics_history) <= 1:
            messagebox.showinfo("提示", "需要多次实验数据才能绘制性能变化图")
            return

        # 在新线程中绘图
        thread = threading.Thread(target=self._plot_performance_thread)
        thread.daemon = True
        thread.start()

    def _plot_performance_thread(self):
        """绘图线程函数"""
        try:
            self.status_var.set("正在生成性能变化图...")
            self.log_message("正在生成性能变化图", "INFO")

            # 绘制性能变化图
            self.system.plot_performance_metrics()

            self.log_message("性能变化图生成完成", "SUCCESS")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"绘制图表时出错: {e}"))
            self.log_message(f"绘制图表出错: {e}", "ERROR")
            self.status_var.set("就绪")

    def plot_comparison(self):
        """绘制指标对比图"""
        if not self.system.metrics_history:
            messagebox.showinfo("提示", "需要实验数据才能绘制指标对比图")
            return

        # 在新线程中绘图
        thread = threading.Thread(target=self._plot_comparison_thread)
        thread.daemon = True
        thread.start()

    def _plot_comparison_thread(self):
        """绘图线程函数"""
        try:
            self.status_var.set("正在生成指标对比图...")
            self.log_message("正在生成指标对比图", "INFO")

            # 绘制指标对比图
            self.system.plot_metrics_comparison()

            self.log_message("指标对比图生成完成", "SUCCESS")
            self.status_var.set("就绪")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"绘制图表时出错: {e}"))
            self.log_message(f"绘制图表出错: {e}", "ERROR")
            self.status_var.set("就绪")

    def save_current_plot(self):
        """保存当前图表"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # 重新绘制并保存
                if len(self.system.metrics_history) > 1:
                    self.system.plot_performance_metrics(save_path=file_path)
                else:
                    self.system.plot_metrics_comparison(save_path=file_path)

                self.log_message(f"图表已保存到: {file_path}", "SUCCESS")
            except Exception as e:
                self.log_message(f"保存图表时出错: {e}", "ERROR")
                messagebox.showerror("错误", f"保存图表时出错: {e}")

    def export_data(self):
        """导出实验数据"""
        if not self.system.metrics_history:
            messagebox.showinfo("提示", "没有实验数据可导出")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # 准备数据
                data = []
                for i, metrics in enumerate(self.system.metrics_history, 1):
                    row = {"实验编号": i}
                    for metric_name, metric_value in metrics.items():
                        row[metric_name] = metric_value
                    data.append(row)

                # 保存为Excel
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)

                self.log_message(f"实验数据已导出到: {file_path}", "SUCCESS")
                messagebox.showinfo("导出成功", f"实验数据已成功导出到:\n{file_path}")

            except Exception as e:
                self.log_message(f"导出数据时出错: {e}", "ERROR")
                messagebox.showerror("错误", f"导出数据时出错: {e}")

    def on_closing(self):
        """关闭窗口时的处理"""
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            self.root.destroy()

    def run(self):
        """运行GUI"""
        # 设置窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        self.root.mainloop()


def main():
    """主函数"""
    print("=" * 70)
    print("冒泡排序网络故障诊断系统 (TRFI-PMC)")
    print("版本: 1.0.0")
    print("功能: 5个核心指标 (ACCR, TNR, FPR, TPR, Precision)")
    print("=" * 70)

    # 设置随机种子
    random.seed(42)

    # 创建并运行GUI
    app = TRFI_PMC_GUI()
    app.run()


if __name__ == "__main__":
    main()