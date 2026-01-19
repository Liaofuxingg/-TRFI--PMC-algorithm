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
    """三轮故障识别算法 - 按照C++代码逻辑修改"""

    def __init__(self, network, symptoms: Dict[Tuple[str, str], int]):
        self.network = network
        self.symptoms = symptoms

    def _generate_neighbor(self, s: str, j: int) -> str:
        """生成邻居节点，交换位置j和j+1"""
        neighbor_list = list(s)
        neighbor_list[j], neighbor_list[j + 1] = neighbor_list[j + 1], neighbor_list[j]
        return ''.join(neighbor_list)

    def run(self, t: int) -> List[str]:
        """运行TRFI-PMC算法（按照C++代码逻辑）"""
        # 分组阶段 - 基于(0,0)边进行BFS分组
        n = self.network.n if hasattr(self.network, 'n') else len(list(self.network.nodes[0]))
        nodes = self.network.nodes
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for idx, node in enumerate(nodes)}

        # 初始化数据结构
        visited = [False] * len(nodes)
        groups = []  # 存储每个组包含的节点索引
        group_of_node = {}  # 节点属于哪个组
        group_count = 0

        # BFS分组
        for i, node in enumerate(nodes):
            if not visited[i]:
                # 新组开始
                group = []
                queue = deque([i])
                visited[i] = True

                while queue:
                    current_idx = queue.popleft()
                    current_node = nodes[current_idx]
                    group.append(current_idx)
                    group_of_node[current_idx] = group_count

                    # 检查所有邻居
                    for j in range(n - 1):
                        neighbor_node = self._generate_neighbor(current_node, j)
                        neighbor_idx = node_to_idx.get(neighbor_node, -1)

                        if neighbor_idx != -1 and not visited[neighbor_idx]:
                            # 检查是否为(0,0)边
                            if ((current_node, neighbor_node) in self.symptoms and
                                    (neighbor_node, current_node) in self.symptoms and
                                    self.symptoms[(current_node, neighbor_node)] == 0 and
                                    self.symptoms[(neighbor_node, current_node)] == 0):
                                visited[neighbor_idx] = True
                                queue.append(neighbor_idx)

                groups.append(group)
                group_count += 1

        # 初始化组状态
        total_nodes = len(nodes)
        total_groups = len(groups)
        st2 = [False] * total_groups  # 标记组是否已被分类

        # 第一轮检测：fuxing() - 直接诊断
        chun = []  # 故障组
        guzhang = 0  # 故障节点数
        xi = []  # 正常组
        wuguzhang = 0  # 正常节点数

        for i in range(total_groups):
            he = False
            for node_idx in groups[i]:
                node = nodes[node_idx]
                # 检查所有邻居
                for j in range(n - 1):
                    neighbor_node = self._generate_neighbor(node, j)
                    neighbor_idx = node_to_idx.get(neighbor_node, -1)

                    if neighbor_idx != -1:
                        # 检查症状是否为(0,1)
                        if ((node, neighbor_node) in self.symptoms and
                                (neighbor_node, node) in self.symptoms and
                                self.symptoms[(node, neighbor_node)] == 0 and
                                self.symptoms[(neighbor_node, node)] == 1):
                            he = True
                            break
                if he:
                    break

            if he and not st2[i]:
                chun.append(i)
                st2[i] = True
                guzhang += len(groups[i])

        # 第二轮检测：fuxing1() - 大小判断
        while True:
            res = 0
            for i in range(total_groups):
                if not st2[i] and len(groups[i]) > t - guzhang:
                    xi.append(i)
                    wuguzhang += len(groups[i])
                    st2[i] = True
                    res += 1

            if res == 0:
                break

        # 第三轮检测：fuxing3() - 假设法
        while True:
            res = 0
            for i in range(total_groups):
                if not st2[i]:
                    # 收集相邻的未分类组的节点数
                    cntt = 0
                    for node_idx in groups[i]:
                        node = nodes[node_idx]
                        # 检查所有邻居
                        for j in range(n - 1):
                            neighbor_node = self._generate_neighbor(node, j)
                            neighbor_idx = node_to_idx.get(neighbor_node, -1)

                            if neighbor_idx != -1:
                                neighbor_group = group_of_node.get(neighbor_idx, -1)
                                if neighbor_group != -1 and not st2[neighbor_group] and neighbor_group != i:
                                    cntt += len(groups[neighbor_group])

                    if cntt > t - wuguzhang - guzhang:
                        chun.append(i)
                        res += 1
                        st2[i] = True
                        guzhang += len(groups[i])

            if res == 0:
                break

        # 收集故障节点 - 确保不超过t个
        faulty_nodes = []
        for group_idx in chun:
            for node_idx in groups[group_idx]:
                if len(faulty_nodes) < t:
                    faulty_nodes.append(nodes[node_idx])
                else:
                    break
            if len(faulty_nodes) >= t:
                break

        # 确保诊断节点数不超过t
        if len(faulty_nodes) > t:
            faulty_nodes = faulty_nodes[:t]

        return faulty_nodes


class GeneralNetwork:
    """通用网络类"""

    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]]):
        self.nodes = nodes
        self.edges = edges
        self.adj_list = self._generate_adjacency_list()
        # 估计n值（用于邻居生成）
        if nodes:
            self.n = len(str(nodes[0]))
        else:
            self.n = 0

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

        # 准确率 (Accuracy) - ACCR
        if self.TP + self.TN + self.FP + self.FN > 0:
            metrics['ACCR'] = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        else:
            metrics['ACCR'] = 0.0

        # 真负率 (True Negative Rate) - TNR
        if self.TN + self.FP > 0:
            metrics['TNR'] = self.TN / (self.TN + self.FP)
        else:
            metrics['TNR'] = 0.0

        # 假正率 (False Positive Rate) - FPR
        if self.TN + self.FP > 0:
            metrics['FPR'] = self.FP / (self.TN + self.FP)
        else:
            metrics['FPR'] = 0.0

        # 真正率 (True Positive Rate / Recall) - TPR
        if self.TP + self.FN > 0:
            metrics['TPR'] = self.TP / (self.TP + self.FN)
        else:
            metrics['TPR'] = 0.0

        # 精确率 (Precision) - Precision
        if self.TP + self.FP > 0:
            metrics['Precision'] = self.TP / (self.TP + self.FP)
        else:
            metrics['Precision'] = 0.0

        # 计算F1分数
        if metrics['TPR'] + metrics['Precision'] > 0:
            metrics['F1分数'] = 2 * (metrics['TPR'] * metrics['Precision']) / (metrics['TPR'] + metrics['Precision'])
        else:
            metrics['F1分数'] = 0.0

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
        """功能1：合成网络诊断（使用故障点数）"""
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
            total_nodes = len(self.network.nodes)
            if num_faults > total_nodes:
                num_faults = max(1, int(total_nodes * 0.1))

            self.true_faulty_nodes = set(random.sample(self.network.nodes, num_faults))

            # 生成症状集
            pmc_model = PMCModel(self.network)
            symptoms = pmc_model.generate_symptoms(self.true_faulty_nodes)

            # 运行诊断
            diagnoser = TRFI_PMC(self.network, symptoms)
            diagnosed_nodes = diagnoser.run(num_faults)
            self.diagnosed_faulty_nodes = set(diagnosed_nodes)

            # 确保诊断节点数不超过故障节点数
            if len(self.diagnosed_faulty_nodes) > num_faults:
                # 如果超过，只保留前num_faults个
                diagnosed_list = list(self.diagnosed_faulty_nodes)[:num_faults]
                self.diagnosed_faulty_nodes = set(diagnosed_list)
                print(f"警告：诊断出 {len(diagnosed_nodes)} 个故障节点，超过设定的 {num_faults} 个，已截断")

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
                "故障节点数": num_faults,
                "节点总数": total_nodes,
                "故障比例": num_faults / total_nodes,
                "真实故障节点": list(self.true_faulty_nodes),
                "诊断故障节点": list(self.diagnosed_faulty_nodes),
                "指标": metrics,
                "运行时间": exp_time
            }
            all_results.append(result)
            self.metrics_history.append(metrics)

            # 显示本次实验指标
            print(f"ACCR: {metrics['ACCR']:.4f}")
            print(f"TNR: {metrics['TNR']:.4f}")
            print(f"FPR: {metrics['FPR']:.4f}")
            print(f"TPR: {metrics['TPR']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"F1分数: {metrics['F1分数']:.4f}")
            print(f"诊断节点数: {len(self.diagnosed_faulty_nodes)} (不超过设定的 {num_faults} 个)")
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
                print(f"{key}: {avg_value:.4f}")
            print(f"平均运行时间: {np.mean(self.experiment_times):.2f}秒")

        # 保存数据到Excel
        self._save_function1_data(all_results, n, num_faults)

        return {
            "实验结果": all_results,
            "平均指标": avg_metrics if avg_metrics else self.metrics_history[0],
            "网络信息": network_info
        }

    def function2(self, nodes: List[str], edges: List[Tuple[str, str]],
                  symptoms: Dict[Tuple[str, str], int], t: int = None) -> Dict[str, Any]:
        """功能2：真实网络诊断（输入症状集）"""
        print(f"功能2：真实网络诊断（输入症状集）")

        # 创建通用网络
        self.network = GeneralNetwork(nodes, edges)

        # 如果没有指定t，则使用节点数的10%作为估计
        if t is None:
            t = max(1, len(nodes) // 10)

        # 运行诊断
        diagnoser = TRFI_PMC(self.network, symptoms)
        diagnosed_nodes = diagnoser.run(t)
        self.diagnosed_faulty_nodes = set(diagnosed_nodes)

        # 确保诊断节点数不超过t
        if len(self.diagnosed_faulty_nodes) > t:
            # 如果超过，只保留前t个
            diagnosed_list = list(self.diagnosed_faulty_nodes)[:t]
            self.diagnosed_faulty_nodes = set(diagnosed_list)
            print(f"警告：诊断出 {len(diagnosed_nodes)} 个故障节点，超过设定的 {t} 个，已截断")

        print(f"诊断度 t: {t}")
        print(f"节点总数: {len(nodes)}")
        print(f"诊断出 {len(self.diagnosed_faulty_nodes)} 个故障节点 (不超过设定的 {t} 个)")
        print(f"诊断故障比例: {len(self.diagnosed_faulty_nodes) / len(nodes) if len(nodes) > 0 else 0:.4f}")

        # 保存结果到Excel
        self._save_function2_data(nodes, diagnosed_nodes, t)

        return {
            "诊断故障节点": list(self.diagnosed_faulty_nodes),
            "诊断度t": t,
            "节点总数": len(nodes),
            "诊断故障比例": len(self.diagnosed_faulty_nodes) / len(nodes) if len(nodes) > 0 else 0,
            "网络信息": {
                "节点数": len(nodes),
                "边数": len(edges),
                "诊断故障节点数": len(self.diagnosed_faulty_nodes)
            }
        }

    def _save_function1_data(self, results: List[Dict], n: int, num_faults: int):
        """保存功能1的数据到Excel"""
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存指标数据
        metrics_data = []
        for result in results:
            metrics = result["指标"]
            metrics_data.append({
                "实验编号": result["实验编号"],
                "网络维度": n,
                "故障节点数": num_faults,
                "节点总数": result["节点总数"],
                "故障比例": result["故障比例"],
                "ACCR": metrics['ACCR'],
                "TNR": metrics['TNR'],
                "FPR": metrics['FPR'],
                "TPR": metrics['TPR'],
                "Precision": metrics['Precision'],
                "F1分数": metrics['F1分数'],
                "诊断节点数": len(result["诊断故障节点"]),
                "运行时间": result["运行时间"]
            })

        metrics_df = pd.DataFrame(metrics_data)
        excel_path = f"results/功能1_性能指标_{timestamp}.xlsx"
        metrics_df.to_excel(excel_path, index=False, float_format="%.4f")
        print(f"性能指标已保存到: {excel_path}")

        # 保存最后一次实验的详细数据
        if results:
            last_result = results[-1]

            # 保存故障节点对比
            faults_data = []
            for node in last_result["真实故障节点"]:
                faults_data.append({
                    "节点": node,
                    "类型": "真实故障",
                    "是否被诊断出": "是" if node in last_result["诊断故障节点"] else "否"
                })

            for node in last_result["诊断故障节点"]:
                if node not in last_result["真实故障节点"]:
                    faults_data.append({
                        "节点": node,
                        "类型": "误诊故障",
                        "是否被诊断出": "是"
                    })

            faults_df = pd.DataFrame(faults_data)
            faults_excel_path = f"results/功能1_故障节点对比_{timestamp}.xlsx"
            faults_df.to_excel(faults_excel_path, index=False)
            print(f"故障节点对比已保存到: {faults_excel_path}")

        print(f"所有数据已保存到 results/ 目录")

    def _save_function2_data(self, nodes: List[str], diagnosed_nodes: List[str], t: int):
        """保存功能2的数据到Excel"""
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存诊断结果
        results_data = []
        for i, node in enumerate(nodes):
            results_data.append({
                "节点编号": i + 1,
                "节点": node,
                "是否被诊断为故障": "是" if node in diagnosed_nodes else "否"
            })

        results_df = pd.DataFrame(results_data)
        excel_path = f"results/功能2_诊断结果_{timestamp}.xlsx"
        results_df.to_excel(excel_path, index=False)

        # 保存汇总信息
        summary_data = [{
            "节点总数": len(nodes),
            "诊断故障节点数": len(diagnosed_nodes),
            "诊断故障比例": len(diagnosed_nodes) / len(nodes) if len(nodes) > 0 else 0,
            "诊断度t": t,
            "是否超过诊断度": "是" if len(diagnosed_nodes) > t else "否"
        }]

        summary_df = pd.DataFrame(summary_data)
        summary_excel_path = f"results/功能2_诊断汇总_{timestamp}.xlsx"
        summary_df.to_excel(summary_excel_path, index=False, float_format="%.4f")

        print(f"诊断结果已保存到: {excel_path}")
        print(f"诊断汇总已保存到: {summary_excel_path}")


class TRFI_PMC_GUI:
    """TRFI-PMC系统GUI界面"""

    def __init__(self):
        self.system = TRFI_PMC_System()
        self.current_result = None

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("冒泡排序网络故障诊断系统 (TRFI-PMC)")
        self.root.geometry("1000x800")

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
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

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
        self.exp_var1 = tk.StringVar(value="5")
        ttk.Entry(func1_frame, textvariable=self.exp_var1, width=12).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        func1_btn = ttk.Button(func1_frame, text="运行功能1", command=self.run_function1,
                               style='Accent.TButton')
        func1_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # 功能2：真实网络诊断（输入症状集）
        func2_frame = ttk.LabelFrame(func_frame, text="功能2: 真实网络诊断（输入症状集）", padding="10")
        func2_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 添加文件格式说明
        info_label = ttk.Label(func2_frame,
                               text="文件格式说明：\n1. 节点文件：Excel/CSV格式，只需一列数字\n2. 边文件：Excel/CSV格式，只需两列数字\n3. 症状集文件：Excel/CSV格式，三列数字(源节点,目标节点,测试结果)",
                               font=('Arial', 8), foreground='#666666')
        info_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 5))

        ttk.Label(func2_frame, text="节点文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.nodes_file_var2 = tk.StringVar()
        nodes_entry2 = ttk.Entry(func2_frame, textvariable=self.nodes_file_var2, width=25)
        nodes_entry2.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_nodes_file2, width=8).grid(row=1, column=2, padx=5,
                                                                                            pady=5)

        ttk.Label(func2_frame, text="边文件:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.edges_file_var2 = tk.StringVar()
        edges_entry2 = ttk.Entry(func2_frame, textvariable=self.edges_file_var2, width=25)
        edges_entry2.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_edges_file2, width=8).grid(row=2, column=2, padx=5,
                                                                                            pady=5)

        ttk.Label(func2_frame, text="症状集文件:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.symptoms_file_var2 = tk.StringVar()
        symptoms_entry2 = ttk.Entry(func2_frame, textvariable=self.symptoms_file_var2, width=25)
        symptoms_entry2.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(func2_frame, text="浏览", command=self.select_symptoms_file2, width=8).grid(row=3, column=2, padx=5,
                                                                                               pady=5)

        ttk.Label(func2_frame, text="诊断度 t:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.t_var2 = tk.StringVar(value="5")
        ttk.Entry(func2_frame, textvariable=self.t_var2, width=12).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(func2_frame, text="(故障节点数上限)").grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)

        func2_btn = ttk.Button(func2_frame, text="运行功能2", command=self.run_function2,
                               style='Accent.TButton')
        func2_btn.grid(row=5, column=0, columnspan=3, pady=(10, 0))

        # 控制按钮框架
        control_frame = ttk.LabelFrame(func_frame, text="控制选项", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.export_btn = ttk.Button(control_frame, text="导出当前结果数据",
                                     command=self.export_data, state=tk.DISABLED, width=20)
        self.export_btn.grid(row=0, column=0, pady=5, padx=5)

        self.clear_btn = ttk.Button(control_frame, text="清空所有结果",
                                    command=self.clear_results, width=20)
        self.clear_btn.grid(row=0, column=1, pady=5, padx=5)

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
        columns = ("实验", "ACCR", "TPR", "Precision", "F1分数", "TNR", "FPR", "诊断节点数")
        self.metrics_tree = ttk.Treeview(metrics_tab, columns=columns, show="headings", height=12)

        # 设置列标题
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            if col in ["ACCR", "TPR", "Precision", "F1分数", "TNR", "FPR"]:
                self.metrics_tree.column(col, width=70, anchor="center")
            else:
                self.metrics_tree.column(col, width=60, anchor="center")

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
        for metric in ["ACCR", "TPR", "Precision", "F1分数", "TNR", "FPR"]:
            frame = ttk.Frame(avg_frame)
            frame.pack(side="left", padx=8)
            ttk.Label(frame, text=f"{metric}:").pack(side="left")
            self.avg_labels[metric] = ttk.Label(frame, text="0.0000", foreground="blue", font=('Arial', 9, 'bold'))
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

    def select_nodes_file2(self):
        """选择功能2的节点文件"""
        filename = filedialog.askopenfilename(
            title="选择节点文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.nodes_file_var2.set(filename)
            self.log_message(f"已选择节点文件: {filename}", "INFO")

    def select_edges_file2(self):
        """选择功能2的边文件"""
        filename = filedialog.askopenfilename(
            title="选择边文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.edges_file_var2.set(filename)
            self.log_message(f"已选择边文件: {filename}", "INFO")

    def select_symptoms_file2(self):
        """选择功能2的症状集文件"""
        filename = filedialog.askopenfilename(
            title="选择症状集文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.symptoms_file_var2.set(filename)
            self.log_message(f"已选择症状集文件: {filename}", "INFO")

    def run_function1(self):
        """运行功能1"""
        try:
            self.status_var.set("正在运行功能1...")
            self.log_message("开始运行功能1: 合成网络诊断", "INFO")

            # 获取参数
            n = int(self.n_var.get())
            num_faults = int(self.faults_var1.get())
            num_experiments = int(self.exp_var1.get())

            # 验证参数
            if num_faults <= 0:
                messagebox.showerror("输入错误", "故障节点数必须大于0")
                self.status_var.set("就绪")
                return

            self.log_message(f"参数: n={n}, 故障节点数={num_faults}, 实验次数={num_experiments}", "INFO")

            # 清空结果
            self.metrics_tree.delete(*self.metrics_tree.get_children())
            for metric in self.avg_labels:
                self.avg_labels[metric].config(text="0.0000")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function1_thread,
                                      args=(n, num_faults, num_experiments))
            thread.daemon = True
            thread.start()

        except ValueError as e:
            messagebox.showerror("输入错误", f"请输入有效的参数: {e}")
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

            # 启用导出按钮
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
            nodes_file = self.nodes_file_var2.get()
            edges_file = self.edges_file_var2.get()
            symptoms_file = self.symptoms_file_var2.get()

            if not nodes_file or not edges_file or not symptoms_file:
                messagebox.showerror("错误", "请选择所有必需的文件")
                self.status_var.set("就绪")
                return

            # 检查文件是否存在
            for file_path, file_type in [(nodes_file, "节点文件"), (edges_file, "边文件"),
                                         (symptoms_file, "症状集文件")]:
                if not os.path.exists(file_path):
                    messagebox.showerror("错误", f"{file_type}不存在: {file_path}")
                    self.status_var.set("就绪")
                    return
                if not os.path.isfile(file_path):
                    messagebox.showerror("错误", f"{file_type}不是有效的文件: {file_path}")
                    self.status_var.set("就绪")
                    return

            # 获取诊断度t
            try:
                t = int(self.t_var2.get())
                if t <= 0:
                    raise ValueError("诊断度t必须大于0")
            except ValueError as e:
                messagebox.showerror("输入错误", f"请输入有效的诊断度t: {e}")
                self.status_var.set("就绪")
                return

            self.log_message(f"节点文件: {nodes_file}", "INFO")
            self.log_message(f"边文件: {edges_file}", "INFO")
            self.log_message(f"症状集文件: {symptoms_file}", "INFO")
            self.log_message(f"诊断度 t: {t}", "INFO")

            # 在新线程中运行
            thread = threading.Thread(target=self._run_function2_thread,
                                      args=(nodes_file, edges_file, symptoms_file, t))
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"运行功能2时出错: {e}")
            self.status_var.set("就绪")

    def _run_function2_thread(self, nodes_file: str, edges_file: str, symptoms_file: str, t: int):
        """功能2线程函数"""
        try:
            # 读取节点文件
            self.log_message("正在读取节点文件...", "INFO")
            if nodes_file.endswith('.csv'):
                try:
                    nodes_df = pd.read_csv(nodes_file, header=None)
                except:
                    nodes_df = pd.read_csv(nodes_file, header=None, encoding='gbk')
            else:
                nodes_df = pd.read_excel(nodes_file, header=None)

            # 处理节点数据
            nodes = []
            for val in nodes_df.iloc[:, 0]:
                if pd.isna(val):
                    continue
                node_str = str(val).strip()
                if node_str:
                    nodes.append(node_str)

            self.log_message(f"读取完成: {len(nodes)}个节点", "INFO")

            if len(nodes) == 0:
                raise ValueError("节点文件中没有有效的节点数据")

            # 读取边文件
            self.log_message("正在读取边文件...", "INFO")
            if edges_file.endswith('.csv'):
                try:
                    edges_df = pd.read_csv(edges_file, header=None)
                except:
                    edges_df = pd.read_csv(edges_file, header=None, encoding='gbk')
            else:
                edges_df = pd.read_excel(edges_file, header=None)

            # 处理边数据
            edges = []
            for _, row in edges_df.iterrows():
                if len(row) >= 2:
                    u = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else None
                    v = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else None
                    if u and v:
                        edges.append((u, v))

            self.log_message(f"读取完成: {len(edges)}条边", "INFO")

            # 读取症状集文件
            self.log_message("正在读取症状集文件...", "INFO")
            if symptoms_file.endswith('.csv'):
                try:
                    symptoms_df = pd.read_csv(symptoms_file, header=None)
                except:
                    symptoms_df = pd.read_csv(symptoms_file, header=None, encoding='gbk')
            else:
                symptoms_df = pd.read_excel(symptoms_file, header=None)

            # 处理症状集数据
            symptoms = {}
            for _, row in symptoms_df.iterrows():
                if len(row) >= 3:
                    u = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else None
                    v = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else None
                    sigma = int(row.iloc[2]) if not pd.isna(row.iloc[2]) else None

                    if u and v is not None and sigma is not None:
                        symptoms[(u, v)] = sigma

            self.log_message(f"读取完成: {len(symptoms)}条测试结果", "INFO")

            if len(symptoms) == 0:
                raise ValueError("症状集文件中没有有效的测试结果")

            self.log_message(f"数据读取完成: {len(nodes)}个节点, {len(edges)}条边, {len(symptoms)}条测试结果", "INFO")
            self.log_message(f"节点示例: {nodes[:5] if len(nodes) > 5 else nodes}", "INFO")
            self.log_message(f"边示例: {edges[:5] if len(edges) > 5 else edges}", "INFO")
            self.log_message(f"症状示例: {list(symptoms.items())[:5] if len(symptoms) > 5 else list(symptoms.items())}",
                             "INFO")

            # 运行功能2
            result = self.system.function2(nodes, edges, symptoms, t)
            self.current_result = result

            # 更新UI
            self.root.after(0, self._update_function2_results, result)

            # 启用导出按钮
            self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL))

            self.log_message("功能2运行完成", "SUCCESS")
            self.status_var.set("就绪")

        except Exception as e:
            error_msg = f"功能2运行出错: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            self.log_message(error_msg, "ERROR")
            self.status_var.set("就绪")

    def _update_function1_results(self, result: Dict[str, Any]):
        """更新功能1的结果显示"""
        # 清空文本区域
        self.info_text.delete(1.0, tk.END)
        self.true_faults_text.delete(1.0, tk.END)
        self.diagnosed_faults_text.delete(1.0, tk.END)

        # 显示网络信息
        network_info = result["网络信息"]
        info_str = "功能1: 合成网络诊断\n\n"
        info_str += "网络信息:\n"
        for key, value in network_info.items():
            info_str += f"  {key}: {value}\n"
        self.info_text.insert(tk.END, info_str)

        # 更新指标表格
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        for exp_result in result["实验结果"]:
            exp_num = exp_result["实验编号"]
            metrics = exp_result["指标"]

            values = (exp_num,
                      f"{metrics['ACCR']:.4f}",
                      f"{metrics['TPR']:.4f}",
                      f"{metrics['Precision']:.4f}",
                      f"{metrics['F1分数']:.4f}",
                      f"{metrics['TNR']:.4f}",
                      f"{metrics['FPR']:.4f}",
                      f"{len(exp_result['诊断故障节点'])}/{exp_result['故障节点数']}")

            self.metrics_tree.insert("", "end", values=values)

        # 更新平均指标
        if "平均指标" in result and result["平均指标"]:
            avg_metrics = result["平均指标"]
            for metric, value in avg_metrics.items():
                if metric in self.avg_labels:
                    self.avg_labels[metric].config(text=f"{value:.4f}")
        elif result["实验结果"]:
            # 计算平均指标
            avg_metrics = {}
            for metric in ["ACCR", "TPR", "Precision", "F1分数", "TNR", "FPR"]:
                values = [r["指标"][metric] for r in result["实验结果"]]
                avg_metrics[metric] = np.mean(values)

            for metric, value in avg_metrics.items():
                if metric in self.avg_labels:
                    self.avg_labels[metric].config(text=f"{value:.4f}")

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
            self.diagnosed_faults_text.insert(tk.END, f"总数: {len(diagnosed_faults)} (不超过设定的 {last_exp['故障节点数']} 个)\n")
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

        # 显示网络信息
        network_info = result.get("网络信息", {})
        info_str = "功能2: 真实网络诊断（输入症状集）\n\n"
        info_str += f"诊断度 t: {result['诊断度t']}\n"
        info_str += f"节点总数: {result['节点总数']}\n"
        info_str += f"诊断故障节点数: {len(result['诊断故障节点'])} (不超过设定的 {result['诊断度t']} 个)\n"
        info_str += f"诊断故障比例: {result['诊断故障比例']:.4f}\n\n"
        info_str += "网络信息:\n"
        for key, value in network_info.items():
            info_str += f"  {key}: {value}\n"

        self.info_text.insert(tk.END, info_str)

        # 清空指标表格
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for metric in self.avg_labels:
            self.avg_labels[metric].config(text="N/A")

        # 显示诊断结果
        diagnosed_faults = result["诊断故障节点"]
        self.diagnosed_faults_text.insert(tk.END, f"诊断故障节点 ({len(diagnosed_faults)}个，不超过设定的 {result['诊断度t']} 个):\n")
        self.diagnosed_faults_text.insert(tk.END, "-" * 30 + "\n")
        for i, node in enumerate(diagnosed_faults, 1):
            self.diagnosed_faults_text.insert(tk.END, f"{i}. {node}\n")

    def export_data(self):
        """导出当前结果数据"""
        if not self.current_result:
            messagebox.showinfo("提示", "没有可导出的结果数据")
            return

        # 根据当前结果类型选择导出方式
        try:
            if "实验结果" in self.current_result:
                # 功能1的结果
                results = self.current_result["实验结果"]
                if results:
                    # 创建数据框
                    data = []
                    for result in results:
                        row = {
                            "实验编号": result["实验编号"],
                            "网络维度": result["网络信息"]["维度"],
                            "节点总数": result["节点总数"],
                            "故障节点数": result["故障节点数"],
                            "故障比例": result["故障比例"],
                            "诊断节点数": len(result["诊断故障节点"]),
                        }
                        # 添加指标
                        for key, value in result["指标"].items():
                            row[key] = value
                        row["运行时间"] = result["运行时间"]
                        data.append(row)

                    df = pd.DataFrame(data)

                    # 选择保存路径
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".xlsx",
                        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                        initialfile="功能1_实验结果.xlsx"
                    )

                    if file_path:
                        df.to_excel(file_path, index=False, float_format="%.4f")
                        self.log_message(f"实验结果已导出到: {file_path}", "SUCCESS")
                        messagebox.showinfo("导出成功", f"数据已成功导出到:\n{file_path}")
            else:
                # 功能2的结果
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                    initialfile="功能2_实验结果.xlsx"
                )

                if file_path:
                    # 功能2的结果
                    data = [self.current_result]
                    df = pd.DataFrame(data)
                    df.to_excel(file_path, index=False, float_format="%.4f")

                    self.log_message(f"功能2实验结果已导出到: {file_path}", "SUCCESS")
                    messagebox.showinfo("导出成功", f"数据已成功导出到:\n{file_path}")

        except Exception as e:
            self.log_message(f"导出数据时出错: {e}", "ERROR")
            messagebox.showerror("错误", f"导出数据时出错: {e}")

    def clear_results(self):
        """清空所有结果"""
        if messagebox.askyesno("确认", "确定要清空所有结果吗？"):
            # 清空表格
            self.metrics_tree.delete(*self.metrics_tree.get_children())

            # 清空文本区域
            self.info_text.delete(1.0, tk.END)
            self.true_faults_text.delete(1.0, tk.END)
            self.diagnosed_faults_text.delete(1.0, tk.END)

            # 重置平均指标
            for metric in self.avg_labels:
                self.avg_labels[metric].config(text="0.0000")

            # 清空系统数据
            self.system.metrics_history = []
            self.system.experiment_times = []
            self.current_result = None

            # 禁用导出按钮
            self.export_btn.config(state=tk.DISABLED)

            self.log_message("所有结果已清空", "INFO")

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
    print("指标: ACCR, TNR, FPR, TPR, Precision, F1分数")
    print("=" * 70)
    print("\nTRFI-PMC算法效率分析:")
    print("1. 算法原理: 基于(0,0)边分组 + 三轮检测")
    print("2. 效率限制因素:")
    print("   - 依赖(0,0)边的连通性")
    print("   - PMC模型测试结果包含随机性")
    print("   - 诊断度t需要准确估计")
    print("   - 故障率过高时诊断效果下降")
    print("3. 改进建议:")
    print("   - 进行多次实验取平均值")
    print("   - 优化网络结构选择")
    print("=" * 70)

    # 设置随机种子
    random.seed(42)

    # 创建并运行GUI
    app = TRFI_PMC_GUI()
    app.run()


if __name__ == "__main__":
    main()
