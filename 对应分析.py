import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 读取数据
# ============================================
file_path = r'C:\Users\lxh75\对应分析.xlsx'

# 读取Sheet1和Sheet2
df1 = pd.read_excel(file_path, sheet_name='Sheet1')
df2 = pd.read_excel(file_path, sheet_name='Sheet2')

print("=" * 80)
print("对应分析")
print("=" * 80)


# ============================================
# 对应分析函数
# ============================================
def correspondence_analysis(contingency_table, title="对应分析"):
    """
    执行对应分析
    参数:
        contingency_table: 列联表
        title: 图表标题
    返回:
        行坐标、列坐标、惯量等
    """
    # 计算总频数
    N = contingency_table.sum().sum()

    # 计算概率矩阵P
    P = contingency_table / N

    # 计算行质量r和列质量c
    r = P.sum(axis=1).values  # 行质量（行边际概率）
    c = P.sum(axis=0).values  # 列质量（列边际概率）

    # 计算标准化残差矩阵
    Dr_inv_sqrt = np.diag(1 / np.sqrt(r))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(c))

    # 计算S矩阵
    S = Dr_inv_sqrt @ (P - np.outer(r, c)) @ Dc_inv_sqrt

    # SVD分解
    U, s, Vt = np.linalg.svd(S, full_matrices=False)

    # 保留有效维度
    dims = min(len(s[s > 1e-10]), min(P.shape) - 1)
    s = s[:dims]
    U = U[:, :dims]
    V = Vt.T[:, :dims]

    # 计算惯量
    total_inertia = np.sum(s ** 2)
    inertia_percent = (s ** 2 / total_inertia * 100)

    # 计算行坐标和列坐标
    row_coords = np.diag(1 / np.sqrt(r)) @ U @ np.diag(s)
    col_coords = np.diag(1 / np.sqrt(c)) @ V @ np.diag(s)

    # 创建DataFrame
    row_coords_df = pd.DataFrame(row_coords,
                                 index=contingency_table.index,
                                 columns=[f'Dim{i + 1}' for i in range(dims)])
    col_coords_df = pd.DataFrame(col_coords,
                                 index=contingency_table.columns,
                                 columns=[f'Dim{i + 1}' for i in range(dims)])

    return {
        'row_coords': row_coords_df,
        'col_coords': col_coords_df,
        'singular_values': s,
        'inertia': s ** 2,
        'inertia_percent': inertia_percent,
        'total_inertia': total_inertia,
        'row_mass': r,
        'col_mass': c,
        'contingency_table': contingency_table,
        'P': P,
        'dims': dims
    }


def print_row_profile(contingency_table, title="行剖面"):
    """打印行剖面"""
    row_profile = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    print(f"\n{title}:")
    print("=" * 60)
    for idx in row_profile.index:
        print(f"\n{idx}:")
        for col in row_profile.columns:
            print(f"  {col}: {row_profile.loc[idx, col]:.4f} ({row_profile.loc[idx, col] * 100:.2f}%)")


def print_col_profile(contingency_table, title="列剖面"):
    """打印列剖面"""
    col_profile = contingency_table.div(contingency_table.sum(axis=0), axis=1)
    print(f"\n{title}:")
    print("=" * 60)
    for col in col_profile.columns:
        print(f"\n{col}:")
        for idx in col_profile.index:
            print(f"  {idx}: {col_profile.loc[idx, col]:.4f} ({col_profile.loc[idx, col] * 100:.2f}%)")


def print_inertia_summary(ca_result, title="惯量摘要"):
    """打印惯量摘要"""
    print(f"\n{title}:")
    print("=" * 60)
    print(f"总惯量: {ca_result['total_inertia']:.6f}")
    print("\n维度\t奇异值\t惯量\t百分比\t累计百分比")
    print("-" * 50)
    cum_percent = 0
    for i in range(ca_result['dims']):
        cum_percent += ca_result['inertia_percent'][i]
        print(f"Dim{i + 1}\t{ca_result['singular_values'][i]:.4f}\t"
              f"{ca_result['inertia'][i]:.6f}\t"
              f"{ca_result['inertia_percent'][i]:.2f}%\t"
              f"{cum_percent:.2f}%")


def plot_ca(ca_result, title="对应分析图"):
    """绘制对应分析图"""
    row_coords = ca_result['row_coords']
    col_coords = ca_result['col_coords']

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制行点（学历）
    ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
               s=200, c='red', marker='s', alpha=0.7, label='学历 (行)')
    for idx, (x, y) in enumerate(zip(row_coords.iloc[:, 0], row_coords.iloc[:, 1])):
        ax.annotate(row_coords.index[idx], (x, y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='red')

    # 绘制列点（保险产品购买倾向）
    ax.scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1],
               s=200, c='blue', marker='o', alpha=0.7, label='购买倾向 (列)')
    for idx, (x, y) in enumerate(zip(col_coords.iloc[:, 0], col_coords.iloc[:, 1])):
        ax.annotate(col_coords.index[idx], (x, y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='blue')

    # 添加原点线
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 设置标签和标题
    ax.set_xlabel(f'维度1 ({ca_result["inertia_percent"][0]:.2f}%)', fontsize=12)
    ax.set_ylabel(f'维度2 ({ca_result["inertia_percent"][1]:.2f}%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================
# Sheet1 对应分析
# ============================================
print("\n" + "=" * 80)
print("Sheet1: 学历 vs 保险产品购买倾向")
print("=" * 80)

# 创建列联表
contingency_table1 = pd.crosstab(df1['学历'], df1['保险产品购买倾向'])
print("\n列联表:")
print(contingency_table1)

# 打印行剖面
print_row_profile(contingency_table1, "学历的行剖面")

# 打印列剖面
print_col_profile(contingency_table1, "保险产品购买倾向的列剖面")

# 执行对应分析
ca_result1 = correspondence_analysis(contingency_table1, "学历 vs 保险产品购买倾向")

# 打印惯量摘要
print_inertia_summary(ca_result1, "惯量摘要")

# 打印行坐标
print("\n行坐标 (学历):")
print(ca_result1['row_coords'])

# 打印列坐标
print("\n列坐标 (保险产品购买倾向):")
print(ca_result1['col_coords'])

# 绘制对应分析图
plot_ca(ca_result1, "对应分析图: 学历 vs 保险产品购买倾向")

# ============================================
# Sheet2 对应分析
# ============================================
print("\n" + "=" * 80)
print("Sheet2: 地区 vs 消费结构 (消费支出)")
print("=" * 80)

# 对Sheet2进行对应分析
# 创建地区与消费结构的列联表（使用消费支出作为权重）
pivot_table2 = df2.pivot_table(index='地区', columns='消费结构', values='消费支出', aggfunc='sum')
pivot_table2 = pivot_table2.fillna(0)

print(f"\n数据维度: {pivot_table2.shape[0]}个地区 × {pivot_table2.shape[1]}个消费类别")
print("\n消费结构类别:", list(pivot_table2.columns))

# 打印部分数据
print("\n前10行数据预览:")
print(pivot_table2.head(10))

# 打印行剖面（部分，避免输出过长）
print("\n行剖面 (各地区消费结构占比) - 前5个地区:")
row_profile2 = pivot_table2.div(pivot_table2.sum(axis=1), axis=0)
for idx in row_profile2.index[:5]:
    print(f"\n{idx}:")
    for col in row_profile2.columns:
        print(f"  {col}: {row_profile2.loc[idx, col]:.4f} ({row_profile2.loc[idx, col] * 100:.2f}%)")

# 打印列剖面（部分）
print("\n列剖面 (各消费类别在各地区占比) - 前5个消费类别:")
col_profile2 = pivot_table2.div(pivot_table2.sum(axis=0), axis=1)
for col in col_profile2.columns[:5]:
    print(f"\n{col}:")
    # 只显示前5个地区
    top_regions = col_profile2[col].nlargest(5)
    for idx, val in top_regions.items():
        print(f"  {idx}: {val:.4f} ({val * 100:.2f}%)")

# 执行对应分析
ca_result2 = correspondence_analysis(pivot_table2, "地区 vs 消费结构")

# 打印惯量摘要
print_inertia_summary(ca_result2, "惯量摘要")

# 打印部分行坐标（前10个地区）
print("\n行坐标 (地区) - 前10个:")
print(ca_result2['row_coords'].head(10))

# 打印列坐标
print("\n列坐标 (消费结构):")
print(ca_result2['col_coords'])

# 绘制对应分析图
# 由于地区较多，我们调整图形大小并只显示部分标签
fig, ax = plt.subplots(figsize=(16, 12))

row_coords2 = ca_result2['row_coords']
col_coords2 = ca_result2['col_coords']

# 绘制行点（地区）
ax.scatter(row_coords2.iloc[:, 0], row_coords2.iloc[:, 1],
           s=100, c='red', marker='s', alpha=0.6, label='地区 (行)')

# 只标注部分地区（避免过于拥挤）
row_coords2_copy = row_coords2.copy()
row_coords2_copy['abs_dim1'] = np.abs(row_coords2_copy.iloc[:, 0])
row_coords2_copy['abs_dim2'] = np.abs(row_coords2_copy.iloc[:, 1])
top_regions = row_coords2_copy.nlargest(20, 'abs_dim1').index.tolist() + \
              row_coords2_copy.nlargest(10, 'abs_dim2').index.tolist()
top_regions = list(set(top_regions))

for idx in top_regions:
    x = row_coords2.loc[idx, 'Dim1']
    y = row_coords2.loc[idx, 'Dim2']
    ax.annotate(idx, (x, y), xytext=(3, 3), textcoords='offset points',
                fontsize=8, alpha=0.7)

# 绘制列点（消费结构）
ax.scatter(col_coords2.iloc[:, 0], col_coords2.iloc[:, 1],
           s=300, c='blue', marker='o', alpha=0.8, label='消费结构 (列)')
for idx, (x, y) in enumerate(zip(col_coords2.iloc[:, 0], col_coords2.iloc[:, 1])):
    ax.annotate(col_coords2.index[idx], (x, y),
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, fontweight='bold', color='blue')

# 添加原点线
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# 设置标签和标题
ax.set_xlabel(f'维度1 ({ca_result2["inertia_percent"][0]:.2f}%)', fontsize=12)
ax.set_ylabel(f'维度2 ({ca_result2["inertia_percent"][1]:.2f}%)', fontsize=12)
ax.set_title('对应分析图: 地区 vs 消费结构', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 额外：Sheet2的汇总统计
# ============================================
print("\n" + "=" * 80)
print("Sheet2 数据汇总统计")
print("=" * 80)

# 按消费结构汇总
summary_by_category = df2.groupby('消费结构')['消费支出'].agg(['sum', 'mean', 'std', 'count'])
print("\n按消费结构汇总:")
print(summary_by_category)

# 按地区汇总
summary_by_region = df2.groupby('地区')['消费支出'].sum().sort_values(ascending=False)
print("\n按地区总消费支出排名 (前10):")
print(summary_by_region.head(10))

# 各地区消费结构占比热力图
fig, ax = plt.subplots(figsize=(14, 10))
# 计算占比
heatmap_data = pivot_table2.div(pivot_table2.sum(axis=1), axis=0) * 100
# 取前15个地区
heatmap_data_top15 = heatmap_data.iloc[:15]
sns.heatmap(heatmap_data_top15, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': '占比 (%)'}, ax=ax)
ax.set_title('各地区消费结构占比热力图 (前15个地区)', fontsize=14, fontweight='bold')
ax.set_xlabel('消费结构')
ax.set_ylabel('地区')
plt.tight_layout()
plt.show()

print("\n分析完成！")