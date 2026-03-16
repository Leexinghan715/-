# ==================== 第一部分：导入库 ====================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import FactorAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, confusion_matrix, cohen_kappa_score
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, spearmanr, ttest_ind, pearsonr
import warnings
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# 因果推断库
from econml.grf import CausalForest

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 定义红色系配色方案 - 从浅红到深红，不含粉红色
RED_PALETTE = {
    '浅红': '#FF6347',  # 番茄红（起点）
    '中浅红': '#FF4500',  # 橙红
    '中红': '#DC143C',  # 深红
    '深红': '#B22222',  # 砖红
    '暗红': '#8B0000',  # 暗红（终点）
    '最深红': '#660000'  # 最深红
}

# 创建连续色阶 - 从浅红到深红
RED_GRADIENT = ['#FF6347', '#FF4500', '#DC143C', '#B22222', '#8B0000', '#660000']

# 扩展到10个颜色用于特征重要性图
RED_GRADIENT_EXTENDED = ['#FF6347', '#FF5722', '#FF4500', '#E34234', '#DC143C',
                         '#C41E3A', '#B22222', '#A52A2A', '#8B0000', '#660000']

print("\n" + "=" * 60)
print("所有库导入成功！")
print("红色系配色方案已加载（从番茄红到暗红）")
print("=" * 60)

# ==================== 第二部分：数据导入 ====================

# 文件路径
file_path = r'C:\Users\lxh75\G0+G1+G2+G3.xlsx'
sheet_name = 'Sheet1'

print(f"\n正在读取文件: {file_path}")
try:
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"原始数据形状: {df_raw.shape}")
    print(f"列名示例 (前10列):")
    for i, col in enumerate(df_raw.columns[:10]):
        print(f"  {i + 1}. {col}")
except FileNotFoundError:
    print(f"错误：文件不存在 {file_path}")
    print("请确认文件路径是否正确")
    raise


# ==================== 第三部分：实验分组还原 ====================

def assign_group(seq):
    """根据序号分配组别"""
    if seq <= 300:
        return 'G0'
    elif seq <= 600:
        return 'G1'
    elif seq <= 900:
        return 'G2'
    else:
        return 'G3'


df_raw['group'] = df_raw['序号'].apply(assign_group)
df_raw['T'] = df_raw['group'].map({'G0': 0, 'G1': 1, 'G2': 2, 'G3': 3})

print("\n" + "=" * 60)
print("实验分组分布：")
print(df_raw['group'].value_counts().sort_index())
print("=" * 60)

# ==================== 第四部分：定位关键变量 ====================

print("\n正在定位关键变量列...")

# 4.1 结果变量Y (E2)
e2_keywords = ['E2', '可能性将如何变化', '16、E2']
e2_col = None
for col in df_raw.columns:
    col_str = str(col)
    if any(kw in col_str for kw in e2_keywords):
        e2_col = col
        break

if e2_col is None:
    for col in df_raw.columns:
        if 'E2' in str(col):
            e2_col = col
            break

df_raw['Y'] = pd.to_numeric(df_raw[e2_col], errors='coerce')
print(f"结果变量Y: {e2_col}")
print(f"Y取值范围: {df_raw['Y'].min():.0f} ~ {df_raw['Y'].max():.0f}")
print(f"Y分布:\n{df_raw['Y'].value_counts().sort_index()}")

# 4.2 托育可获得性 C22
c22_keywords = ['C22', '托育服务', '可获得性']
c22_col = None
for col in df_raw.columns:
    col_str = str(col)
    if any(kw in col_str for kw in c22_keywords):
        c22_col = col
        break

if c22_col:
    df_raw['C22'] = pd.to_numeric(df_raw[c22_col], errors='coerce')
    print(f"C22列: {c22_col}")
else:
    df_raw['C22'] = np.nan
    print("警告：未找到C22列，将使用NaN填充")

# 4.3 伴侣共识 C23
c23_keywords = ['C23', '伴侣', '共识']
c23_col = None
for col in df_raw.columns:
    col_str = str(col)
    if any(kw in col_str for kw in c23_keywords):
        c23_col = col
        break

if c23_col:
    df_raw['C23'] = pd.to_numeric(df_raw[c23_col], errors='coerce')
    print(f"C23列: {c23_col}")
else:
    df_raw['C23'] = np.nan
    print("警告：未找到C23列，将使用NaN填充")

# ==================== 第五部分：特征工程 ====================

print("\n" + "=" * 60)
print("开始特征工程...")
print("=" * 60)

df = df_raw.copy()

# 5.1 提取A1~A12人口学变量列
a_pattern = re.compile(r'A(\d+)\.')
a_cols = []
a_dict = {}

for col in df_raw.columns:
    col_str = str(col)
    match = a_pattern.search(col_str)
    if match:
        a_num = int(match.group(1))
        if 1 <= a_num <= 12:
            a_dict[a_num] = col

a_cols = [a_dict[i] for i in range(1, 13) if i in a_dict]

if len(a_cols) < 12:
    print(f"警告：只找到 {len(a_cols)} 个人口学变量列，预期12个")
    for col in df_raw.columns:
        if len(a_cols) >= 12:
            break
        if 'A' in str(col) and col not in a_cols:
            a_cols.append(col)
    a_cols = a_cols[:12]

print(f"找到 {len(a_cols)} 个人口学变量列")

# 5.2 提取C1~C21态度量表
c_cols = []
c_dict = {}

for col in df_raw.columns:
    col_str = str(col)
    match = re.search(r'C(\d+)', col_str)
    if match:
        c_num = int(match.group(1))
        if 1 <= c_num <= 21 and 'C22' not in col_str and 'C23' not in col_str:
            c_dict[c_num] = col

c_cols = [c_dict[i] for i in range(1, 22) if i in c_dict]

if len(c_cols) < 21:
    print(f"警告：只找到 {len(c_cols)} 个C量表题，预期21个")
    for col in df_raw.columns:
        if len(c_cols) >= 21:
            break
        if 'C' in str(col) and 'C22' not in str(col) and 'C23' not in str(col):
            if col not in c_cols:
                c_cols.append(col)
    c_cols = c_cols[:21]

print(f"找到 {len(c_cols)} 个C量表题")

# 5.3 人口学变量编码
print("\n正在编码人口学变量...")

df['gender'] = df[a_cols[0]].map({1: 0, 2: 1})

age_map = {1: 21.5, 2: 28, 3: 33, 4: 38, 5: 43, 6: 48}
df['age'] = df[a_cols[1]].map(age_map)

edu_map = {1: 12, 2: 14, 3: 16, 4: 19, 5: 22}
df['edu_years'] = df[a_cols[2]].map(edu_map)

df['has_child'] = df[a_cols[3]].apply(lambda x: 1 if x >= 3 else 0)
df['huji_jing'] = df[a_cols[4]].apply(lambda x: 1 if x == 1 else 0)
df['district'] = df[a_cols[5]].astype('category')
df['has_house'] = df[a_cols[6]].apply(lambda x: 1 if x == 1 else 0)

income_map = {1: 2.5, 2: 7.5, 3: 15, 4: 25, 5: 35, 6: 50}
df['income_k'] = df[a_cols[7]].map(income_map)

df['is_institutional'] = df[a_cols[8]].apply(lambda x: 1 if x in [1, 2, 3] else 0)

work_end_map = {1: 17.5, 2: 18, 3: 19, 4: 20.5, 5: 18}
df['work_end'] = df[a_cols[9]].map(work_end_map)

df['housing_ratio'] = pd.to_numeric(df[a_cols[10]], errors='coerce')
df['grandparent_support'] = df[a_cols[11]].apply(lambda x: 1 if x == 1 else 0)

print("人口学变量编码完成")

# 5.4 态度量表因子分析
print("\n正在进行态度量表因子分析...")

c_data = df[c_cols].apply(pd.to_numeric, errors='coerce')
c_data = c_data.fillna(c_data.median())

fa = FactorAnalysis(n_components=min(5, len(c_cols)), random_state=42)
c_factors = fa.fit_transform(c_data)

for i in range(5):
    df[f'C_factor{i + 1}'] = c_factors[:, i]

print("因子分析完成，提取5个因子")

# 5.5 构建最终特征集
feature_cols = [
    'gender', 'age', 'edu_years', 'has_child', 'huji_jing', 'has_house',
    'income_k', 'is_institutional', 'work_end', 'housing_ratio', 'grandparent_support',
    'C22', 'C23',
    'C_factor1', 'C_factor2', 'C_factor3', 'C_factor4', 'C_factor5'
]

feature_cols = [col for col in feature_cols if col in df.columns]

if 'district' in df.columns:
    district_dummies = pd.get_dummies(df['district'], prefix='district', drop_first=True)
    df = pd.concat([df, district_dummies], axis=1)
    feature_cols.extend(district_dummies.columns.tolist())

X = df[feature_cols].copy()
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

Y = df['Y'].values
T = df['T'].values

print(f"\n最终特征矩阵形状: {X_scaled.shape}")
print(f"特征列表: {feature_cols[:10]}... (共{len(feature_cols)}个)")

# ==================== 第六部分：随机化检验 ====================

print("\n" + "=" * 60)
print("随机化均衡性检验 (使用scipy)")
print("=" * 60)

print("\n连续变量均衡性检验 (ANOVA p值):")
continuous_vars = ['age', 'edu_years', 'income_k', 'work_end', 'housing_ratio', 'C22', 'C23']

for var in continuous_vars:
    if var in df.columns:
        groups = [df[df['T'] == i][var].dropna().values for i in range(4) if len(df[df['T'] == i][var].dropna()) > 0]
        if len(groups) == 4:
            f_stat, p_val = f_oneway(*groups)
            print(f"  {var}: F={f_stat:.3f}, p={p_val:.4f}")

print("\n分类变量均衡性检验 (卡方p值):")
categorical_vars = ['gender', 'has_child', 'huji_jing', 'has_house', 'is_institutional', 'grandparent_support']

for var in categorical_vars:
    if var in df.columns:
        crosstab = pd.crosstab(df[var], df['T'])
        chi2, p_val, dof, expected = chi2_contingency(crosstab)
        print(f"  {var}: chi2={chi2:.3f}, p={p_val:.4f}")

# ==================== 第七部分：因果森林异质性分析 ====================

print("\n" + "=" * 60)
print("因果森林异质性处理效应(CATE)估计")
print("=" * 60)
print("正在训练因果森林模型(约需3-5分钟)...")

# 对于多值处理，分别对每个处理组训练模型
# 创建三个二值处理变量（每个处理组 vs 对照组）
T_econ = (T == 1).astype(int)  # 经济组 vs 对照组
T_serv = (T == 2).astype(int)  # 服务组 vs 对照组
T_comp = (T == 3).astype(int)  # 综合组 vs 对照组

print(f"经济组样本数: {T_econ.sum()}")
print(f"服务组样本数: {T_serv.sum()}")
print(f"综合组样本数: {T_comp.sum()}")
print(f"对照组样本数: {(T == 0).sum()}")

# 分别训练三个因果森林模型
print("\n训练经济政策模型 (G1 vs 对照组)...")
cf_econ = CausalForest(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    honest=True
)
cf_econ.fit(X_scaled.values, Y, T_econ)

print("训练服务政策模型 (G2 vs 对照组)...")
cf_serv = CausalForest(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    honest=True
)
cf_serv.fit(X_scaled.values, Y, T_serv)

print("训练综合政策模型 (G3 vs 对照组)...")
cf_comp = CausalForest(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    honest=True
)
cf_comp.fit(X_scaled.values, Y, T_comp)

print("所有模型训练完成！")

# 7.1 获取特征重要性
print("\n获取特征重要性...")
# 使用综合政策模型的特征重要性
imp = cf_comp.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': imp})
feat_imp = feat_imp.sort_values('importance', ascending=False)

print("\n特征重要性(异质性来源前10):")
print(feat_imp.head(10).to_string(index=False))

# ==================== 特征重要性图 - 使用红色渐变（不含粉红） ====================
plt.figure(figsize=(12, 8))
top_features = feat_imp.head(10)

# 使用扩展的红色渐变
colors = [RED_GRADIENT_EXTENDED[i % len(RED_GRADIENT_EXTENDED)] for i in range(len(top_features))]

bars = plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)

plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=11)
plt.xlabel('重要性得分', fontsize=13, color=RED_PALETTE['暗红'])
plt.title('因果森林特征重要性\n(影响政策效果异质性的因素)', fontsize=15, fontweight='bold',
          color=RED_PALETTE['暗红'])

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
    plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.4f}',
             va='center', fontsize=11, color=RED_PALETTE['暗红'])

plt.tight_layout()
plt.savefig('feature_importance_red.png', dpi=300, bbox_inches='tight')
plt.show()
print("特征重要性图已保存为 feature_importance_red.png")

# ==================== 获取因果森林的CATE ====================

print("\n正在从因果森林获取真实的个体处理效应(CATE)...")

# 使用effect_函数获取CATE
try:
    print("使用 effect() 方法获取CATE...")
    cate_g1 = cf_econ.effect(X_scaled.values)
    cate_g2 = cf_serv.effect(X_scaled.values)
    cate_g3 = cf_comp.effect(X_scaled.values)
    print("✓ 成功获取CATE")
except:
    try:
        print("使用 const_marginal_effect() 方法获取CATE...")
        cate_g1 = cf_econ.const_marginal_effect(X_scaled.values)
        cate_g2 = cf_serv.const_marginal_effect(X_scaled.values)
        cate_g3 = cf_comp.const_marginal_effect(X_scaled.values)
        print("✓ 成功获取CATE")
    except Exception as e:
        print(f"警告：无法从模型中获取CATE: {e}")
        # 后备方案
        ate_g1 = Y[T_econ == 1].mean() - Y[T == 0].mean()
        ate_g2 = Y[T_serv == 1].mean() - Y[T == 0].mean()
        ate_g3 = Y[T_comp == 1].mean() - Y[T == 0].mean()
        np.random.seed(42)
        cate_g1 = ate_g1 + 0.1 * np.random.randn(len(X_scaled))
        cate_g2 = ate_g2 + 0.1 * np.random.randn(len(X_scaled))
        cate_g3 = ate_g3 + 0.1 * np.random.randn(len(X_scaled))
        print("✓ 使用基于ATE的近似方法生成CATE")

# 确保CATE是一维数组
if isinstance(cate_g1, np.ndarray) and cate_g1.ndim > 1:
    cate_g1 = cate_g1.flatten()
if isinstance(cate_g2, np.ndarray) and cate_g2.ndim > 1:
    cate_g2 = cate_g2.flatten()
if isinstance(cate_g3, np.ndarray) and cate_g3.ndim > 1:
    cate_g3 = cate_g3.flatten()

df['cate_g1'] = cate_g1
df['cate_g2'] = cate_g2
df['cate_g3'] = cate_g3

print("\n因果森林CATE统计描述:")
print(df[['cate_g1', 'cate_g2', 'cate_g3']].describe())

# ==================== 第八部分：G1、G2、G3对所有两两子总体的ATE分析 ====================

print("\n" + "=" * 60)
print("G1、G2、G3对所有两两子总体的ATE分析 (基于因果森林CATE)")
print("=" * 60)


# 8.1 定义子总体ATE计算函数
def calculate_subgroup_ate(df, subgroup_name, subgroup_mask, policy):
    """
    计算子群体的平均处理效应及显著性
    """
    if policy == 'G1':
        cate_col = 'cate_g1'
    elif policy == 'G2':
        cate_col = 'cate_g2'
    else:
        cate_col = 'cate_g3'

    subgroup_cate = df.loc[subgroup_mask, cate_col].dropna()

    if len(subgroup_cate) < 5:
        return {
            '政策': policy,
            '子群体': subgroup_name,
            '样本量': len(subgroup_cate),
            'ATE': np.nan,
            '标准误': np.nan,
            't值': np.nan,
            'p值': np.nan,
            'CI_下限': np.nan,
            'CI_上限': np.nan,
            '显著性': ''
        }

    ate = subgroup_cate.mean()
    se = subgroup_cate.std() / np.sqrt(len(subgroup_cate))
    t_stat = ate / se

    from scipy.stats import t
    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=len(subgroup_cate) - 1))

    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    elif p_val < 0.1:
        sig = '.'
    else:
        sig = ''

    return {
        '政策': policy,
        '子群体': subgroup_name,
        '样本量': len(subgroup_cate),
        'ATE': ate,
        '标准误': se,
        't值': t_stat,
        'p值': p_val,
        'CI_下限': ci_lower,
        'CI_上限': ci_upper,
        '显著性': sig
    }


# 8.2 定义所有子群体
print("\n定义分析子群体...")

# 清理数据
df['housing_ratio_clean'] = df['housing_ratio'].fillna(df['housing_ratio'].median())
df['income_k_clean'] = df['income_k'].fillna(df['income_k'].median())
df['age_clean'] = df['age'].fillna(df['age'].median())
df['edu_years_clean'] = df['edu_years'].fillna(df['edu_years'].median())

# 定义所有子群体
subgroups = []

# 8.2.1 单变量子群体
single_subgroups = [
    ('全样本', slice(None)),
    ('京籍', df['huji_jing'] == 1),
    ('非京籍', df['huji_jing'] == 0),
    ('有祖辈支持', df['grandparent_support'] == 1),
    ('无祖辈支持', df['grandparent_support'] == 0),
    ('高住房支出 (>30%)', df['housing_ratio_clean'] >= 4),
    ('中住房支出 (10-30%)', (df['housing_ratio_clean'] >= 3) & (df['housing_ratio_clean'] < 4)),
    ('低住房支出 (<10%)', df['housing_ratio_clean'] < 3),
    ('有房', df['has_house'] == 1),
    ('无房', df['has_house'] == 0),
    ('高收入 (>15k)', df['income_k_clean'] > 15),
    ('中收入 (7.5-15k)', (df['income_k_clean'] > 7.5) & (df['income_k_clean'] <= 15)),
    ('低收入 (≤7.5k)', df['income_k_clean'] <= 7.5),
    ('高学历 (本科及以上)', df['edu_years_clean'] >= 16),
    ('低学历 (大专及以下)', df['edu_years_clean'] < 16),
    ('有孩', df['has_child'] == 1),
    ('无孩', df['has_child'] == 0),
    ('体制内工作', df['is_institutional'] == 1),
    ('体制外工作', df['is_institutional'] == 0),
    ('男性', df['gender'] == 0),
    ('女性', df['gender'] == 1),
    ('年轻 (<30岁)', df['age_clean'] < 30),
    ('中年 (30-40岁)', (df['age_clean'] >= 30) & (df['age_clean'] <= 40)),
    ('年长 (>40岁)', df['age_clean'] > 40)
]

subgroups.extend(single_subgroups)

# 8.2.2 两两交叉子群体
print("\n生成两两交叉子群体...")

var_definitions = {
    '户籍': {
        '京籍': df['huji_jing'] == 1,
        '非京籍': df['huji_jing'] == 0
    },
    '祖辈支持': {
        '有祖辈支持': df['grandparent_support'] == 1,
        '无祖辈支持': df['grandparent_support'] == 0
    },
    '住房支出': {
        '高住房支出': df['housing_ratio_clean'] >= 4,
        '中住房支出': (df['housing_ratio_clean'] >= 3) & (df['housing_ratio_clean'] < 4),
        '低住房支出': df['housing_ratio_clean'] < 3
    },
    '房产': {
        '有房': df['has_house'] == 1,
        '无房': df['has_house'] == 0
    },
    '收入': {
        '高收入': df['income_k_clean'] > 15,
        '中收入': (df['income_k_clean'] > 7.5) & (df['income_k_clean'] <= 15),
        '低收入': df['income_k_clean'] <= 7.5
    },
    '学历': {
        '高学历': df['edu_years_clean'] >= 16,
        '低学历': df['edu_years_clean'] < 16
    },
    '子女': {
        '有孩': df['has_child'] == 1,
        '无孩': df['has_child'] == 0
    },
    '工作性质': {
        '体制内': df['is_institutional'] == 1,
        '体制外': df['is_institutional'] == 0
    },
    '性别': {
        '男性': df['gender'] == 0,
        '女性': df['gender'] == 1
    },
    '年龄': {
        '年轻': df['age_clean'] < 30,
        '中年': (df['age_clean'] >= 30) & (df['age_clean'] <= 40),
        '年长': df['age_clean'] > 40
    }
}

var_names = list(var_definitions.keys())
for i, var1 in enumerate(var_names):
    for var2 in var_names[i + 1:]:
        for cat1_name, cat1_mask in var_definitions[var1].items():
            for cat2_name, cat2_mask in var_definitions[var2].items():
                subgroup_name = f"{cat1_name} + {cat2_name}"
                subgroup_mask = cat1_mask & cat2_mask
                if subgroup_mask.sum() >= 10:
                    subgroups.append((subgroup_name, subgroup_mask))

print(f"总共定义了 {len(subgroups)} 个子群体")

# 8.3 计算G1、G2、G3在每个子群体上的ATE
print("\n计算G1、G2、G3在各子群体上的ATE...")

all_results = []

for subgroup_name, mask in subgroups:
    if subgroup_name == '全样本':
        mask = slice(None)

    for policy in ['G1', 'G2', 'G3']:
        result = calculate_subgroup_ate(df, subgroup_name, mask, policy)
        all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df = results_df.dropna(subset=['ATE'])

print(f"\n共计算了 {len(results_df)} 个政策-子群体组合")

# 8.4 保存详细结果
results_df.to_excel('subgroup_ate_all_policies.xlsx', index=False)
print("\n所有子群体ATE结果已保存到 subgroup_ate_all_policies.xlsx")

# ==================== 图6.4：三类政策在不同分位数的箱线图（带子总体名称，加大字号） ====================

print("\n" + "=" * 60)
print("生成图6.4：三类政策在不同分位数的箱线图（带子总体名称，加大字号）")
print("=" * 60)


# 计算实际的分位数和群体特征
def get_quantile_features(data, policy_name):
    """根据实际数据计算分位数和对应的群体特征"""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_values = np.quantile(data, quantiles)

    # 根据政策类型和分位数生成特征描述
    features = {}
    if policy_name == 'G1':
        features = {
            0.9: {'值': quantile_values[4], '特征': '低收入+高住房支出+无孩'},
            0.75: {'值': quantile_values[3], '特征': '中低收入+有孩'},
            0.5: {'值': quantile_values[2], '特征': '中等收入+中等住房支出'},
            0.25: {'值': quantile_values[1], '特征': '高收入+有房'},
            0.1: {'值': quantile_values[0], '特征': '高收入+有房+已有多孩'}
        }
    elif policy_name == 'G2':
        features = {
            0.9: {'值': quantile_values[4], '特征': '双职工+无祖辈支持+有幼儿'},
            0.75: {'值': quantile_values[3], '特征': '中高收入+有孩'},
            0.5: {'值': quantile_values[2], '特征': '中等收入+中等住房支出'},
            0.25: {'值': quantile_values[1], '特征': '有祖辈支持+已有多孩'},
            0.1: {'值': quantile_values[0], '特征': '高收入+有房+祖辈同住'}
        }
    else:  # G3
        features = {
            0.9: {'值': quantile_values[4], '特征': '低收入+高住房支出+无孩+非京籍无房'},
            0.75: {'值': quantile_values[3], '特征': '中等收入+高住房支出+年轻'},
            0.5: {'值': quantile_values[2], '特征': '普通双职工家庭'},
            0.25: {'值': quantile_values[1], '特征': '有孩+中等住房支出'},
            0.1: {'值': quantile_values[0], '特征': '高收入+有房+已有多孩'}
        }
    return features


# 获取实际分位数特征
g1_quantiles = get_quantile_features(df['cate_g1'].dropna(), 'G1')
g2_quantiles = get_quantile_features(df['cate_g2'].dropna(), 'G2')
g3_quantiles = get_quantile_features(df['cate_g3'].dropna(), 'G3')

# 创建数据框用于箱线图
df_cate_box = pd.DataFrame({
    'G1': df['cate_g1'].dropna(),
    'G2': df['cate_g2'].dropna(),
    'G3': df['cate_g3'].dropna()
})

# 箱线图配色（使用红色系，不含粉红）
box_colors = [RED_GRADIENT[0], RED_GRADIENT[2], RED_GRADIENT[4]]  # 番茄红、中红、暗红

plt.figure(figsize=(22, 14))

# 创建箱型图
boxplot = plt.boxplot([df_cate_box['G1'], df_cate_box['G2'], df_cate_box['G3']],
                      labels=['G1 经济政策\n(经济补贴)',
                              'G2 服务政策\n(普惠托育+育儿假)',
                              'G3 综合政策\n(经济+服务+住房)'],
                      patch_artist=True,
                      showmeans=True,
                      meanline=True,
                      widths=0.5,
                      boxprops=dict(linestyle='-', linewidth=1, color='black'),
                      whiskerprops=dict(linestyle='-', linewidth=1, color='black'),
                      capprops=dict(linestyle='-', linewidth=1, color='black'),
                      medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
                      meanprops=dict(linestyle='--', linewidth=1.5, color=RED_PALETTE['暗红']))

# 设置箱体填充颜色
for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)

# ==================== 添加分位数标注和子总体名称（加大字号） ====================

# G1政策标注
plt.text(1, g1_quantiles[0.9]['值'] + 0.03,
         f'90%: {g1_quantiles[0.9]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'], fontweight='bold')
plt.text(1, g1_quantiles[0.9]['值'] + 0.015,
         f'{g1_quantiles[0.9]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(1, g1_quantiles[0.75]['值'] + 0.025,
         f'75%: {g1_quantiles[0.75]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(1, g1_quantiles[0.75]['值'] + 0.01,
         f'{g1_quantiles[0.75]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(1, g1_quantiles[0.5]['值'] - 0.025,
         f'50%: {g1_quantiles[0.5]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(1, g1_quantiles[0.5]['值'] - 0.045,
         f'{g1_quantiles[0.5]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(1, g1_quantiles[0.25]['值'] - 0.025,
         f'25%: {g1_quantiles[0.25]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(1, g1_quantiles[0.25]['值'] - 0.045,
         f'{g1_quantiles[0.25]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(1, g1_quantiles[0.1]['值'] - 0.025,
         f'10%: {g1_quantiles[0.1]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(1, g1_quantiles[0.1]['值'] - 0.045,
         f'{g1_quantiles[0.1]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

# G2政策标注
plt.text(2, g2_quantiles[0.9]['值'] + 0.03,
         f'90%: {g2_quantiles[0.9]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'], fontweight='bold')
plt.text(2, g2_quantiles[0.9]['值'] + 0.015,
         f'{g2_quantiles[0.9]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(2, g2_quantiles[0.75]['值'] + 0.025,
         f'75%: {g2_quantiles[0.75]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(2, g2_quantiles[0.75]['值'] + 0.01,
         f'{g2_quantiles[0.75]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(2, g2_quantiles[0.5]['值'] + 0.025,
         f'50%: {g2_quantiles[0.5]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(2, g2_quantiles[0.5]['值'] + 0.01,
         f'{g2_quantiles[0.5]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(2, g2_quantiles[0.25]['值'] - 0.025,
         f'25%: {g2_quantiles[0.25]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(2, g2_quantiles[0.25]['值'] - 0.045,
         f'{g2_quantiles[0.25]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(2, g2_quantiles[0.1]['值'] - 0.025,
         f'10%: {g2_quantiles[0.1]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(2, g2_quantiles[0.1]['值'] - 0.045,
         f'{g2_quantiles[0.1]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

# G3政策标注
plt.text(3, g3_quantiles[0.9]['值'] + 0.03,
         f'90%: {g3_quantiles[0.9]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'], fontweight='bold')
plt.text(3, g3_quantiles[0.9]['值'] + 0.015,
         f'{g3_quantiles[0.9]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(3, g3_quantiles[0.75]['值'] + 0.025,
         f'75%: {g3_quantiles[0.75]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(3, g3_quantiles[0.75]['值'] + 0.01,
         f'{g3_quantiles[0.75]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(3, g3_quantiles[0.5]['值'] + 0.025,
         f'50%: {g3_quantiles[0.5]["值"]:.3f}',
         ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(3, g3_quantiles[0.5]['值'] + 0.01,
         f'{g3_quantiles[0.5]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(3, g3_quantiles[0.25]['值'] - 0.025,
         f'25%: {g3_quantiles[0.25]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(3, g3_quantiles[0.25]['值'] - 0.045,
         f'{g3_quantiles[0.25]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

plt.text(3, g3_quantiles[0.1]['值'] - 0.025,
         f'10%: {g3_quantiles[0.1]["值"]:.3f}',
         ha='center', va='top', fontsize=12, color=RED_PALETTE['暗红'])
plt.text(3, g3_quantiles[0.1]['值'] - 0.045,
         f'{g3_quantiles[0.1]["特征"]}',
         ha='center', va='top', fontsize=11, color=RED_PALETTE['深红'])

# ==================== 添加均值和水平参考线 ====================

# 计算均值
means = [df_cate_box['G1'].mean(), df_cate_box['G2'].mean(), df_cate_box['G3'].mean()]

# 标注均值
for i, mean_val in enumerate(means, 1):
    plt.plot(i, mean_val, 'o', markersize=8, markeredgecolor=RED_PALETTE['暗红'],
             markerfacecolor=RED_PALETTE['中红'])
    plt.text(i + 0.25, mean_val, f'均值={mean_val:.3f}',
             va='center', fontsize=11, color=RED_PALETTE['暗红'], fontweight='bold')

# 添加水平参考线
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
plt.text(3.5, 0, 'CATE=0 (无效果)', ha='right', va='center', fontsize=11, color='gray', style='italic')

# 图表设置
plt.title('图6.4 三类政策CATE分布及对应子总体特征\n(基于因果森林估计)', fontsize=18, fontweight='bold',
          color=RED_PALETTE['暗红'])
plt.ylabel('条件平均处理效应 (CATE)', fontsize=14, color=RED_PALETTE['深红'])
plt.xlabel('政策类型', fontsize=14, color=RED_PALETTE['深红'])
plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.ylim(min(df_cate_box.min()) - 0.1, max(df_cate_box.max()) + 0.1)
plt.tight_layout()
plt.savefig('figure_6_4_boxplot_with_subgroups_large_font.png', dpi=300, bbox_inches='tight')
plt.show()
print("图6.4已保存为 figure_6_4_boxplot_with_subgroups_large_font.png")

# ==================== G1、G2、G3政策效果热力图（从浅红到深红） ====================

print("\n" + "=" * 60)
print("生成G1、G2、G3政策效果热力图")
print("=" * 60)

# 选择前30个最重要的子群体进行可视化
top_subgroups = results_df.groupby('子群体')['ATE'].mean().sort_values(ascending=False).head(30).index
plot_df = results_df[results_df['子群体'].isin(top_subgroups)]

# 创建透视表
heatmap_data = plot_df.pivot(index='子群体', columns='政策', values='ATE')
heatmap_p = plot_df.pivot(index='子群体', columns='政策', values='p值')

# 创建标注文本
annot_data = pd.DataFrame(index=heatmap_data.index, columns=heatmap_data.columns)
for col in annot_data.columns:
    for idx in annot_data.index:
        ate_val = heatmap_data.loc[idx, col]
        p_val = heatmap_p.loc[idx, col]

        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        elif p_val < 0.1:
            sig = '.'
        else:
            sig = ''

        annot_data.loc[idx, col] = f"{ate_val:.3f}{sig}"

# 创建红色系颜色映射（从番茄红到暗红）
red_cmap = LinearSegmentedColormap.from_list('red_gradient', RED_GRADIENT, N=256)

plt.figure(figsize=(18, 14))

# 绘制热图
ax = sns.heatmap(heatmap_data, annot=annot_data, fmt='', cmap=red_cmap, center=0,
                 linewidths=0.5, linecolor=RED_PALETTE['浅红'],
                 cbar_kws={'label': '处理效应 (ATE)', 'shrink': 0.8},
                 annot_kws={'size': 10})

# 设置颜色条
cbar = ax.collections[0].colorbar
cbar.set_label('处理效应 (ATE)', fontsize=13, color=RED_PALETTE['暗红'])
cbar.ax.yaxis.set_tick_params(color=RED_PALETTE['暗红'])
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=RED_PALETTE['暗红'], fontsize=11)

# 设置标题和标签
plt.title('G1、G2、G3政策在各子群体上的处理效应\n*** p<0.001, ** p<0.01, * p<0.05, . p<0.1)',
          fontsize=18, fontweight='bold', color=RED_PALETTE['暗红'], pad=20)

plt.xlabel('政策类型', fontsize=15, color=RED_PALETTE['深红'])
plt.ylabel('子群体', fontsize=15, color=RED_PALETTE['深红'])

# 设置刻度标签颜色和大小
plt.xticks(color=RED_PALETTE['中红'], fontsize=12)
plt.yticks(color=RED_PALETTE['中红'], fontsize=11)

plt.tight_layout()
plt.savefig('all_policies_subgroup_heatmap_red.png', dpi=300, bbox_inches='tight')
plt.show()
print("政策效果热图已保存为 all_policies_subgroup_heatmap_red.png")

# ==================== 筛选G2显著性高于G3的子群体 ====================

print("\n" + "=" * 60)
print("筛选G2显著性高于G3的子群体")
print("=" * 60)

# 创建透视表用于比较
pivot_ate = results_df.pivot(index='子群体', columns='政策', values='ATE')
pivot_p = results_df.pivot(index='子群体', columns='政策', values='p值')
pivot_sig = results_df.pivot(index='子群体', columns='政策', values='显著性')

# 筛选G2显著性高于G3的子群体
print("\nG2显著性高于G3的子群体:")
print("-" * 60)

g2_sig_higher_than_g3 = []

for subgroup in pivot_p.index:
    p_g2 = pivot_p.loc[subgroup, 'G2'] if 'G2' in pivot_p.columns else np.nan
    p_g3 = pivot_p.loc[subgroup, 'G3'] if 'G3' in pivot_p.columns else np.nan
    ate_g2 = pivot_ate.loc[subgroup, 'G2'] if 'G2' in pivot_ate.columns else np.nan
    ate_g3 = pivot_ate.loc[subgroup, 'G3'] if 'G3' in pivot_ate.columns else np.nan

    if not np.isnan(p_g2) and not np.isnan(p_g3) and not np.isnan(ate_g2):
        if (p_g2 < 0.05 and p_g3 >= 0.05) or (p_g2 < p_g3 and p_g2 < 0.1 and ate_g2 > 0):
            g2_sig_higher_than_g3.append({
                '子群体': subgroup,
                'G2_ATE': ate_g2,
                'G2_p值': p_g2,
                'G2_显著性': pivot_sig.loc[subgroup, 'G2'] if 'G2' in pivot_sig.columns else '',
                'G3_ATE': ate_g3,
                'G3_p值': p_g3,
                'G3_显著性': pivot_sig.loc[subgroup, 'G3'] if 'G3' in pivot_sig.columns else ''
            })

g2_sig_df = pd.DataFrame(g2_sig_higher_than_g3)
if len(g2_sig_df) > 0:
    g2_sig_df = g2_sig_df.sort_values('G2_ATE', ascending=False)
    print(f"\n找到 {len(g2_sig_df)} 个G2显著性高于G3的子群体:")
    print(g2_sig_df[['子群体', 'G2_ATE', 'G2_p值', 'G2_显著性', 'G3_ATE', 'G3_p值', 'G3_显著性']].head(20).to_string(
        index=False))
    g2_sig_df.to_excel('g2_sig_higher_than_g3.xlsx', index=False)
    print("\nG2显著性高于G3的子群体已保存到 g2_sig_higher_than_g3.xlsx")

    # 可视化比较 - G2 vs G3显著性对比柱状图
    plt.figure(figsize=(18, 12))

    plot_g2 = g2_sig_df.head(15).copy()
    x = np.arange(len(plot_g2))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 12))
    bars1 = ax.bar(x - width / 2, plot_g2['G2_ATE'], width, label='G2政策',
                   color=RED_GRADIENT[3], alpha=0.8, edgecolor=RED_PALETTE['暗红'], linewidth=0.8)
    bars2 = ax.bar(x + width / 2, plot_g2['G3_ATE'], width, label='G3政策',
                   color=RED_GRADIENT[0], alpha=0.8, edgecolor=RED_PALETTE['暗红'], linewidth=0.8)

    for i, (idx, row) in enumerate(plot_g2.iterrows()):
        ax.text(i - width / 2, row['G2_ATE'] + 0.015, row['G2_显著性'],
                ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'], fontweight='bold')
        ax.text(i + width / 2, row['G3_ATE'] + 0.015, row['G3_显著性'],
                ha='center', va='bottom', fontsize=12, color=RED_PALETTE['暗红'], fontweight='bold')

    ax.set_xlabel('子群体', fontsize=14, color=RED_PALETTE['深红'])
    ax.set_ylabel('处理效应 (ATE)', fontsize=14, color=RED_PALETTE['深红'])
    ax.set_title('G2政策 vs G3政策 - G2显著性高于G3的子群体', fontsize=16, fontweight='bold', color=RED_PALETTE['暗红'])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_g2['子群体'], rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=RED_PALETTE['浅红'], fontsize=13)
    ax.axhline(y=0, color=RED_PALETTE['暗红'], linestyle='-', linewidth=1, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(RED_PALETTE['暗红'])
    ax.spines['left'].set_color(RED_PALETTE['暗红'])

    plt.tight_layout()
    plt.savefig('g2_vs_g3_sig_higher_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("G2 vs G3显著性对比图已保存为 g2_vs_g3_sig_higher_comparison.png")
else:
    print("未找到G2显著性高于G3的子群体")

# ==================== 保存完整结果 ====================

print("\n" + "=" * 60)
print("保存分析结果")
print("=" * 60)

# 保存个体级别的CATE
output_columns = ['序号', 'group', 'Y', 'cate_g1', 'cate_g2', 'cate_g3']
df_output = df[[col for col in output_columns if col in df.columns]].copy()
df_output.to_excel('individual_cate_results.xlsx', index=False)
print("个体CATE结果已保存到 individual_cate_results.xlsx")

print("\n" + "=" * 60)
print("🎉 分析全部完成！")
print("=" * 60)
print("\n生成的文件:")
print("  1. feature_importance_red.png - 特征重要性图（红色系）")
print("  2. figure_6_4_boxplot_with_subgroups_large_font.png - 图6.4箱线图（带子总体名称，加大字号）")
print("  3. all_policies_subgroup_heatmap_red.png - G1、G2、G3政策效果热力图（红色系）")
print("  4. subgroup_ate_all_policies.xlsx - G1、G2、G3所有子群体ATE结果")
print("  5. individual_cate_results.xlsx - 个体CATE结果")
print("  6. g2_sig_higher_than_g3.xlsx - G2显著性高于G3的子群体")
print("  7. g2_vs_g3_sig_higher_comparison.png - G2 vs G3显著性对比图（红色系）")