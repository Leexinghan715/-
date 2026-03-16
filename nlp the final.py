# 1. 导入必要的库
import pandas as pd
import numpy as np
import os
import re
import jieba
import jieba.analyse
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 文本处理相关
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# 可视化
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec


# 2. 设置中文字体和解决中文显示问题
def setup_chinese_font():
    """配置中文字体"""
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc'
    ]

    available_fonts = []
    for font_path in font_paths:
        if os.path.exists(font_path):
            available_fonts.append(font_path)

    if available_fonts:
        for font_path in available_fonts:
            matplotlib.font_manager.fontManager.addfont(font_path)

        font_names = []
        for font_path in available_fonts:
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            font_names.append(font_prop.get_name())

        if font_names:
            plt.rcParams['font.sans-serif'] = font_names + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.size'] = 12
    print("✓ Chinese fonts configured")


# 3. 文件路径配置和输出目录创建
class FileConfig:
    """文件路径配置类"""

    def __init__(self):
        base_path = r"C:\Users\lxh75"
        self.input_excel_path = os.path.join(base_path, "Natural Language.xlsx")

        if not os.path.exists(self.input_excel_path):
            alt_path = os.path.join(base_path, "Natural Language.xls")
            if os.path.exists(alt_path):
                self.input_excel_path = alt_path
                print(f"Using file: {self.input_excel_path}")
            else:
                print(f"Warning: File not found - {self.input_excel_path}")

        self.output_dir = os.path.join(base_path, "NLP_Analysis_Red_Theme")
        self.output_excel = os.path.join(self.output_dir, "Topic_Analysis_Results.xlsx")
        self.charts_dir = os.path.join(self.output_dir, "Red_Theme_Charts")
        self.custom_dict_path = os.path.join(self.output_dir, "custom_dict.txt")

        self.create_output_dirs()

    def create_output_dirs(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        print(f"✓ Output directory created: {self.output_dir}")


# 4. 数据加载器
class DataLoader:
    """数据加载和处理类"""

    def __init__(self, file_config):
        self.config = file_config

    def load_excel_data(self):
        """从Excel文件加载数据"""
        print(f"Loading file: {self.config.input_excel_path}")

        if not os.path.exists(self.config.input_excel_path):
            print(f"Error: File does not exist - {self.config.input_excel_path}")
            return None

        try:
            excel_file = pd.ExcelFile(self.config.input_excel_path)
            sheet_names = excel_file.sheet_names

            common_sheet_names = ['Sheet1', '数据', 'Data', 'Sheet', '原始数据', 'responses', '回答', '建议', '文本']

            for sheet_name in sheet_names:
                if any(name.lower() in sheet_name.lower() for name in common_sheet_names):
                    df = pd.read_excel(self.config.input_excel_path, sheet_name=sheet_name)
                    print(f"✓ Loaded data from sheet '{sheet_name}'")
                    break
            else:
                df = pd.read_excel(self.config.input_excel_path, sheet_name=sheet_names[0])
                print(f"✓ Loaded data from first sheet '{sheet_names[0]}'")

            print(f"Data shape: {df.shape}")

            text_columns = []
            for col in df.columns:
                if isinstance(col, str):
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in
                           ['文本', '内容', '回答', '建议', '意见', '评论', '反馈', '描述', '文字', '表述']):
                        text_columns.append(col)
                else:
                    try:
                        col_str = str(col)
                        col_lower = col_str.lower()
                        if any(keyword in col_lower for keyword in
                               ['文本', '内容', '回答', '建议', '意见', '评论', '反馈', '描述', '文字', '表述']):
                            text_columns.append(col)
                    except:
                        continue

            if text_columns:
                print(f"Found text columns: {text_columns}")
                text_column = text_columns[0]
            else:
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].astype(str).str.len().mean() > 10:
                        text_columns.append(col)

                if text_columns:
                    text_column = text_columns[0]
                    print(f"Auto-identified text column: {text_column}")
                else:
                    text_column = df.columns[0]
                    print(f"No clear text column found, using first column: {text_column}")

            processed_df = pd.DataFrame({
                'ID': range(1, len(df) + 1),
                'Original_Text': df[text_column].fillna('').astype(str).tolist()
            })

            for col in df.columns:
                if col != text_column:
                    processed_df[col] = df[col].values

            print(f"✓ Successfully loaded {len(processed_df)} records")
            return processed_df

        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None


# 5. 数据预处理类
class ImprovedTextPreprocessor:
    """改进的文本预处理类"""

    def __init__(self, config):
        self.config = config
        self.stopwords = self._load_stopwords()
        self._setup_custom_dict()

    def _load_stopwords(self):
        """加载停用词表"""
        basic_stopwords = {
            '的', '了', '和', '是', '在', '都', '就', '也', '有', '要',
            '能', '可以', '很', '好', '多', '大', '太', '不', '没有',
            '一个', '一些', '这种', '那种', '这样', '那样', '为了',
            '因为', '所以', '但是', '而且', '如果', '那么', '如何',
            '我', '我们', '你', '你们', '他', '她', '他们', '她们',
            '它', '这', '那', '这些', '那些', '什么', '怎么', '为什么',
            '吗', '呢', '啊', '呀', '吧', '啦', '哇', '哦', '噢',
            '对', '对于', '关于', '将', '并', '或', '与', '及'
        }

        return basic_stopwords

    def _setup_custom_dict(self):
        """创建和加载自定义词典"""
        custom_dict_content = """生育鼓励政策 100 n
生育支持政策 100 n
托育服务政策 100 n
教育公平政策 100 n
教育质量政策 100 n
职场平等政策 100 n
经济补贴政策 100 n
住房保障政策 100 n
医疗保障政策 100 n
社会支持体系 100 n
家庭支持体系 100 n
资源分配机制 95 n
女性就业权益 95 n
产后复工支持 95 n
育儿成本分担 95 n
学区制度优化 95 n
税收优惠政策 95 n
弹性工作制度 90 n
社区配套服务 90 n
公租房保障 90 n
课外辅导规范 90 n
教育焦虑缓解 90 n
隐形歧视消除 90 n
产假延长落实 90 n
育儿假期保障 90 n
家庭友好环境 85 n
工作平衡支持 85 n
普惠托育体系 85 n
公立机构建设 85 n
多子女家庭支持 85 n
生育补贴发放 85 n
职业发展保障 85 n
性别平等促进 85 n
资源分配机制 85 n
政策落实监督 85 n
长期稳定机制 85 n
体系完善建设 85 n
就业歧视消除 85 n
住房补贴政策 85 n
医疗费用保障 85 n
时间分配优化 85 n
精力消耗减轻 85 n
个人付出认可 85 n
"""

        try:
            with open(self.config.custom_dict_path, 'w', encoding='utf-8') as f:
                f.write(custom_dict_content)
            print(f"✓ Custom dictionary created: {self.config.custom_dict_path}")

            jieba.load_userdict(self.config.custom_dict_path)
            print("✓ Custom dictionary loaded successfully")
        except Exception as e:
            print(f"Error creating custom dictionary: {e}")
            print("Will use default dictionary")

    def clean_text(self, text):
        """文本清洗"""
        if not isinstance(text, str):
            return ""

        text = re.sub(r'[^\w\u4e00-\u9fa5，。！？；：、\s\.\,\!\"\'\?]', '', text)

        replacements = {
            '幼稚园': '幼儿园',
            '托儿': '托育',
            '幼稚': '幼儿',
            '託育': '托育',
            '房价': '住房',
            '住房价格': '住房',
            '工作歧视': '职场歧视',
            '性别歧视': '职场歧视',
            '课外班': '课外辅导',
            '辅导班': '课外辅导',
            '补习班': '课外辅导',
            '个税': '税收',
            '看病': '医疗',
            '医院': '医疗',
            '产假': '育儿假',
            '育儿': '托育',
            '减负': '减轻负担',
            '天花板': '职业障碍',
            '隐形': '隐性',
            '数字': '数据',
            '生育率': '生育数据',
            '996': '长时间工作',
            '压力': '负担',
            '成本': '费用',
            '需要': '需求',
            '希望': '期待',
            '提供': '给予',
            '确保': '保障',
            '解决': '处理',
            '关注': '重视',
            '非京籍': '非本地户籍',
            '京籍': '本地户籍',
            '权利': '权益'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

    def segment_text(self, text, use_stopwords=True):
        """中文分词"""
        if not text or text.strip() == '':
            return []

        words = jieba.lcut(text)

        if use_stopwords:
            words = [word for word in words if word not in self.stopwords and len(word) > 1]

        return words

    def preprocess_pipeline(self, df, text_column='Original_Text'):
        """完整的预处理流水线"""
        print("\nStarting data preprocessing...")
        print("-" * 50)

        df['Cleaned_Text'] = df[text_column].apply(self.clean_text)

        empty_texts = df['Cleaned_Text'].apply(lambda x: len(x.strip()) == 0)
        empty_count = empty_texts.sum()
        if empty_count > 0:
            print(f"Warning: Found {empty_count} empty text records")

        df['Segmented_Result'] = df['Cleaned_Text'].apply(lambda x: self.segment_text(x))
        df['Segmented_Text'] = df['Segmented_Result'].apply(lambda x: ' '.join(x))

        all_words = []
        for words in df['Segmented_Result']:
            all_words.extend(words)

        word_freq = Counter(all_words)
        print(f"✓ Total words: {len(all_words):,}")
        print(f"✓ Unique words: {len(word_freq):,}")

        print("\nTop 20 Frequent Words:")
        print("-" * 40)
        top_words = word_freq.most_common(20)
        for word, freq in top_words:
            print(f"{word:>8}: {freq:>4}")

        return df, word_freq


# 6. 优化的主题模型评估类
class OptimizedTopicModelEvaluator:
    """优化的主题模型评估类"""

    def __init__(self, min_topics=3, max_topics=10):
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.vectorizer = None
        self.full_dtm = None
        self.feature_names = None
        self.best_lda = None
        self.best_n_topics = None
        self.evaluation_results = {}
        self.topic_distances = {}  # 存储主题间距矩阵

    def prepare_data(self, texts):
        """准备数据"""
        print(f"\nPreparing text data...")
        print(f"Document count: {len(texts)}")

        # 根据数据量调整参数
        if len(texts) < 500:
            max_features = 500
            min_df = 3
        elif len(texts) < 1000:
            max_features = 800
            min_df = 5
        else:
            max_features = 1000
            min_df = 8

        self.vectorizer = CountVectorizer(
            max_df=0.85,
            min_df=min_df,
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b'
        )

        self.full_dtm = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"✓ Document-term matrix shape: {self.full_dtm.shape}")
        print(f"✓ Vocabulary size: {len(self.feature_names)}")

        return self.full_dtm

    def calculate_topic_diversity(self, topic_terms):
        """计算主题多样性"""
        all_keywords = []
        for terms in topic_terms:
            all_keywords.extend(terms[:5])

        unique_keywords = set(all_keywords)
        total_keywords = len(all_keywords)

        if total_keywords > 0:
            diversity_score = len(unique_keywords) / total_keywords
        else:
            diversity_score = 0

        return diversity_score

    def evaluate_models(self):
        """评估主题模型"""
        print(f"\nEvaluating topic models...")
        print(f"Topic range: {self.min_topics} to {self.max_topics}")

        perplexities = []
        diversity_scores = []
        avg_distances = []
        min_distances = []
        topics_range = range(self.min_topics, self.max_topics + 1)

        for n_topics in topics_range:
            print(f"\n  Testing topics: {n_topics}")

            # 使用优化参数
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=50,
                learning_method='batch',
                random_state=42,
                n_jobs=-1,
                doc_topic_prior=0.1,
                topic_word_prior=0.01,
                verbose=0
            )

            # 训练模型
            lda.fit(self.full_dtm)
            perplexity = lda.perplexity(self.full_dtm)

            # 获取主题关键词
            topic_terms = self.get_topic_terms(lda, n_words=15)

            # 计算各项指标
            diversity_score = self.calculate_topic_diversity(topic_terms)

            # 计算主题间距离
            topic_distance, avg_distance, min_distance = self.calculate_topic_distance(lda)

            # 存储距离矩阵
            self.topic_distances[n_topics] = topic_distance

            perplexities.append(perplexity)
            diversity_scores.append(diversity_score)
            avg_distances.append(avg_distance)
            min_distances.append(min_distance)

            print(f"    Perplexity: {perplexity:.2f}")
            print(f"    Topic diversity: {diversity_score:.3f}")
            print(f"    Avg topic distance: {avg_distance:.3f}")
            print(f"    Min topic distance: {min_distance:.3f}")

            # 保存结果
            self.evaluation_results[n_topics] = {
                'perplexity': perplexity,
                'model': lda,
                'topic_terms': topic_terms,
                'diversity_score': diversity_score,
                'avg_distance': avg_distance,
                'min_distance': min_distance
            }

        return topics_range, perplexities, diversity_scores, avg_distances, min_distances

    def get_topic_terms(self, lda_model, n_words=15):
        """获取主题关键词"""
        topic_terms = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            topic_terms.append(top_words)
        return topic_terms

    def calculate_topic_distance(self, lda_model):
        """计算主题间距离"""
        topic_similarity = cosine_similarity(lda_model.components_)
        topic_distance = 1 - topic_similarity

        triu_indices = np.triu_indices_from(topic_distance, k=1)
        distances = topic_distance[triu_indices]

        avg_distance = np.mean(distances) if len(distances) > 0 else 0
        min_distance = np.min(distances) if len(distances) > 0 else 0

        return topic_distance, avg_distance, min_distance

    def get_best_model_distances(self):
        """获取最佳模型的主题间距"""
        if self.best_n_topics and self.best_n_topics in self.topic_distances:
            return self.topic_distances[self.best_n_topics]
        return None

    def find_optimal_topics_balanced(self):
        """平衡评估寻找最佳主题数"""
        print(f"\nFinding optimal number of topics...")

        topics = list(self.evaluation_results.keys())
        perplexities = [self.evaluation_results[t]['perplexity'] for t in topics]
        diversity_scores = [self.evaluation_results[t]['diversity_score'] for t in topics]

        # 策略1：直接选择困惑度最低的
        min_perplexity_idx = np.argmin(perplexities)
        best_by_perplexity = topics[min_perplexity_idx]

        # 策略2：选择多样性较高的（至少0.8以上）
        high_diversity_topics = []
        for i, (topic, diversity) in enumerate(zip(topics, diversity_scores)):
            if diversity >= 0.8:
                high_diversity_topics.append((topic, diversity, perplexities[i]))

        if high_diversity_topics:
            # 在多样性高的主题中选择困惑度较低的
            high_diversity_topics.sort(key=lambda x: x[2])  # 按困惑度排序
            best_by_diversity = high_diversity_topics[0][0]
        else:
            best_by_diversity = best_by_perplexity

        # 策略3：肘部点检测
        elbow_point = None
        if len(perplexities) >= 4:
            improvements = []
            for i in range(1, len(perplexities)):
                improvement = perplexities[i - 1] - perplexities[i]
                improvements.append(improvement)

            # 寻找改进率显著下降的点
            for i in range(1, len(improvements) - 1):
                if improvements[i] < 0.3 * improvements[i - 1] and improvements[i] < 2.0:
                    elbow_point = topics[i]
                    break

            if elbow_point:
                print(f"  Found elbow point at {elbow_point} topics")

        # 最终决策：优先选择困惑度最低的，但考虑主题数量合理性
        # 通常6-8个主题对于分析比较合适
        if best_by_perplexity >= 6 and best_by_perplexity <= 8:
            best_topic = best_by_perplexity
            reason = "Lowest perplexity within optimal range (6-8)"
        elif best_by_diversity >= 6 and best_by_diversity <= 8:
            best_topic = best_by_diversity
            reason = "High diversity within optimal range"
        else:
            # 调整到合理范围
            if best_by_perplexity < 6:
                best_topic = 6
                reason = f"Adjusted to minimum optimal topics (original: {best_by_perplexity})"
            elif best_by_perplexity > 8:
                best_topic = 8
                reason = f"Adjusted to maximum optimal topics (original: {best_by_perplexity})"
            else:
                best_topic = best_by_perplexity
                reason = "Lowest perplexity"

        print(f"\n✓ Recommended topics: {best_topic} ({reason})")
        print(f"  Perplexity: {self.evaluation_results[best_topic]['perplexity']:.2f}")
        print(f"  Topic diversity: {self.evaluation_results[best_topic]['diversity_score']:.3f}")
        print(f"  Average topic distance: {self.evaluation_results[best_topic]['avg_distance']:.3f}")

        self.best_n_topics = best_topic
        self.best_lda = self.evaluation_results[best_topic]['model']

        return best_topic


# 7. 完美的主题标签生成类
class PerfectTopicLabeler:

    def __init__(self):
        self.used_labels = set()
        self.used_domains = set()

        # 领域关键词映射（无重叠）
        self.domain_keywords = {
            'Fertility_Encouragement': ['鼓励', '期待', '感谢', '年轻人', '未来', '希望', '积极', '正面'],
            'Fertility_Support': ['支持', '保障', '责任', '分担', '共同', '家庭', '托育', '稳定'],
            'Childcare_Services': ['托育', '幼儿园', '托管', '保育', '普惠', '公立', '服务', '质量'],
            'Education_Equity': ['公平', '平等', '资源', '分配', '均衡', '机会', '无差别', '公正'],
            'Education_Quality': ['质量', '水平', '标准', '优质', '提升', '改善', '优化', '发展'],
            'Workplace_Equality': ['职场', '工作', '就业', '歧视', '女性', '保障', '权益', '产假'],
            'Economic_Subsidies': ['补贴', '经济', '成本', '费用', '负担', '压力', '支持', '福利'],
            'Housing_Security': ['住房', '房价', '租房', '公租房', '购房', '居住', '成本', '空间'],
            'Healthcare_Coverage': ['医疗', '健康', '保险', '看病', '医院', '费用', '保障', '儿童'],
            'Social_Support': ['社会', '支持', '环境', '观念', '文化', '理解', '包容', '氛围'],
            'Family_Support': ['家庭', '父母', '孩子', '亲子', '夫妻', '时间', '精力', '关系'],
            'Resource_Allocation': ['资源', '配置', '分配', '供给', '需求', '匹配', '合理', '优化']
        }

        # 方面词库
        self.aspect_words = {
            'implementation': ['落实', '执行', '实施', '推进', '兑现', '到位', '有效'],
            'equity': ['公平', '平等', '均衡', '公正', '无差别', '机会均等'],
            'quality': ['质量', '水平', '标准', '优质', '提升', '改善', '优秀'],
            'cost': ['成本', '费用', '支出', '负担', '压力', '经济压力'],
            'rights': ['权益', '权利', '保障', '保护', '维护', '基本权利'],
            'support': ['支持', '帮助', '协助', '配套', '服务', '援助', '帮扶'],
            'environment': ['环境', '氛围', '条件', '状况', '生态', '文化环境'],
            'system': ['体系', '系统', '机制', '制度', '框架', '结构']
        }

    def identify_domain_precise(self, keywords):
        """精确识别主题领域 - 避免重复"""
        domain_scores = {}

        # 第一轮：精确匹配
        for domain, domain_words in self.domain_keywords.items():
            score = 0
            for keyword in keywords[:12]:  # 只看前12个关键词
                if keyword in domain_words:
                    score += 3  # 精确匹配得分高
                elif any(word in keyword for word in domain_words):
                    score += 2  # 部分包含得分中
                elif any(keyword in word for word in domain_words):
                    score += 1  # 被包含得分低

            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return 'Comprehensive_Support'

        # 排除已使用的领域（除非得分非常高）
        available_domains = []
        for domain, score in domain_scores.items():
            if domain not in self.used_domains:
                available_domains.append((domain, score))
            elif score > 10:  # 得分非常高，可以考虑重复使用
                print(f"  Warning: Domain '{domain}' already used but high score ({score})")
                available_domains.append((domain, score))

        if available_domains:
            # 选择得分最高的可用领域
            available_domains.sort(key=lambda x: x[1], reverse=True)
            selected_domain = available_domains[0][0]
        else:
            # 所有相关领域都已使用，选择得分最高的
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            selected_domain = sorted_domains[0][0]

        return selected_domain

    def identify_aspect_precise(self, keywords, domain):
        """精确识别方面"""
        aspect_scores = {}

        for aspect, aspect_list in self.aspect_words.items():
            score = 0
            for keyword in keywords[:10]:
                if keyword in aspect_list:
                    score += 2
                elif any(word in keyword for word in aspect_list):
                    score += 1

            if score > 0:
                aspect_scores[aspect] = score

        if aspect_scores:
            # 根据领域调整方面优先级
            domain_aspect_priority = {
                'Fertility_Encouragement': ['implementation', 'support', 'environment'],
                'Fertility_Support': ['system', 'support', 'rights'],
                'Childcare_Services': ['quality', 'cost', 'system'],
                'Education_Equity': ['equity', 'implementation', 'system'],
                'Education_Quality': ['quality', 'improvement', 'environment'],
                'Workplace_Equality': ['equity', 'rights', 'implementation'],
                'Economic_Subsidies': ['implementation', 'cost', 'support'],
                'Housing_Security': ['cost', 'system', 'implementation'],
                'Healthcare_Coverage': ['cost', 'quality', 'system'],
                'Social_Support': ['environment', 'system', 'support'],
                'Family_Support': ['support', 'time', 'system'],
                'Resource_Allocation': ['equity', 'system', 'implementation']
            }

            # 如果领域有优先级列表，调整分数
            if domain in domain_aspect_priority:
                priority_list = domain_aspect_priority[domain]
                for i, aspect in enumerate(priority_list):
                    if aspect in aspect_scores:
                        aspect_scores[aspect] += (len(priority_list) - i) * 0.5

            # 选择得分最高的方面
            return max(aspect_scores.items(), key=lambda x: x[1])[0]
        else:
            # 根据领域选择默认方面
            default_aspects = {
                'Fertility_Encouragement': 'implementation',
                'Fertility_Support': 'system',
                'Childcare_Services': 'quality',
                'Education_Equity': 'equity',
                'Education_Quality': 'quality',
                'Workplace_Equality': 'rights',
                'Economic_Subsidies': 'implementation',
                'Housing_Security': 'system',
                'Healthcare_Coverage': 'system',
                'Social_Support': 'environment',
                'Family_Support': 'support',
                'Resource_Allocation': 'system'
            }
            return default_aspects.get(domain, 'improvement')

    def create_unique_perfect_label(self, keywords, topic_id, domain, aspect):
        """创建完美唯一的主题标签"""

        # 基础标签模板
        label_templates = {
            'Fertility_Encouragement': {
                'implementation': 'Fertility Policy Implementation',
                'support': 'Fertility Support System',
                'environment': 'Fertility Social Environment',
                'default': 'Fertility Policy Enhancement'
            },
            'Fertility_Support': {
                'system': 'Fertility Support System',
                'support': 'Fertility Support Measures',
                'rights': 'Fertility Rights Protection',
                'default': 'Fertility Policy Optimization'
            },
            'Childcare_Services': {
                'quality': 'Childcare Quality Improvement',
                'cost': 'Childcare Cost Optimization',
                'system': 'Childcare System Enhancement',
                'default': 'Childcare Policy Support'
            },
            'Education_Equity': {
                'equity': 'Education Resource Equity',
                'implementation': 'Education Equity Implementation',
                'system': 'Education Equity System',
                'default': 'Education Equity Promotion'
            },
            'Education_Quality': {
                'quality': 'Education Quality Assurance',
                'improvement': 'Education Quality Improvement',
                'environment': 'Education Support Environment',
                'default': 'Education Policy Enhancement'
            },
            'Workplace_Equality': {
                'equity': 'Gender Equality in Workplace',
                'rights': 'Workplace Rights Protection',
                'implementation': 'Workplace Equality Implementation',
                'default': 'Workplace Equality Policy'
            },
            'Economic_Subsidies': {
                'implementation': 'Economic Subsidy Implementation',
                'cost': 'Fertility Cost Reduction',
                'support': 'Economic Support Policy',
                'default': 'Economic Subsidy Optimization'
            },
            'Housing_Security': {
                'cost': 'Housing Cost Management',
                'system': 'Housing Security System',
                'implementation': 'Housing Policy Implementation',
                'default': 'Housing Policy Enhancement'
            },
            'Healthcare_Coverage': {
                'cost': 'Medical Cost Sharing',
                'quality': 'Healthcare Quality Assurance',
                'system': 'Healthcare System Development',
                'default': 'Healthcare Policy Optimization'
            },
            'Social_Support': {
                'environment': 'Social Support Environment',
                'system': 'Social Support System',
                'support': 'Social Support Network',
                'default': 'Social Support Enhancement'
            },
            'Family_Support': {
                'support': 'Family Support System',
                'time': 'Family Time Balance',
                'system': 'Family Support System',
                'default': 'Family Policy Optimization'
            },
            'Resource_Allocation': {
                'equity': 'Resource Allocation Equity',
                'system': 'Resource Allocation System',
                'implementation': 'Resource Policy Implementation',
                'default': 'Resource Allocation Optimization'
            }
        }

        # 获取标签
        if domain in label_templates:
            if aspect in label_templates[domain]:
                base_label = label_templates[domain][aspect]
            else:
                base_label = label_templates[domain]['default']
        else:
            base_label = f"{domain.replace('_', ' ')} {aspect.title()}"

        # 确保标签唯一
        final_label = base_label
        suffix_num = 1

        # 检查是否同一领域重复使用
        domain_use_count = sum(1 for label in self.used_labels if domain in label)

        while final_label in self.used_labels:
            if domain_use_count >= 1:
                if suffix_num == 1:
                    final_label = f"{base_label} (Priority)"
                elif suffix_num == 2:
                    final_label = f"{base_label} (Deepening)"
                elif suffix_num == 3:
                    final_label = f"{base_label} (Strengthening)"
                else:
                    final_label = f"{base_label} (Topic {suffix_num})"
            else:
                final_label = f"{base_label} (Topic {suffix_num})"

            suffix_num += 1

            # 安全限制
            if suffix_num > 10:
                final_label = f"{base_label}_Unique_{topic_id}"
                break

        self.used_labels.add(final_label)
        self.used_domains.add(domain)

        return final_label


# 8. 代表性建议提取器
class RepresentativeExtractor:
    """代表性建议提取器"""

    def extract_representative_text(self, texts, topic_distribution, topic_id, topic_keywords):
        """提取代表性建议"""
        topic_weights = topic_distribution[:, topic_id]

        # 找到权重最高的前30个文档
        top_indices = np.argsort(topic_weights)[-30:][::-1]

        best_score = -1
        best_text = ""
        best_idx = -1

        for idx in top_indices:
            text = texts[idx]
            weight = topic_weights[idx]

            # 计算文本与主题关键词的匹配度
            match_score = self.calculate_match_score(text, topic_keywords)

            # 综合评分：权重 * (1 + 匹配度 * 权重因子)
            total_score = weight * (1 + match_score * 0.3)

            if total_score > best_score and len(text.strip()) > 10:
                best_score = total_score
                best_text = text
                best_idx = idx

        # 处理文本长度
        if best_text:
            cleaned_text = re.sub(r'\s+', ' ', best_text).strip()

            if len(cleaned_text) > 120:
                # 尝试在句子边界处截断
                sentences = re.split(r'[。！？；]', cleaned_text)
                if len(sentences) > 1:
                    truncated = sentences[0]
                    # 确保截断后文本有意义
                    if len(truncated) >= 20:
                        truncated += '。'
                    else:
                        # 如果第一句太短，加入第二句
                        truncated = sentences[0] + sentences[1] + '。'

                    if len(truncated) > 120:
                        truncated = truncated[:117] + '...'
                else:
                    # 没有句子分隔符，在逗号处截断
                    parts = cleaned_text.split('，')
                    if len(parts) > 1:
                        truncated = '，'.join(parts[:2]) + '...'
                    else:
                        truncated = cleaned_text[:117] + '...'
            else:
                truncated = cleaned_text
        else:
            truncated = "No representative suggestion available"

        return truncated, best_idx, best_score

    def calculate_match_score(self, text, topic_keywords):
        """计算匹配度"""
        if not text or not topic_keywords:
            return 0

        # 分词并去重
        words = set(jieba.lcut(text))

        match_count = 0
        for keyword in topic_keywords[:10]:  # 只考虑前10个关键词
            if keyword in words:
                match_count += 2
            elif any(word in keyword for word in words):
                match_count += 1
            elif any(keyword in word for word in words):
                match_count += 1

        # 归一化匹配度
        max_possible = min(10, len(topic_keywords)) * 2
        return match_count / max_possible if max_possible > 0 else 0


# 9. 红色主题的可视化类（从淡红到深红，无粉色）
class RedThemeVisualizer:
    """红色主题可视化类 - 使用从淡红到深红的渐变，无粉色"""

    def __init__(self, config):
        self.config = config
        # 红色渐变定义：从淡红 (#FFE4E1) 到深红 (#8B0000)，中间无粉色
        self.red_gradient = {
            'lightest': '#FFE4E1',  # 淡红 (Misty Rose)
            'light': '#F08080',  # 中浅红 (Light Coral)
            'medium': '#CD5C5C',  # 中红 (Indian Red)
            'medium_dark': '#B22222',  # 中深红 (Firebrick)
            'dark': '#8B0000',  # 深红 (Dark Red)
            'darkest': '#660000'  # 最深红
        }

        # 主要颜色定义
        self.colors = {
            'primary': '#CD5C5C',  # 主要折线颜色 (Indian Red)
            'secondary': '#B22222',  # 次要颜色 (Firebrick)
            'accent': '#8B0000',  # 强调色 (Dark Red)
            'highlight': '#DC143C',  # 高亮色 (Crimson)
            'background': '#FFF5F5',  # 背景色 (非常浅的红)
            'text': '#660000',  # 文字颜色 (最深红)
            'grid': '#D3D3D3',  # 网格颜色 (浅灰)
            'improvement_pos': '#006400',  # 正改进率颜色 (深绿色)
            'improvement_neg': '#8B0000',  # 负改进率颜色 (深红色)
            'lightest': '#FFE4E1',  # 淡红
            'light': '#F08080',  # 中浅红
            'medium': '#CD5C5C',  # 中红
            'medium_dark': '#B22222',  # 中深红
            'dark': '#8B0000',  # 深红
            'darkest': '#660000'  # 最深红
        }

        # 红色渐变序列 (从淡到深)
        self.red_sequence = [
            '#FFE4E1',  # 淡红
            '#F08080',  # 中浅红
            '#CD5C5C',  # 中红
            '#B22222',  # 中深红
            '#8B0000',  # 深红
            '#660000'  # 最深红
        ]

    def _setup_style(self):
        """设置绘图风格"""
        # 检查可用的样式
        available_styles = plt.style.available

        if 'seaborn-v0_8-whitegrid' in available_styles:
            plt.style.use('seaborn-v0_8-whitegrid')
        elif 'seaborn-whitegrid' in available_styles:
            plt.style.use('seaborn-whitegrid')
        elif 'ggplot' in available_styles:
            plt.style.use('ggplot')
        else:
            plt.style.use('default')

        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'grid.color': self.colors['grid'],
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titlecolor': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
        })

    def plot_perplexity_analysis(self, topics_range, perplexities, best_n_topics):
        """绘制红色主题的困惑度分析图（从淡红到深红渐变）"""
        print(f"\n{'=' * 60}")
        print("生成红色渐变困惑度分析图...")
        print(f"{'=' * 60}")
        print(f"主题范围: {min(topics_range)} - {max(topics_range)}")
        print(f"困惑度值: {[f'{p:.1f}' for p in perplexities]}")
        print(f"最佳主题数: {best_n_topics}")

        self._setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Red Gradient Perplexity Analysis',
                     fontsize=16, fontweight='bold', color=self.colors['text'], y=1.02)

        # 1. 红色渐变的困惑度折线图
        x = list(topics_range)
        y = list(perplexities)

        # 绘制渐变折线 - 使用红色渐变
        for i in range(len(x) - 1):
            # 根据位置选择颜色 (从淡到深)
            color_idx = int((i / (len(x) - 2)) * (len(self.red_sequence) - 1))
            color_idx = min(color_idx, len(self.red_sequence) - 1)
            line_color = self.red_sequence[color_idx]

            ax1.plot(x[i:i + 2], y[i:i + 2], '-',
                     color=line_color, linewidth=3,
                     solid_capstyle='round', zorder=2)

        # 绘制数据点 - 使用从淡到深的渐变
        for i, (xi, yi) in enumerate(zip(x, y)):
            color_idx = int((i / (len(x) - 1)) * (len(self.red_sequence) - 1))
            color_idx = min(color_idx, len(self.red_sequence) - 1)
            point_color = self.red_sequence[color_idx]

            ax1.scatter(xi, yi, s=120, color=point_color,
                        edgecolor=self.colors['darkest'], linewidth=2,
                        zorder=3)

        # 标记最佳点
        best_idx = list(topics_range).index(best_n_topics)
        ax1.scatter(best_n_topics, perplexities[best_idx],
                    s=250, color=self.colors['accent'], zorder=4,
                    edgecolor=self.colors['darkest'], linewidth=2.5,
                    label=f'Optimal: {best_n_topics} topics\nPerplexity: {perplexities[best_idx]:.1f}')

        # 设置标签和标题
        ax1.set_xlabel('Number of Topics', fontweight='bold', color=self.colors['text'])
        ax1.set_ylabel('Perplexity', fontweight='bold', color=self.colors['text'])
        ax1.set_title('Perplexity Analysis (Red Gradient)',
                      fontweight='bold', pad=15, color=self.colors['text'])
        ax1.grid(True, alpha=0.2, linestyle='--', color=self.colors['grid'])
        ax1.legend(loc='best', framealpha=0.9, edgecolor=self.colors['primary'])

        # 美化坐标轴
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color(self.colors['text'])
        ax1.spines['bottom'].set_color(self.colors['text'])

        # 2. 红色渐变的改进率分析
        improvements = np.diff(perplexities)

        # 为每个柱使用不同的红色渐变
        for i, (x_val, imp) in enumerate(zip(topics_range[1:], improvements)):
            color_idx = int((i / (len(improvements) - 1)) * (len(self.red_sequence) - 1))
            color_idx = min(color_idx, len(self.red_sequence) - 1)
            bar_color = self.red_sequence[color_idx]

            bar = ax2.bar(x_val, imp, width=0.6,
                          color=bar_color,
                          edgecolor=self.colors['darkest'],
                          linewidth=1.5, alpha=0.9)

        ax2.axhline(y=0, color=self.colors['darkest'], linewidth=1.2, linestyle='-', alpha=0.5)
        ax2.set_xlabel('Number of Topics', fontweight='bold', color=self.colors['text'])
        ax2.set_ylabel('Perplexity Improvement', fontweight='bold', color=self.colors['text'])
        ax2.set_title('Marginal Improvement Analysis',
                      fontweight='bold', pad=15, color=self.colors['text'])
        ax2.grid(True, alpha=0.2, linestyle='--', axis='y', color=self.colors['grid'])

        # 添加数值标签
        for i, (x, y_val) in enumerate(zip(topics_range[1:], improvements)):
            if y_val > 0:
                color = self.colors['improvement_pos']
                va = 'bottom'
                y_offset = 0.1
            else:
                color = self.colors['improvement_neg']
                va = 'top'
                y_offset = -0.8

            ax2.text(x, y_val + y_offset, f'{y_val:.1f}',
                     ha='center', va=va,
                     fontsize=10, fontweight='bold', color=color,
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor='white', alpha=0.9,
                               edgecolor=color))

        # 美化坐标轴
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(self.colors['text'])
        ax2.spines['bottom'].set_color(self.colors['text'])

        plt.tight_layout()

        output_path = os.path.join(self.config.charts_dir, "Perplexity_Analysis_Red_Gradient.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 红色渐变困惑度分析图已保存: {output_path}")
        plt.show()
        print(f"✓ 红色渐变困惑度分析图已显示")

    def plot_topic_metrics_comparison(self, topics_range, perplexities, diversity_scores, avg_distances, best_n_topics):
        """绘制红色渐变的主题指标对比图"""
        print(f"\nGenerating red gradient topic metrics comparison chart...")

        self._setup_style()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Red Gradient Topic Model Performance Metrics',
                     fontsize=16, fontweight='bold', color=self.colors['text'], y=0.98)

        # 定义指标
        metrics = [
            ('Perplexity', perplexities, True, 'lower'),
            ('Topic Diversity', diversity_scores, False, 'higher'),
            ('Avg Topic Distance', avg_distances, False, 'higher'),
        ]

        axes_flat = axes.flatten()

        for i, (metric_name, values, is_perplexity, better_direction) in enumerate(metrics[:3]):
            ax = axes_flat[i]
            x = list(topics_range)
            y = list(values)

            # 绘制红色渐变的折线
            for j in range(len(x) - 1):
                color_idx = int((j / (len(x) - 2)) * (len(self.red_sequence) - 1))
                color_idx = min(color_idx, len(self.red_sequence) - 1)
                line_color = self.red_sequence[color_idx]

                ax.plot(x[j:j + 2], y[j:j + 2], '-',
                        color=line_color, linewidth=2.5,
                        solid_capstyle='round', zorder=2)

            # 绘制数据点
            for j, (xj, yj) in enumerate(zip(x, y)):
                color_idx = int((j / (len(x) - 1)) * (len(self.red_sequence) - 1))
                color_idx = min(color_idx, len(self.red_sequence) - 1)
                point_color = self.red_sequence[color_idx]

                ax.scatter(xj, yj, s=100, color=point_color,
                           edgecolor=self.colors['darkest'], linewidth=2, zorder=3)

            # 标记最佳点
            if is_perplexity:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)

            best_topic = topics_range[best_idx]
            best_value = values[best_idx]

            ax.scatter(best_topic, best_value, s=150, color=self.colors['accent'],
                       zorder=4, edgecolor=self.colors['darkest'], linewidth=2)

            ax.set_xlabel('Number of Topics', fontweight='bold', color=self.colors['text'])
            ax.set_ylabel(metric_name, fontweight='bold', color=self.colors['text'])

            better_text = 'Lower' if better_direction == 'lower' else 'Higher'
            ax.set_title(f'{metric_name} ({better_text} is Better)',
                         fontweight='bold', pad=10, color=self.colors['text'])

            ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['grid'])

            # 美化坐标轴
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.colors['text'])
            ax.spines['bottom'].set_color(self.colors['text'])

            # 添加最佳值标签
            ax.text(0.95, 0.95, f'Best: {best_value:.3f}\n@ {best_topic} topics',
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=self.colors['background'],
                              alpha=0.9,
                              edgecolor=self.colors['primary']),
                    color=self.colors['text'])

        # 4. 综合评分图
        ax4 = axes_flat[3]

        # 归一化函数
        def normalize(values, reverse=False):
            values = np.array(values)
            if len(values) == 0 or np.max(values) - np.min(values) == 0:
                return np.ones_like(values) * 0.5
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            return 1 - normalized if reverse else normalized

        # 归一化各项指标
        norm_perplexity = normalize(perplexities, reverse=True)
        norm_diversity = normalize(diversity_scores, reverse=False)
        norm_distance = normalize(avg_distances, reverse=False)

        # 计算综合评分
        composite_scores = []
        for i in range(len(topics_range)):
            score = (0.5 * norm_perplexity[i] +
                     0.25 * norm_diversity[i] +
                     0.25 * norm_distance[i])
            composite_scores.append(score)

        x = list(topics_range)
        y = composite_scores

        # 绘制红色渐变的综合评分
        for i in range(len(x) - 1):
            color_idx = int((i / (len(x) - 2)) * (len(self.red_sequence) - 1))
            color_idx = min(color_idx, len(self.red_sequence) - 1)
            line_color = self.red_sequence[color_idx]

            ax4.plot(x[i:i + 2], y[i:i + 2], '-',
                     color=line_color, linewidth=2.5,
                     solid_capstyle='round', zorder=2)

        # 绘制数据点
        for i, (xi, yi) in enumerate(zip(x, y)):
            color_idx = int((i / (len(x) - 1)) * (len(self.red_sequence) - 1))
            color_idx = min(color_idx, len(self.red_sequence) - 1)
            point_color = self.red_sequence[color_idx]

            ax4.scatter(xi, yi, s=100, color=point_color,
                        edgecolor=self.colors['darkest'], linewidth=2, zorder=3)

        # 标记最佳综合评分
        best_composite_idx = np.argmax(composite_scores)
        best_composite_topic = topics_range[best_composite_idx]
        best_composite_score = composite_scores[best_composite_idx]

        ax4.scatter(best_composite_topic, best_composite_score,
                    s=150, color=self.colors['accent'], zorder=4,
                    edgecolor=self.colors['darkest'], linewidth=2,
                    label=f'Best Score: {best_composite_score:.3f}\n@ {best_composite_topic} topics')

        ax4.set_xlabel('Number of Topics', fontweight='bold', color=self.colors['text'])
        ax4.set_ylabel('Composite Score', fontweight='bold', color=self.colors['text'])
        ax4.set_title('Model Composite Performance Score',
                      fontweight='bold', pad=10, color=self.colors['text'])
        ax4.grid(True, alpha=0.2, linestyle='--', color=self.colors['grid'])
        ax4.legend(loc='lower right', framealpha=0.9, edgecolor=self.colors['primary'])

        # 美化坐标轴
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color(self.colors['text'])
        ax4.spines['bottom'].set_color(self.colors['text'])

        plt.tight_layout()

        output_path = os.path.join(self.config.charts_dir, "Topic_Metrics_Comparison_Red_Gradient.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Red gradient topic metrics comparison chart saved: {output_path}")
        plt.show()


# 10. 主程序
def main():
    print("=" * 60)
    print("Open-ended Text Analysis System (Red Gradient Theme)")
    print("=" * 60)

    setup_chinese_font()

    # Step 1: 初始化配置
    print("\n1. Initializing configuration...")
    config = FileConfig()

    # Step 2: 加载数据
    print("\n2. Loading data...")
    loader = DataLoader(config)

    df = loader.load_excel_data()

    if df is None or len(df) == 0:
        print("Error: No data file found")
        return

    print(f"✓ Data loaded successfully, {len(df)} records")

    # Step 3: 数据预处理
    print("\n3. Data preprocessing...")
    preprocessor = ImprovedTextPreprocessor(config)
    df_processed, word_freq = preprocessor.preprocess_pipeline(df)

    valid_texts = df_processed['Cleaned_Text'].apply(lambda x: len(x.strip()) > 0).sum()
    print(f"✓ Valid texts: {valid_texts} records")

    # Step 4: 主题模型评估
    print("\n4. Topic model evaluation...")

    # 根据数据量确定最大主题数
    max_topics = min(10, max(8, valid_texts // 100))
    max_topics = max(4, max_topics)

    evaluator = OptimizedTopicModelEvaluator(min_topics=4, max_topics=max_topics)

    # 准备数据
    full_dtm = evaluator.prepare_data(df_processed['Segmented_Text'].tolist())

    # 评估模型
    topics_range, perplexities, diversity_scores, avg_distances, min_distances = evaluator.evaluate_models()

    # 寻找最佳主题数
    best_n_topics = evaluator.find_optimal_topics_balanced()

    # 使用最佳模型
    best_results = evaluator.evaluation_results[best_n_topics]
    best_lda = best_results['model']
    topic_terms = best_results['topic_terms']

    print(f"\nBest model statistics:")
    print(f"  Number of topics: {best_n_topics}")
    print(f"  Perplexity: {best_results['perplexity']:.2f}")
    print(f"  Topic diversity: {best_results['diversity_score']:.3f}")
    print(f"  Average topic distance: {best_results['avg_distance']:.3f}")
    print(f"  Minimum topic distance: {best_results['min_distance']:.3f}")

    # Step 5: 生成主题分析结果
    print("\n5. Generating topic analysis results...")

    # 预测主题分布
    topic_distribution = best_lda.transform(full_dtm)

    # 创建主题标签
    labeler = PerfectTopicLabeler()
    extractor = RepresentativeExtractor()

    topics_data = []
    for topic_id in range(best_n_topics):
        keywords = topic_terms[topic_id]

        # 精确识别领域
        domain = labeler.identify_domain_precise(keywords)

        # 精确识别方面
        aspect = labeler.identify_aspect_precise(keywords, domain)

        # 创建完美唯一的标签
        label = labeler.create_unique_perfect_label(keywords, topic_id + 1, domain, aspect)

        # 提取代表性建议
        rep_text, rep_index, match_score = extractor.extract_representative_text(
            df_processed['Cleaned_Text'].tolist(),
            topic_distribution,
            topic_id,
            keywords
        )

        # 计算主题权重
        topic_weight = np.sum(best_lda.components_[topic_id])

        # 计算涉及文档数
        doc_weights = topic_distribution[:, topic_id]
        doc_count = np.sum(doc_weights > 0.1)

        topics_data.append({
            'Topic_ID': topic_id + 1,
            'Domain': domain,
            'Aspect': aspect,
            'Topic_Label': label,
            'Top_Keywords': '，'.join(keywords[:8]),
            'Keyword_List': keywords,
            'Topic_Weight': round(topic_weight, 3),
            'Document_Count': int(doc_count),
            'Representative_Text': rep_text,
            'Match_Score': round(match_score, 3),
            'Representative_Doc_ID': rep_index + 1 if rep_index >= 0 else None
        })

    topics_df = pd.DataFrame(topics_data)

    # 按主题权重排序
    topics_df = topics_df.sort_values('Topic_Weight', ascending=False).reset_index(drop=True)
    topics_df['Topic_ID'] = range(1, len(topics_df) + 1)

    print("\nTopic analysis results:")
    print("=" * 120)
    display_columns = ['Topic_ID', 'Domain', 'Topic_Label', 'Top_Keywords', 'Document_Count', 'Representative_Text']
    print(topics_df[display_columns].to_string(index=False))

    # 检查标签唯一性
    unique_labels = len(set(topics_df['Topic_Label']))
    total_labels = len(topics_df)
    print(f"\n✓ Label uniqueness: {unique_labels}/{total_labels} ({(unique_labels / total_labels * 100):.1f}%)")

    # 检查领域唯一性
    unique_domains = len(set(topics_df['Domain']))
    print(f"✓ Domain uniqueness: {unique_domains}/{total_labels} ({(unique_domains / total_labels * 100):.1f}%)")

    # Step 6: 生成红色渐变可视化图表
    print("\n6. Generating red gradient visualization charts...")
    visualizer = RedThemeVisualizer(config)

    # 绘制红色渐变的困惑度分析图
    visualizer.plot_perplexity_analysis(list(topics_range), perplexities, best_n_topics)

    # 绘制红色渐变的主题指标对比图
    visualizer.plot_topic_metrics_comparison(list(topics_range), perplexities, diversity_scores, avg_distances,
                                             best_n_topics)

    # 获取最佳模型的主题间距矩阵
    distance_matrix = evaluator.get_best_model_distances()
    if distance_matrix is not None:
        # 准备主题标签和权重
        topic_labels = topics_df['Topic_Label'].tolist()
        topic_weights = topics_df['Topic_Weight'].tolist()

    # Step 7: 保存结果
    print("\n7. Saving analysis results...")

    try:
        with pd.ExcelWriter(config.output_excel, engine='openpyxl') as writer:
            # 原始数据
            df.to_excel(writer, sheet_name='Raw_Data', index=False)

            # 预处理数据
            df_processed.to_excel(writer, sheet_name='Preprocessed_Data', index=False)

            # 主题分析结果
            topics_df.to_excel(writer, sheet_name='Topic_Analysis', index=False)

            # 模型评估结果
            eval_data = []
            for n_topics, results in evaluator.evaluation_results.items():
                eval_data.append({
                    'Topics': n_topics,
                    'Perplexity': results['perplexity'],
                    'Diversity': results['diversity_score'],
                    'Avg_Distance': results['avg_distance'],
                    'Min_Distance': results['min_distance'],
                    'Is_Best': 'Yes' if n_topics == best_n_topics else 'No'
                })

            eval_df = pd.DataFrame(eval_data)
            eval_df.to_excel(writer, sheet_name='Model_Evaluation', index=False)

            # 详细分析结果
            detailed_results = pd.DataFrame({
                'Doc_ID': df_processed['ID'],
                'Original_Text': df_processed['Original_Text'],
                'Cleaned_Text': df_processed['Cleaned_Text'],
                'Segmented_Text': df_processed['Segmented_Text']
            })

            for i in range(best_n_topics):
                detailed_results[f'Topic_{i + 1}_Weight'] = topic_distribution[:, i]

            detailed_results['Main_Topic_ID'] = np.argmax(topic_distribution, axis=1) + 1
            detailed_results['Main_Topic_Label'] = detailed_results['Main_Topic_ID'].map(
                dict(zip(range(1, best_n_topics + 1), topics_df['Topic_Label']))
            )
            detailed_results['Topic_Confidence'] = np.max(topic_distribution, axis=1)
            detailed_results['Topic_Confidence'] = detailed_results['Topic_Confidence'].round(3)

            detailed_results.to_excel(writer, sheet_name='Detailed_Analysis', index=False)

            # 词频统计
            word_freq_df = pd.DataFrame(word_freq.most_common(100), columns=['Word', 'Frequency'])
            word_freq_df.to_excel(writer, sheet_name='Top_100_Words', index=False)

            # 分析总结
            summary_data = {
                'Metric': ['Total Documents', 'Valid Texts', 'Optimal Topics', 'Perplexity',
                           'Topic Diversity', 'Avg Topic Distance', 'Min Topic Distance',
                           'Label Uniqueness', 'Domain Uniqueness', 'Avg Doc Count', 'Avg Match Score'],
                'Value': [len(df), valid_texts, best_n_topics,
                          f"{best_results['perplexity']:.2f}",
                          f"{best_results['diversity_score']:.3f}",
                          f"{best_results['avg_distance']:.3f}",
                          f"{best_results['min_distance']:.3f}",
                          f"{unique_labels}/{total_labels}",
                          f"{unique_domains}/{total_labels}",
                          f"{topics_df['Document_Count'].mean():.1f}",
                          f"{topics_df['Match_Score'].mean():.3f}"]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)

        print(f"✓ Analysis results saved: {config.output_excel}")

    except Exception as e:
        print(f"Error saving Excel file: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Analysis completed successfully!")
    print(f"{'=' * 60}")

    # 打印摘要
    result_summary = {
        'Total Documents': len(df),
        'Valid Texts': valid_texts,
        'Optimal Topics': best_n_topics,
        'Perplexity': f"{best_results['perplexity']:.2f}",
        'Topic Diversity': f"{best_results['diversity_score']:.3f}",
        'Avg Topic Distance': f"{best_results['avg_distance']:.3f}",
        'Min Topic Distance': f"{best_results['min_distance']:.3f}",
        'Label Uniqueness': f"{unique_labels}/{total_labels}",
        'Domain Uniqueness': f"{unique_domains}/{total_labels}",
        'Output File': config.output_excel
    }

    print("\nAnalysis Summary:")
    for key, value in result_summary.items():
        print(f"  {key}: {value}")

    print(f"\nDiscovered Topics ({best_n_topics}):")
    for _, row in topics_df.iterrows():
        print(f"  Topic{row['Topic_ID']}: {row['Topic_Label']}")
        print(f"      Domain: {row['Domain']} | Aspect: {row['Aspect']}")
        print(f"      Documents: {row['Document_Count']} | Match Score: {row['Match_Score']:.2f}")
        print(f"      Key Keywords: {', '.join(row['Keyword_List'][:5])}")

    print(f"{'=' * 60}")


# 11. 运行主程序
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram execution error: {str(e)}")
        import traceback

        traceback.print_exc()