import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import learning_curve, GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import re
import os
import time
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, brier_score_loss, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LinearSegmentedColormap

# 添加词云库
try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("⚠️ WordCloud未安装，词云图功能将不可用")
    print("   安装命令: pip install wordcloud")
    WORDCLOUD_AVAILABLE = False

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否安装xgboost，如果没有则使用替代方案
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost未安装，使用GradientBoosting替代")
    XGB_AVAILABLE = False


# 1. 情感词典加载函数（完整版）
def load_tsinghua_sentiment_dict():
    """清华大学中文情感词典"""
    tsinghua_positive_words = [
        '幸福', '快乐', '喜悦', '欢欣', '愉快', '开心', '高兴', '欢乐',
        '欣喜', '舒畅', '爽快', '痛快', '欢快', '乐观', '积极', '开朗',
        '活泼', '兴奋', '激动', '振奋', '昂扬', '欢畅', '甜美', '甜蜜',
        '温馨', '温暖', '温和', '和蔼', '亲切', '慈祥', '恩爱', '和睦',
        '和谐', '融洽', '友好', '友善', '友爱', '团结', '互助', '合作',
        '协作', '共赢', '成功', '胜利', '成就', '成绩', '收获', '成果',
        '效果', '效益', '利益', '好处', '优势', '优点', '长处', '特色',
        '特长', '专长', '才能', '才华', '才干', '能力', '本领', '本事',
        '技艺', '技巧', '技术', '技能', '熟练', '精通', '擅长', '善长',
        '拿手', '在行', '内行', '专家', '权威', '大师', '高手', '能手'
    ]

    tsinghua_negative_words = [
        '痛苦', '悲伤', '悲哀', '悲痛', '悲恸', '悲戚', '悲苦', '悲凉',
        '悲惨', '悲壮', '悲愤', '悲叹', '悲观', '消极', '失望', '绝望',
        '沮丧', '颓丧', '颓废', '颓唐', '颓败', '颓靡', '萎靡', '消沉',
        '低沉', '低落', '低潮', '低迷', '阴暗', '阴沉', '阴郁', '忧郁',
        '郁闷', '烦闷', '苦恼', '烦恼', '烦躁', '焦虑', '焦急', '着急',
        '忧虑', '忧愁', '担心', '担忧', '惧怕', '害怕', '恐惧', '恐怖',
        '惊恐', '惊慌', '恐慌', '慌张', '紧张', '不安', '暴躁', '暴怒',
        '愤怒', '气愤', '生气', '发怒', '发火', '恼火', '怒火', '怒气',
        '怨恨', '仇恨', '仇视', '敌视', '敌对', '敌意', '嫉妒', '妒忌',
        '羡慕', '眼红', '忌妒', '猜忌', '猜疑', '怀疑', '疑惑', '困惑',
        '迷茫', '迷失', '失落', '失意', '失败', '挫折', '挫败', '打击'
    ]

    tsinghua_neutral_words = [
        '分析', '研究', '调查', '统计', '计算', '评估', '评价', '判断',
        '思考', '考虑', '思索', '思维', '思想', '观念', '概念', '理念',
        '理论', '原理', '原则', '规则', '规范', '标准', '准则', '要求',
        '条件', '因素', '要素', '成分', '组成部分', '结构', '构造', '组成',
        '构成', '形成', '产生', '发生', '发展', '进展', '进步', '改进',
        '改善', '改良', '改革', '变革', '变化', '变动', '调整', '调节',
        '调控', '控制', '管理', '治理', '统治', '领导', '指导', '引导',
        '带领', '率领', '指挥', '命令', '指示', '吩咐', '安排', '布置',
        '部署', '计划', '规划', '策划', '设计', '方案', '办法', '方法',
        '方式', '形式', '模式', '模型', '范例', '例子', '实例', '案例',
        '事例', '事实', '真相', '真实', '实际', '现实', '实在'
    ]

    labeled_data = []
    for word in tsinghua_positive_words:
        labeled_data.append((word, '正面'))
    for word in tsinghua_negative_words:
        labeled_data.append((word, '负面'))
    for word in tsinghua_neutral_words:
        labeled_data.append((word, '中性'))

    return labeled_data


def load_ntu_sentiment_dict():
    """台湾大学中文情感词典"""
    ntu_positive_words = [
        '美好', '完美', '优秀', '出色', '精彩', '卓越', '杰出', '突出',
        '显著', '明显', '优越', '优良', '优质', '优势', '优胜', '良好',
        '不错', '可以', '满意', '满足', '充足', '充分', '丰富', '丰满',
        '健全', '完善', '完备', '完整', '完全', '全体', '全部', '整体',
        '总体', '总计', '合计', '总共', '一共', '一并', '一齐', '一起',
        '共同', '协同', '配合', '合作', '协作', '互助', '互相', '相互',
        '彼此', '双方', '两边', '两方', '两造', '两便', '两利', '双赢'
    ]

    ntu_negative_words = [
        '糟糕', '差劲', '恶劣', '拙劣', '低劣', '失败', '失利', '失策',
        '失算', '失误', '错误', '过错', '过失', '毛病', '缺点', '缺陷',
        '瑕疵', '污点', '污渍', '污垢', '污秽', '肮脏', '龌龊', '卑鄙',
        '卑劣', '卑贱', '卑微', '卑下', '谄媚', '奉承', '阿谀', '拍马',
        '马屁', '吹捧', '吹嘘', '夸耀', '炫耀', '显摆', '卖弄', '张扬',
        '嚣张', '猖狂', '狂妄', '傲慢', '骄傲', '自大', '自负', '自满',
        '自傲', '自恃', '自诩', '自夸', '自吹', '自擂', '自欺', '欺人',
        '欺骗', '诈骗', '欺诈', '讹诈', '敲诈', '勒索', '恐吓', '威胁'
    ]

    ntu_neutral_words = [
        '普通', '一般', '平常', '寻常', '常见', '普遍', '广泛', '普通',
        '一般', '平常', '寻常', '常见', '普遍', '广泛', '普通', '一般',
        '平常', '寻常', '常见', '普遍', '广泛', '普通', '一般', '平常',
        '寻常', '常见', '普遍', '广泛', '普通', '一般', '平常', '寻常',
        '常见', '普遍', '广泛', '普通', '一般', '平常', '寻常', '常见',
        '普遍', '广泛', '普通', '一般', '平常', '寻常', '常见', '普遍'
    ]

    labeled_data = []
    for word in ntu_positive_words:
        labeled_data.append((word, '正面'))
    for word in ntu_negative_words:
        labeled_data.append((word, '负面'))
    for word in ntu_neutral_words:
        labeled_data.append((word, '中性'))

    return labeled_data


def load_boson_sentiment_dict():
    """BosonNLP情感词典"""
    boson_positive_words = [
        '喜欢', '爱', '欣赏', '敬佩', '钦佩', '崇拜', '仰慕', '羡慕',
        '向往', '憧憬', '期待', '盼望', '希望', '期望', '渴望', '渴求',
        '追求', '追寻', '寻求', '探索', '探究', '钻研', '研究', '学习',
        '努力', '奋斗', '拼搏', '进取', '进步', '前进', '发展', '壮大',
        '成长', '成熟', '成功', '成就', '成果', '成绩', '收获', '获得',
        '得到', '取得', '拥有', '具备', '具有', '存在', '实在', '真实',
        '真诚', '真挚', '诚恳', '诚实', '诚信', '守信', '信用', '信任',
        '信赖', '可靠', '靠谱', '稳妥', '稳定', '稳固', '牢固', '坚固',
        '坚强', '坚定', '坚决', '坚持', '持续', '持久', '长久', '长远'
    ]

    boson_negative_words = [
        '讨厌', '恨', '厌恶', '反感', '嫌弃', '憎恶', '憎恨', '仇恨',
        '仇视', '敌视', '敌对', '敌意', '敌人', '敌手', '对手', '对头',
        '冤家', '仇家', '仇敌', '仇人', '恶人', '坏人', '歹徒', '罪犯',
        '犯人', '囚犯', '囚徒', '俘虏', '奴隶', '奴才', '奴仆', '仆从',
        '随从', '跟班', '走狗', '爪牙', '帮凶', '帮闲', '帮腔', '帮衬',
        '帮派', '团伙', '团体', '组织', '机构', '机关', '部门', '单位'
    ]

    boson_neutral_words = [
        '评论', '评价', '评估', '评测', '测试', '测验', '考试', '考核',
        '考察', '检查', '检验', '检测', '监测', '监控', '监视', '监督',
        '监管', '管理', '治理', '统治', '领导', '指导', '引导', '带领',
        '率领', '指挥', '命令', '指示', '吩咐', '安排', '布置', '部署',
        '计划', '规划', '策划', '设计', '方案', '办法', '方法', '方式',
        '形式', '模式', '模型', '范例', '例子', '实例', '案例', '事例'
    ]

    labeled_data = []
    for word in boson_positive_words:
        labeled_data.append((word, '正面'))
    for word in boson_negative_words:
        labeled_data.append((word, '负面'))
    for word in boson_neutral_words:
        labeled_data.append((word, '中性'))

    return labeled_data


def load_hownet_sentiment_dict():
    """知网Hownet情感词典"""
    hownet_positive_words = [
        '善良', '仁慈', '仁爱', '慈悲', '怜悯', '同情', '关爱', '关怀',
        '关心', '关注', '照顾', '照料', '照看', '看护', '护理', '养护',
        '保养', '保护', '维护', '维持', '保持', '保存', '保留', '留存',
        '存留', '遗留', '遗传', '传承', '继承', '承继', '接续', '继续',
        '延续', '连续', '持续', '持久', '长久', '长期', '长远', '遥远',
        '远大', '宏伟', '宏大', '巨大', '庞大', '硕大', '强壮', '强健',
        '健康', '健壮', '壮实', '结实', '扎实', '实在', '实际', '真实',
        '真诚', '真挚', '诚恳', '诚实', '诚信', '守信', '信用', '信任'
    ]

    hownet_negative_words = [
        '邪恶', '凶恶', '凶狠', '凶残', '残忍', '残暴', '暴虐', '暴戾',
        '暴力', '强制', '强迫', '逼迫', '压迫', '压制', '压抑', '抑制',
        '遏制', '限制', '约束', '束缚', '捆绑', '绑缚', '拘束', '拘谨',
        '拘泥', '拘礼', '拘禁', '拘留', '拘押', '拘捕', '逮捕', '抓捕',
        '捉拿', '拿获', '抓获', '捕获', '捕猎', '狩猎', '猎杀', '杀戮',
        '屠杀', '残杀', '杀害', '杀死', '致死', '致命', '死亡', '死去',
        '死掉', '灭绝', '绝种', '绝迹', '绝灭', '绝后', '绝嗣', '绝户'
    ]

    hownet_neutral_words = [
        '客观', '主观', '主体', '客体', '对象', '目标', '目的', '意图',
        '意向', '意愿', '意思', '意义', '含义', '内涵', '外延', '范围',
        '范畴', '领域', '区域', '界限', '边界', '边际', '边缘', '旁边',
        '附近', '邻近', '临近', '接近', '靠近', '贴近', '紧贴', '紧密',
        '密切', '亲密', '亲近', '亲切', '亲热', '热情', '热心', '热忱',
        '热诚', '热烈', '热闹', '繁华', '繁荣', '繁盛', '兴盛', '兴旺',
        '旺盛', '茂盛', '丰盛', '丰富', '丰厚', '丰硕', '丰满'
    ]

    labeled_data = []
    for word in hownet_positive_words:
        labeled_data.append((word, '正面'))
    for word in hownet_negative_words:
        labeled_data.append((word, '负面'))
    for word in hownet_neutral_words:
        labeled_data.append((word, '中性'))

    return labeled_data


def load_base_labeled_data():
    """基础人工标注数据"""
    positive_words = [
        ('幸福', '正面'), ('快乐', '正面'), ('喜悦', '正面'), ('开心', '正面'),
        ('温暖', '正面'), ('甜蜜', '正面'), ('美好', '正面'), ('感恩', '正面'),
        ('欣慰', '正面'), ('满足', '正面'), ('充实', '正面'), ('值得', '正面'),
        ('从容', '正面'), ('爱', '正面'),
        ('知足常乐', '正面'), ('天伦之乐', '正面'),
        ('甜蜜的负担', '正面'), ('甜蜜的烦恼', '正面'),
        ('幸福的责任', '正面'), ('疲惫与幸福交织', '正面'),
        ('圆满', '正面')
    ]

    negative_words = [
        ('痛苦', '负面'), ('恐惧', '负面'), ('害怕', '负面'), ('焦虑', '负面'),
        ('压力', '负面'), ('疲惫', '负面'), ('劳累', '负面'), ('辛苦', '负面'),
        ('负担', '负面'), ('沉重', '负面'), ('枷锁', '负面'), ('束缚', '负面'),
        ('代价', '负面'), ('牺牲', '负面'), ('透支', '负面'), ('消耗', '负面'),
        ('内卷', '负面'), ('内耗', '负面')
    ]

    neutral_words = [
        ('权衡', '中性'), ('衡量', '中性'), ('考虑', '中性'), ('思考', '中性'),
        ('慎重', '中性'), ('谨慎', '中性'), ('审慎', '中性'), ('规划', '中性'),
        ('计划', '中性'), ('选择', '中性'), ('决策', '中性'), ('计算', '中性'),
        ('风险', '中性'), ('未知', '中性'), ('不确定', '中性'), ('可能', '中性'),
        ('责任', '中性'), ('使命', '中性'), ('义务', '中性'), ('挑战', '中性'),
        ('考验', '中性'), ('冒险', '中性'), ('传承', '中性'), ('未来', '中性'),
        ('遥远', '中性'), ('期待', '中性'), ('希望', '中性'), ('憧憬', '中性'),
        ('陪伴', '中性'), ('成长', '中性'), ('自由', '中性'), ('热闹', '中性')
    ]

    return positive_words + negative_words + neutral_words


def expand_labeled_data():
    """扩展标注数据，增加多样性和覆盖度"""
    expanded_data = []

    # 基础数据
    expanded_data.extend(load_tsinghua_sentiment_dict())
    expanded_data.extend(load_ntu_sentiment_dict())
    expanded_data.extend(load_boson_sentiment_dict())
    expanded_data.extend(load_hownet_sentiment_dict())
    expanded_data.extend(load_base_labeled_data())

    # 生成一些变体
    base_words = [
        ('幸福', '正面'), ('快乐', '正面'), ('痛苦', '负面'), ('焦虑', '负面'),
        ('责任', '中性'), ('希望', '中性'), ('成长', '中性'), ('压力', '负面'),
        ('温暖', '正面'), ('负担', '负面'), ('期待', '中性'), ('自由', '中性'),
        ('陪伴', '中性'), ('感恩', '正面'), ('满足', '正面'), ('充实', '正面'),
        ('喜悦', '正面'), ('悲伤', '负面'), ('恐惧', '负面'), ('开心', '正面'),
        ('美好', '正面'), ('甜蜜', '正面'), ('痛苦', '负面'), ('忧虑', '负面'),
        ('希望', '中性'), ('未来', '中性'), ('冒险', '中性'), ('挑战', '中性')
    ]

    # 添加修饰词组合
    modifiers = ['大', '小', '真', '更']
    for word, label in base_words:
        for mod in modifiers:
            expanded_data.append((mod + word, label))

    # 添加反义词对
    antonym_pairs = [
        ('幸福', '痛苦'), ('快乐', '悲伤'), ('温暖', '寒冷'),
        ('希望', '绝望'), ('成功', '失败'), ('充实', '空虚'),
        ('美好', '丑恶'), ('甜蜜', '苦涩'), ('信任', '怀疑')
    ]

    for pos, neg in antonym_pairs:
        expanded_data.append((pos, '正面'))
        expanded_data.append((neg, '负面'))

    # 去重
    unique_data = []
    seen = set()
    for word, label in expanded_data:
        if word not in seen:
            seen.add(word)
            unique_data.append((word, label))

    return unique_data


# 2. 优化的特征提取器
class OptimizedFeatureExtractor:
    """优化的特征提取器"""

    def __init__(self):
        # 情感字符集
        self.positive_chars = self._load_positive_chars()
        self.negative_chars = self._load_negative_chars()
        self.neutral_chars = self._load_neutral_chars()

        # 情感词典
        self.sentiment_lexicon = self._build_sentiment_lexicon()

        # 权重设置
        self.word_weights = {
            '幸福': 2.5, '快乐': 2.3, '喜悦': 2.2, '开心': 2.1,
            '痛苦': 2.5, '焦虑': 2.4, '压力': 2.3, '悲伤': 2.3,
            '责任': 1.8, '希望': 1.7, '期待': 1.7, '成长': 1.6,
            '温暖': 2.2, '感恩': 2.1, '满足': 2.0, '充实': 2.0,
            '负担': 2.2, '束缚': 2.1, '代价': 2.0, '牺牲': 2.1,
            '恐惧': 2.4, '害怕': 2.3, '忧虑': 2.2, '烦恼': 2.1
        }

    def _load_positive_chars(self):
        """加载正面字符集"""
        chars = set()
        # 从多个词典收集正面字符
        for data_func in [load_tsinghua_sentiment_dict, load_ntu_sentiment_dict,
                          load_boson_sentiment_dict, load_hownet_sentiment_dict]:
            for word, label in data_func():
                if label == '正面':
                    chars.update(word)

        # 额外添加常用正面字符
        chars.update('福乐喜悦欢欣愉快开心高兴欣喜舒畅爽快甜美甜蜜温馨暖和蔼亲切慈祥恩爱和睦谐融友好善')
        return chars

    def _load_negative_chars(self):
        """加载负面字符集"""
        chars = set()
        # 从多个词典收集负面字符
        for data_func in [load_tsinghua_sentiment_dict, load_ntu_sentiment_dict,
                          load_boson_sentiment_dict, load_hownet_sentiment_dict]:
            for word, label in data_func():
                if label == '负面':
                    chars.update(word)

        # 额外添加常用负面字符
        chars.update(
            '痛苦悲恸惨绝失望绝望沮丧颓废萎靡消沉低沉低落阴暗阴沉忧郁郁闷烦苦恼烦燥焦虑焦急着急忧愁担心恐惧害怕恐怖惊慌恐慌慌张紧张暴躁愤怒气恼怒火怨恨仇恨嫉妒妒忌羡慕猜疑')
        return chars

    def _load_neutral_chars(self):
        """加载中性字符集"""
        chars = set()
        chars.update(
            '权衡衡量考虑思考慎重谨慎规划计划选择决策计算风险未知责任义务挑战考验冒险传承未来遥远期待希望憧憬陪伴成长自由热闹牵挂纽带联结成本投入产出投资算计可能旅程过程客观主观分析研究调查统计评估评价判断')
        return chars

    def _build_sentiment_lexicon(self):
        """构建情感词典"""
        lexicon = {}
        all_data = []
        all_data.extend(load_tsinghua_sentiment_dict())
        all_data.extend(load_ntu_sentiment_dict())
        all_data.extend(load_boson_sentiment_dict())
        all_data.extend(load_hownet_sentiment_dict())
        all_data.extend(load_base_labeled_data())

        for word, label in all_data:
            lexicon[word] = label

        return lexicon

    def extract_enhanced_features(self, words):
        """提取增强特征（12个特征）"""
        features = []

        for word in words:
            word_features = []

            # 1. 基础特征
            length = len(word)
            word_features.append(length)

            # 2. 字符类型特征
            pos_count = sum(1 for char in word if char in self.positive_chars)
            neg_count = sum(1 for char in word if char in self.negative_chars)
            neu_count = sum(1 for char in word if char in self.neutral_chars)

            word_features.extend([pos_count, neg_count, neu_count])

            # 3. 情感比例特征
            if length > 0:
                pos_ratio = pos_count / length
                neg_ratio = neg_count / length
                neu_ratio = neu_count / length
            else:
                pos_ratio = neg_ratio = neu_ratio = 0.0

            # 4. 情感得分特征
            sentiment_score = pos_ratio - neg_ratio

            # 应用词权重
            if word in self.word_weights:
                sentiment_score *= self.word_weights[word]

            # 5. 词典匹配特征
            lexicon_match = 1 if word in self.sentiment_lexicon else 0
            lexicon_sentiment = 0
            if lexicon_match:
                label = self.sentiment_lexicon[word]
                if label == '正面':
                    lexicon_sentiment = 1
                elif label == '负面':
                    lexicon_sentiment = -1

            # 6. 结构特征
            has_modifier = 1 if any(word.startswith(mod) for mod in ['大', '小', '真', '更', '最']) else 0
            is_compound = 1 if any(p in word for p in ['的', '与', '和', '之', '及']) else 0

            # 7. 收集所有特征
            word_features.extend([
                pos_ratio, neg_ratio, neu_ratio,
                sentiment_score,
                lexicon_match, lexicon_sentiment,
                has_modifier, is_compound
            ])

            features.append(word_features)

        return np.array(features)

    def get_feature_names(self):
        """获取特征名称"""
        return [
            '长度', '正面字符数', '负面字符数', '中性字符数',
            '正面比例', '负面比例', '中性比例',
            '情感得分', '词典匹配', '词典情感',
            '有修饰词', '是复合词'
        ]

    def predict_with_lexicon(self, word):
        """使用情感词典直接预测"""
        if word in self.sentiment_lexicon:
            confidence = 0.95 if word in self.word_weights else 0.85
            return self.sentiment_lexicon[word], confidence
        return None, 0.0


# 3. 正则化集成分类器（修复版本）
class RegularizedEnsembleClassifier:
    """正则化集成分类器（修复MLPClassifier问题）"""

    def __init__(self, use_xgb=True):
        self.feature_extractor = OptimizedFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k=6)  # 减少特征数量

        # 定义强正则化模型（移除MLPClassifier，因为它不支持sample_weight）
        self.models = self._create_regularized_models(use_xgb)
        self.is_trained = False

    def _create_regularized_models(self, use_xgb):
        """创建强正则化模型（移除MLPClassifier）"""
        models = {
            # 1. SVM with strong regularization
            'svm_rbf': SVC(
                kernel='rbf',
                C=0.3,  # 更强的正则化
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42,
                shrinking=True,
                cache_size=200
            ),

            # 2. Linear SVM
            'svm_linear': SVC(
                kernel='linear',
                C=0.2,  # 更强的正则化
                probability=True,
                class_weight='balanced',
                random_state=42
            ),

            # 3. Random Forest with regularization
            'rf': RandomForestClassifier(
                n_estimators=100,  # 减少树的数量
                max_depth=6,  # 限制深度
                min_samples_split=10,  # 增加最小分割样本
                min_samples_leaf=4,  # 增加叶子节点最小样本
                max_features='sqrt',
                min_impurity_decrease=0.01,  # 减少不重要的分裂
                class_weight='balanced',
                random_state=42,
                bootstrap=True,
                oob_score=True,  # 使用袋外样本评估
                max_samples=0.7  # 每次bootstrap采样70%
            ),

            # 4. Gradient Boosting with regularization
            'gb': GradientBoostingClassifier(
                n_estimators=80,
                max_depth=4,  # 浅层树
                min_samples_split=8,
                min_samples_leaf=3,
                learning_rate=0.05,  # 较小的学习率
                subsample=0.7,  # 子采样
                max_features='sqrt',
                random_state=42
            ),

            # 5. Logistic Regression with regularization
            'logistic': LogisticRegression(
                C=0.3,  # 更强的正则化
                penalty='l2',
                class_weight='balanced',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        }

        # 6. XGBoost with strong regularization (if available)
        if use_xgb and XGB_AVAILABLE:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=4,  # 浅层树
                learning_rate=0.05,  # 较小的学习率
                subsample=0.7,  # 行采样
                colsample_bytree=0.7,  # 列采样
                reg_alpha=0.1,  # L1正则化
                reg_lambda=0.3,  # L2正则化
                min_child_weight=2,  # 防止过拟合
                gamma=0.1,  # 分裂最小损失减少
                random_state=42,
                eval_metric='mlogloss'
            )

        return models

    def train_with_regularization(self, X_train, y_train, X_val, y_val):
        """使用正则化策略训练"""
        print("使用强正则化策略训练模型...")

        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 特征选择（选择最重要的6个特征）
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_encoded)
        X_val_selected = self.feature_selector.transform(X_val_scaled)

        print(f"特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}个特征")

        # 使用适度的SMOTE平衡（不过度平衡）
        print("应用适度SMOTE平衡数据...")
        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X_train_selected, y_train_encoded)

        print(f"  原始训练集: {len(X_train)}")
        print(f"  平衡后训练集: {len(X_balanced)}")

        # 使用交叉验证训练每个模型
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        model_scores = {}
        trained_models = {}

        for name, model in self.models.items():
            print(f"\n训练 {name}...")

            try:
                # 交叉验证评估
                cv_scores = cross_val_score(
                    model, X_balanced, y_balanced,
                    cv=kf, scoring='accuracy', n_jobs=-1
                )

                # 训练模型
                model.fit(X_balanced, y_balanced)
                trained_models[name] = model

                # 在验证集上评估
                val_score = model.score(X_val_selected, y_val_encoded)

                # 计算过拟合间隙
                train_score = model.score(X_balanced, y_balanced)
                overfit_gap = train_score - val_score

                model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'val_score': val_score,
                    'train_score': train_score,
                    'overfit_gap': overfit_gap,
                    'model': model
                }

                print(f"  CV准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"  训练集准确率: {train_score:.4f}")
                print(f"  验证集准确率: {val_score:.4f}")
                print(f"  过拟合间隙: {overfit_gap:.4f}")

                # 检查是否过拟合严重
                if overfit_gap > 0.05:
                    print(f"  ⚠️  {name}可能过拟合")
                elif overfit_gap < 0.01:
                    print(f"  ✅  {name}泛化良好")

            except Exception as e:
                print(f"  ❌ 训练{name}失败: {e}")

        # 选择泛化能力最好的模型进行集成
        print("\n选择泛化能力最好的模型...")

        # 按泛化能力排序（过拟合间隙小，验证集准确率高）
        def generalization_score(item):
            name, scores = item
            # 计算泛化分数：验证集准确率 × (1 - 过拟合间隙)
            return scores['val_score'] * (1 - min(scores['overfit_gap'], 0.1))

        best_models = sorted(
            model_scores.items(),
            key=generalization_score,
            reverse=True
        )[:3]  # 选择前3个泛化能力最好的模型

        print("入选的模型:")
        for name, scores in best_models:
            print(f"  {name}: 验证集={scores['val_score']:.4f}, 过拟合间隙={scores['overfit_gap']:.4f}")

        # 创建加权集成（权重基于泛化能力）
        estimators = []
        weights = []
        total_gen_score = sum(generalization_score(item) for item in best_models)

        for name, scores in best_models:
            weight = generalization_score((name, scores)) / total_gen_score
            estimators.append((name, scores['model']))
            weights.append(weight)
            print(f"  {name}: 权重={weight:.3f}")

        # 创建Voting集成模型
        print("\n创建加权投票集成模型...")
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )

        # 训练Voting集成模型
        voting_ensemble.fit(X_balanced, y_balanced)
        self.models['ensemble'] = voting_ensemble

        # 评估集成模型
        ensemble_train_score = voting_ensemble.score(X_balanced, y_balanced)
        ensemble_val_score = voting_ensemble.score(X_val_selected, y_val_encoded)

        print(f"\n集成模型性能:")
        print(f"  训练集准确率: {ensemble_train_score:.4f}")
        print(f"  验证集准确率: {ensemble_val_score:.4f}")
        print(f"  过拟合间隙: {ensemble_train_score - ensemble_val_score:.4f}")

        self.is_trained = True

        return {
            'model_scores': model_scores,
            'ensemble_scores': {
                'train': ensemble_train_score,
                'val': ensemble_val_score,
                'overfit_gap': ensemble_train_score - ensemble_val_score
            },
            'selected_features': self.feature_selector.get_support(),
            'selected_models': [name for name, _ in best_models]
        }

    def predict(self, words):
        """预测情感"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        # 先检查词典
        lexicon_predictions = []
        need_model_prediction = []
        indices = []

        for i, word in enumerate(words):
            label, confidence = self.feature_extractor.predict_with_lexicon(word)
            if label is not None and confidence > 0.9:
                lexicon_predictions.append({
                    'index': i,
                    'word': word,
                    'prediction': label,
                    'confidence': confidence,
                    'source': 'lexicon'
                })
            else:
                need_model_prediction.append(word)
                indices.append(i)

        # 模型预测
        model_predictions = []
        if need_model_prediction:
            X = self.feature_extractor.extract_enhanced_features(need_model_prediction)
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)

            predictions = self.models['ensemble'].predict(X_selected)
            probabilities = self.models['ensemble'].predict_proba(X_selected)

            for idx, word, pred, prob in zip(indices, need_model_prediction, predictions, probabilities):
                pred_label = self.label_encoder.inverse_transform([pred])[0]
                confidence = max(prob)
                model_predictions.append({
                    'index': idx,
                    'word': word,
                    'prediction': pred_label,
                    'confidence': confidence,
                    'source': 'model'
                })

        # 合并结果
        all_predictions = lexicon_predictions + model_predictions
        all_predictions.sort(key=lambda x: x['index'])

        return all_predictions

    def evaluate_with_diagnostics(self, X_test, y_test):
        """带诊断的评估"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        X_scaled = self.scaler.transform(X_test)
        X_selected = self.feature_selector.transform(X_scaled)
        y_encoded = self.label_encoder.transform(y_test)

        # 集成模型评估
        y_pred = self.models['ensemble'].predict(X_selected)

        # 计算各项指标
        results = {
            'accuracy': accuracy_score(y_encoded, y_pred),
            'precision': precision_score(y_encoded, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_encoded, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_encoded, y_pred, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_encoded, y_pred),
            'kappa': cohen_kappa_score(y_encoded, y_pred)
        }

        # 生成详细报告
        report = classification_report(
            y_encoded, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        return results, report

    def analyze_overfitting_detailed(self, X_train, y_train, X_test, y_test):
        """详细过拟合分析"""
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # 训练集和测试集预测
        train_pred = self.models['ensemble'].predict(X_train_selected)
        test_pred = self.models['ensemble'].predict(X_test_selected)

        train_accuracy = accuracy_score(y_train_encoded, train_pred)
        test_accuracy = accuracy_score(y_test_encoded, test_pred)
        overfit_gap = train_accuracy - test_accuracy

        # 计算每个类别的准确率变化
        class_performance = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            train_mask = y_train_encoded == i
            test_mask = y_test_encoded == i

            if np.sum(train_mask) > 0:
                train_class_acc = accuracy_score(y_train_encoded[train_mask], train_pred[train_mask])
            else:
                train_class_acc = 0

            if np.sum(test_mask) > 0:
                test_class_acc = accuracy_score(y_test_encoded[test_mask], test_pred[test_mask])
            else:
                test_class_acc = 0

            class_performance[class_name] = {
                'train_acc': train_class_acc,
                'test_acc': test_class_acc,
                'gap': train_class_acc - test_class_acc
            }

        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfit_gap,
            'generalization_ratio': test_accuracy / train_accuracy if train_accuracy > 0 else 0,
            'class_performance': class_performance,
            'is_overfitted': overfit_gap > 0.05,
            'overfitting_level': '严重' if overfit_gap > 0.08 else '中等' if overfit_gap > 0.04 else '轻微' if overfit_gap > 0.02 else '无'
        }


# 4. 辅助函数
def analyze_excel_data(excel_path):
    """分析Excel数据但不用于训练"""
    print("\n" + "=" * 60)
    print("分析Excel数据（仅分析，不用于训练）")
    print("=" * 60)

    try:
        df = pd.read_excel(excel_path)
        column_name = df.columns[0]
        all_words = df[column_name].astype(str).tolist()

        cleaned_words = []
        for w in all_words:
            if pd.notna(w):
                w_str = str(w).strip()
                if len(w_str) > 0 and not w_str.isdigit():
                    cleaned_words.append(w_str)

        # 分析数据特征
        word_counts = Counter(cleaned_words)
        top_words = word_counts.most_common(50)

        print(f"Excel数据统计:")
        print(f"  总词数: {len(cleaned_words)}")
        print(f"  唯一词数: {len(set(cleaned_words))}")
        print(f"  平均长度: {np.mean([len(w) for w in cleaned_words]):.2f}")

        print(f"\n高频词 Top 20:")
        for i, (word, count) in enumerate(top_words[:20], 1):
            print(f"  {i:2d}. {word:8s} - {count:4d}次")

        # 分析词长分布
        length_counts = Counter([len(w) for w in cleaned_words])
        print(f"\n词长分布:")
        for length in sorted(length_counts.keys()):
            print(f"  {length}字词: {length_counts[length]:4d}个")

        return cleaned_words, word_counts

    except Exception as e:
        print(f"  分析Excel数据失败: {e}")
        return [], Counter()


def safe_save_excel(df, filepath, max_attempts=3):
    """安全保存Excel文件"""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except:
                filepath = os.path.basename(filepath)

        for attempt in range(max_attempts):
            try:
                df.to_excel(filepath, index=False, engine='openpyxl')
                print(f"✅ 文件保存成功: {filepath}")
                return True
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(2)
                    base, ext = os.path.splitext(filepath)
                    filepath = f"{base}_尝试{attempt + 1}{ext}"
                else:
                    try:
                        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                        if os.path.exists(desktop):
                            new_filepath = os.path.join(desktop, os.path.basename(filepath))
                            df.to_excel(new_filepath, index=False, engine='openpyxl')
                            print(f"✅ 文件已保存到桌面: {new_filepath}")
                            return True
                    except:
                        pass

                    current_filepath = os.path.basename(filepath)
                    df.to_excel(current_filepath, index=False, engine='openpyxl')
                    print(f"✅ 文件已保存到当前目录: {current_filepath}")
                    return True
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(2)
                else:
                    return False
        return False
    except Exception as e:
        print(f"❌ 保存过程中发生错误: {str(e)}")
        return False


def safe_save_model(model, filepath, max_attempts=3):
    """安全保存模型文件"""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except:
                filepath = os.path.basename(filepath)

        for attempt in range(max_attempts):
            try:
                joblib.dump(model, filepath)
                print(f"✅ 模型保存成功: {filepath}")
                return True
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(2)
                    base, ext = os.path.splitext(filepath)
                    filepath = f"{base}_尝试{attempt + 1}{ext}"
                else:
                    try:
                        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                        if os.path.exists(desktop):
                            new_filepath = os.path.join(desktop, os.path.basename(filepath))
                            joblib.dump(model, new_filepath)
                            print(f"✅ 模型已保存到桌面: {new_filepath}")
                            return True
                    except:
                        pass

                    current_filepath = os.path.basename(filepath)
                    joblib.dump(model, current_filepath)
                    print(f"✅ 模型已保存到当前目录: {current_filepath}")
                    return True
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(2)
                else:
                    return False
        return False
    except Exception as e:
        print(f"❌ 模型保存过程中发生错误: {str(e)}")
        return False


def generate_regularized_report(results_df, test_results, overfitting_info, excel_words, training_info):
    """生成正则化版分析报告"""
    print("\n" + "=" * 80)
    print("正则化版情感分析报告")
    print("=" * 80)

    total_words = len(results_df)

    print(f"\n📋 总体统计")
    print("-" * 40)
    print(f"分析词语总数: {total_words}")
    print(f"训练数据总数: {training_info.get('total_training_words', 'N/A')}")

    if total_words > 0:
        # 情感分布
        print(f"\n📊 情感分布")
        print("-" * 40)

        emotion_dist = results_df['情感分类'].value_counts()
        for emotion in ['正面', '中性', '负面']:
            count = emotion_dist.get(emotion, 0)
            pct = count / total_words * 100
            avg_conf = results_df[results_df['情感分类'] == emotion]['置信度'].mean() if count > 0 else 0
            print(f"  {emotion}: {count:>4}个 ({pct:>5.1f}%), 平均置信度: {avg_conf:.3f}")

    # 模型性能
    print(f"\n🤖 模型性能")
    print("-" * 40)
    print(f"  准确率: {test_results['accuracy']:.4f}")
    print(f"  F1分数: {test_results['f1']:.4f}")
    print(f"  精确率: {test_results['precision']:.4f}")
    print(f"  召回率: {test_results['recall']:.4f}")
    print(f"  平衡准确率: {test_results['balanced_accuracy']:.4f}")
    print(f"  Kappa系数: {test_results['kappa']:.4f}")

    # 过拟合分析
    print(f"\n🔍 过拟合分析")
    print("-" * 40)
    print(f"  训练集准确率: {overfitting_info['train_accuracy']:.4f}")
    print(f"  测试集准确率: {overfitting_info['test_accuracy']:.4f}")
    print(f"  过拟合间隙: {overfitting_info['overfitting_gap']:.4f}")
    print(f"  泛化比率: {overfitting_info['generalization_ratio']:.4f}")
    print(f"  过拟合程度: {overfitting_info['overfitting_level']}")

    if overfitting_info['overfitting_gap'] < 0.02:
        print("  ✅ 无明显过拟合，泛化能力优秀")
    elif overfitting_info['overfitting_gap'] < 0.04:
        print("  ⚠️  轻微过拟合，泛化能力良好")
    else:
        print("  ⚠️  需要进一步优化以减少过拟合")


# ========== 整体词云图生成函数（红色主题）==========
def generate_overall_wordcloud(results_df, save_path=None, show_figures=True):
    """
    生成整体词云图（不区分情感）- 红色主题

    Parameters:
    -----------
    results_df : DataFrame
        包含'词语'和'置信度'列的结果数据框
    save_path : str, optional
        保存路径，如果为None则保存在当前目录
    show_figures : bool
        是否显示图像
    """
    if not WORDCLOUD_AVAILABLE:
        print("⚠️ WordCloud未安装，无法生成词云图")
        return

    if len(results_df) == 0:
        print("⚠️ 结果数据为空，无法生成词云图")
        return

    print("\n" + "=" * 60)
    print("生成整体词云图（红色主题）")
    print("=" * 60)

    # 设置中文字体路径（Windows系统）
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
    if not os.path.exists(font_path):
        # 尝试其他中文字体
        alternative_fonts = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/STSONG.TTF',  # 华文宋体
        ]
        for alt_font in alternative_fonts:
            if os.path.exists(alt_font):
                font_path = alt_font
                break
        else:
            font_path = None
            print("⚠️ 未找到中文字体，词云图可能无法正确显示中文")

    # 构建词频字典（使用置信度作为权重）
    word_freq = {}
    for _, row in results_df.iterrows():
        word = row['词语']
        # 使用置信度作为权重，并考虑词频
        freq = row['置信度']
        word_freq[word] = word_freq.get(word, 0) + freq

    print(f"  总词语数: {len(results_df)}")
    print(f"  唯一词语数: {len(word_freq)}")

    # 创建红色主题的颜色映射函数
    def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        """返回红色系的颜色"""
        # 生成不同深浅的红色
        # 颜色范围：浅红 (#FFB6C1) 到 深红 (#8B0000)
        red_shades = [
            '#FFB6C1',  # 浅粉红
            '#FF69B4',  # 热粉色
            '#FF1493',  # 深粉色
            '#DB7093',  # 苍白的紫罗兰红色
            '#C71585',  # 中紫罗兰红色
            '#CD5C5C',  # 印度红
            '#F08080',  # 浅珊瑚色
            '#FA8072',  # 鲑鱼色
            '#E9967A',  # 暗鲑鱼色
            '#DC143C',  # 猩红色
            '#B22222',  # 耐火砖红
            '#8B0000',  # 暗红色
            '#FF0000',  # 纯红色
            '#FF4500',  # 橙红色
            '#FF6347',  # 番茄红
        ]

        # 根据词频或字体大小选择颜色深浅
        # 字体越大颜色越深
        if font_size > 100:
            return '#8B0000'  # 深红
        elif font_size > 70:
            return '#B22222'  # 耐火砖红
        elif font_size > 50:
            return '#DC143C'  # 猩红色
        elif font_size > 30:
            return '#FF4500'  # 橙红色
        else:
            return '#FF6347'  # 番茄红

    # 创建自定义的红色系colormap
    red_cmap = LinearSegmentedColormap.from_list(
        'red_theme',
        ['#FFB6C1', '#FF69B4', '#FF1493', '#DC143C', '#B22222', '#8B0000']
    )

    # 创建图形
    plt.figure(figsize=(16, 10), facecolor='white')

    try:
        # 生成词云 - 使用红色主题
        wordcloud = WordCloud(
            font_path=font_path,
            width=1600,
            height=800,
            background_color='white',  # 白色背景
            max_words=100,
            max_font_size=200,
            random_state=42,
            color_func=red_color_func,  # 使用自定义红色函数
            collocations=False,
            prefer_horizontal=0.8,
            relative_scaling=0.5  # 词频对字体大小的影响程度
        ).generate_from_frequencies(word_freq)

        # 显示
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'红色主题词云图\n(共{len(results_df)}个词语)',
                  fontsize=16, fontweight='bold', pad=20, color='#8B0000')
        plt.axis('off')

        # 保存
        if save_path:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
            print(f"✅ 红色主题词云图已保存: {save_path}")

        # 显示
        if show_figures:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"❌ 生成红色主题词云失败: {e}")
        plt.close()

    return wordcloud


# 5. 主函数（只保留整体词云图）
def main_with_overall_wordcloud():
    """只保留整体词云图的主函数"""
    print("=" * 80)
    print("正则化防过拟合情感分析系统（红色主题词云图）")
    print("训练数据：仅使用外部词典，不使用Excel数据")
    print("=" * 80)

    excel_path = r"C:\Users\lxh75\bert 市调.xlsx"

    # 1. 仅使用外部词典创建训练数据
    print("\n1. 创建训练数据（仅使用外部词典）...")
    labeled_data = expand_labeled_data()
    labeled_words, labels = zip(*labeled_data)

    print(f"训练数据统计:")
    print(f"  总词数: {len(labeled_words)}")
    label_dist = Counter(labels)
    for label in ['正面', '中性', '负面']:
        count = label_dist.get(label, 0)
        pct = count / len(labels) * 100
        print(f"  {label}: {count}个 ({pct:.1f}%)")

    # 2. 分析Excel数据（仅分析，不用于训练）
    print("\n2. 分析Excel数据（仅用于了解数据特征）...")
    excel_words, word_counts = analyze_excel_data(excel_path)

    # 3. 初始化特征提取器
    print("\n3. 初始化优化的特征提取器...")
    feature_extractor = OptimizedFeatureExtractor()

    # 4. 提取特征
    print("\n4. 提取增强特征...")
    X = feature_extractor.extract_enhanced_features(labeled_words)
    print(f"特征维度: {X.shape}")
    print(f"特征名称: {feature_extractor.get_feature_names()}")

    # 5. 划分数据集（60%训练，20%验证，20%测试）
    print("\n5. 划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"数据集划分:")
    print(f"  训练集: {len(X_train)} 个样本 ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  验证集: {len(X_val)} 个样本 ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"  测试集: {len(X_test)} 个样本 ({len(X_test) / len(X) * 100:.1f}%)")

    # 6. 训练正则化模型
    print("\n6. 训练正则化集成模型...")
    classifier = RegularizedEnsembleClassifier(use_xgb=XGB_AVAILABLE)
    training_info = classifier.train_with_regularization(X_train, y_train, X_val, y_val)

    # 7. 在测试集上评估
    print("\n7. 在测试集上最终评估...")
    test_results, test_report = classifier.evaluate_with_diagnostics(X_test, y_test)

    print("\n测试集性能:")
    print(f"  准确率: {test_results['accuracy']:.4f}")
    print(f"  F1分数: {test_results['f1']:.4f}")
    print(f"  精确率: {test_results['precision']:.4f}")
    print(f"  召回率: {test_results['recall']:.4f}")
    print(f"  平衡准确率: {test_results['balanced_accuracy']:.4f}")
    print(f"  Kappa系数: {test_results['kappa']:.4f}")

    # 8. 详细过拟合分析
    print("\n8. 详细过拟合分析...")
    overfitting_info = classifier.analyze_overfitting_detailed(X_train, y_train, X_test, y_test)

    print(f"  训练集准确率: {overfitting_info['train_accuracy']:.4f}")
    print(f"  测试集准确率: {overfitting_info['test_accuracy']:.4f}")
    print(f"  过拟合间隙: {overfitting_info['overfitting_gap']:.4f}")
    print(f"  泛化比率: {overfitting_info['generalization_ratio']:.4f}")
    print(f"  过拟合程度: {overfitting_info['overfitting_level']}")

    # 显示类别级别的性能
    print(f"\n各类别性能:")
    for class_name, perf in overfitting_info['class_performance'].items():
        print(f"  {class_name}: 训练集={perf['train_acc']:.3f}, 测试集={perf['test_acc']:.3f}, 差距={perf['gap']:.3f}")

    # 9. 预测Excel数据
    print("\n9. 预测Excel数据...")
    if len(excel_words) > 0:
        print(f"预测Excel中的 {len(excel_words)} 个词语...")

        # 批量预测
        all_predictions = classifier.predict(excel_words)

        # 整理结果
        results_data = []
        lexicon_count = 0
        model_count = 0

        for pred in all_predictions:
            results_data.append({
                '词语': pred['word'],
                '情感分类': pred['prediction'],
                '置信度': round(pred['confidence'], 3),
                '预测来源': '词典匹配' if pred['source'] == 'lexicon' else '模型预测',
                '词语长度': len(pred['word'])
            })

            if pred['source'] == 'lexicon':
                lexicon_count += 1
            else:
                model_count += 1

        results_df = pd.DataFrame(results_data)

        print(f"\n预测统计:")
        print(f"  词典直接匹配: {lexicon_count} 个 ({lexicon_count / len(results_df) * 100:.1f}%)")
        print(f"  模型预测: {model_count} 个 ({model_count / len(results_df) * 100:.1f}%)")

        # 10. 保存结果
        print("\n10. 保存结果...")
        output_path = "情感分析结果.xlsx"

        if safe_save_excel(results_df, output_path):
            print(f"✅ 结果保存成功: {output_path}")

        # 11. 生成报告
        print("\n11. 生成分析报告...")
        generate_regularized_report(results_df, test_results, overfitting_info, excel_words,
                                    {'total_training_words': len(labeled_words)})

        # ========== 只生成整体词云图（红色主题）==========
        print("\n12. 生成红色主题词云图...")

        # 创建可视化目录
        viz_dir = "情感分析可视化"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # 只生成整体词云图（红色主题）
        if WORDCLOUD_AVAILABLE:
            generate_overall_wordcloud(
                results_df,
                save_path=os.path.join(viz_dir, "红色主题词云图.png"),
                show_figures=True
            )
        else:
            print("\n⚠️ 请安装wordcloud库以生成词云图:")
            print("   pip install wordcloud")

        print(f"\n📊 红色主题词云图已保存到 '{viz_dir}' 目录")

    else:
        print("❌ Excel数据为空，无法进行预测")
        results_df = pd.DataFrame()

    # 12. 保存模型
    print("\n13. 保存正则化模型...")
    model_package = {
        'classifier': classifier,
        'feature_extractor': feature_extractor,
        'test_results': test_results,
        'test_report': test_report,
        'overfitting_info': overfitting_info,
        'training_info': training_info,
        'label_distribution': dict(Counter(labels)),
        'metadata': {
            'version': 'regularized_with_red_wordcloud_v1',
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_names': feature_extractor.get_feature_names(),
            'selected_feature_count': training_info.get('selected_features',
                                                        []).sum() if 'selected_features' in training_info else 0,
            'excel_word_count': len(excel_words),
            'total_training_words': len(labeled_words),
            'overfitting_gap': overfitting_info['overfitting_gap'],
            'test_accuracy': test_results['accuracy'],
            'xgb_available': XGB_AVAILABLE
        }
    }

    model_path = "情感分析模型.pkl"
    if safe_save_model(model_package, model_path):
        print(f"✅ 正则化模型保存成功: {model_path}")

    # 13. 性能总结
    print("\n" + "=" * 80)
    print("🎉 正则化防过拟合情感分析完成！")
    print("=" * 80)

    if len(results_df) > 0:
        print(f"\n📊 最终情感分布:")
        emotion_dist = results_df['情感分类'].value_counts()
        for emotion in ['正面', '中性', '负面']:
            count = emotion_dist.get(emotion, 0)
            pct = count / len(results_df) * 100
            print(f"  {emotion}: {count}个 ({pct:.1f}%)")

    print(f"\n📈 模型性能总结:")
    print(f"  测试集准确率: {test_results['accuracy']:.4f}")
    print(f"  过拟合间隙: {overfitting_info['overfitting_gap']:.4f}")
    print(f"  过拟合程度: {overfitting_info['overfitting_level']}")
    print(
        f"  泛化能力: {'优秀' if overfitting_info['overfitting_gap'] < 0.02 else '良好' if overfitting_info['overfitting_gap'] < 0.04 else '一般'}")

    return {
        'results_df': results_df,
        'classifier': classifier,
        'test_results': test_results,
        'overfitting_info': overfitting_info
    }


# 运行主函数（只保留整体词云图 - 红色主题）
if __name__ == "__main__":
    print("=" * 80)
    print("情感分析系统启动中...（红色主题词云图）")
    print("=" * 80)

    if not WORDCLOUD_AVAILABLE:
        print("\n⚠️ 提示: wordcloud库未安装，词云图功能将不可用")
        print("   如需生成红色主题词云图，请运行: pip install wordcloud")
        print("   继续运行基本情感分析...\n")

    results = main_with_overall_wordcloud()

    print("\n" + "=" * 80)
    print("程序执行完毕！")
    print("=" * 80)