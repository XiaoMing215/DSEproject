import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rcParams

# 定义情绪中文到英文的映射
emotion_map = {
    '厌恶/讨厌': '厌恶',
    '平静/冷静': '冷静',
    '恐惧/害怕': '恐惧',
    '愤怒/生气': '愤怒',
    '悲伤/痛苦': '悲伤',
    '愉快/幸福': '愉快',
    '羞耻': '羞耻',
    '希望/期待': '希望',
    '惊讶/震惊': '惊讶',
    '孤独/寂寞': '孤独',
    '好奇/兴趣': '好奇'
}

input_to_actual_emotion_map = {
    '愉快': '愉快',
    '愤怒': '愤怒',
    '悲伤': '悲伤',
    '冷静': '冷静',
    '恐惧': '恐惧',
    '惊讶': '惊讶',
    '羞耻': '羞耻',
    '希望': '希望',
    '孤独': '孤独',
    '好奇': '好奇',
    '厌恶': '厌恶',
    '正向': 'Positive',
    '负向': 'Negative'
}

# 找到支持中文的字体路径
font_path = next((f.fname for f in font_manager.fontManager.ttflist if 'SimHei' in f.name), None)
if font_path is None:
    raise FileNotFoundError("找不到支持中文的字体 SimHei")
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.sans-serif'] = font_prop.get_name()


def PicGenerate(file_path, selected_emotions, save_path, save_path_gender):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 映射情绪列
    df['Emotion_Chinese'] = df['emotions'].map(emotion_map)

    # 只保留 selected_emotions 中的情绪数据
    df_filtered = df[df['Emotion_Chinese'].isin(selected_emotions)]

    # 统计 Positive 和 Negative 标签的总次数，且只对 selected_emotions 进行筛选
    positive_count = df_filtered[df_filtered['tags'] == 1].shape[0]
    negative_count = df_filtered[df_filtered['tags'] == 0].shape[0]

    # 将 Positive 和 Negative 计数加入到情绪计数
    emotion_counts = df_filtered['Emotion_Chinese'].value_counts()
    emotion_counts['正向'] = positive_count
    emotion_counts['负向'] = negative_count

    # 将性别编码替换为文字标签
    df_filtered['性别'] = df_filtered['性别'].replace({0: '保密', 1: '男性', 2: '女性'})

    # 统计性别分布
    gender_counts = df_filtered['性别'].value_counts()

    # 绘制性别分布柱状图
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='性别', data=df_filtered, palette="Set2")

    # 在性别柱状图上添加数字标签
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',  # 显示整数值
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # 设置文本位置
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')  # 偏移位置

    # 设置性别分布图标题和标签
    plt.title('性别分布', fontsize=16)
    plt.xlabel('性别', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # 保存性别分布图
    if save_path_gender:
        plt.savefig(f"{save_path_gender}", dpi=300)
    else:
        plt.show()

    plt.close()  # 关闭当前图表以释放内存

    def plot_emotions(selected_emotions=None, save_path=None):
        # 选择要显示的情绪
        emotions_to_display = emotion_counts[emotion_counts.index.isin(selected_emotions)]

        # 绘制情绪分布柱状图
        ax = emotions_to_display.plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black')

        # 在情绪柱状图上添加数字标签
        for idx, value in enumerate(emotions_to_display):
            ax.text(idx, value + 0.5, str(value), ha='center', fontsize=10)

        # 设置图表标题和标签
        plt.title('情绪和标签频率分布', fontsize=14)
        plt.xlabel('情绪/标签', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)

        # 显示图表
        plt.tight_layout()

        # 如果提供了 save_path，则保存图片
        if save_path:
            plt.savefig(save_path, dpi=300)

        # plt.show()

    plot_emotions(selected_emotions, save_path)


# 示例：展示输入的情绪并保存图片
if __name__ == '__main__':
    PicGenerate(
        file_path="特朗普_result.csv",
        selected_emotions=['正向', '负向', '愉快', '愤怒', '悲伤', '冷静', '恐惧', '惊讶', '羞耻', '希望', '孤独', '好奇', '厌恶'],
        save_path='emotion_frequency.png',
        save_path_gender='gender_distribution.png'
    )
