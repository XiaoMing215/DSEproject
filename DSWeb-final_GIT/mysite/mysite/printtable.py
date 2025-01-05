import pandas as pd
import matplotlib.pyplot as plt
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


def PicGenerate(file_path, selected_emotions, save_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 映射情绪列
    df['Emotion_Chinese'] = df['emotions'].map(emotion_map)


    # 统计每种情绪出现的次数
    emotion_counts = df['Emotion_Chinese'].value_counts()

    # 统计 Positive 和 Negative 标签的总次数
    positive_count = df[df['tags'] == 1].shape[0]
    negative_count = df[df['tags'] == 0].shape[0]

    # 将 Positive 和 Negative 计数加入到情绪计数
    emotion_counts['正向'] = positive_count
    emotion_counts['负向'] = negative_count

    def plot_emotions(selected_emotions=None, save_path=None):
        # 选择要显示的情绪
        emotions_to_display = emotion_counts[emotion_counts.index.isin(selected_emotions)]

        # 绘制柱状图
        ax = emotions_to_display.plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black')

        # 添加数字标签
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
        save_path='emotion_frequency.png'
    )
