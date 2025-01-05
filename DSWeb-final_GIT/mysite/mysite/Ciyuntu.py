import pandas as pd
import jieba 
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def GenerateWordCloud(csv_file, cloud_path,stopwords_file= 'stopwords_full.txt'):

    # 读取停用词文件，并将其存储在一个集合中
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().strip().split('\n'))  # 停用词列表，按行分隔

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    comments = df['评论内容'].dropna()  

    def filter_stopwords_and_single_char_words(text):
        words = jieba.cut(text)
        return " ".join([word for word in words if len(word) > 1 and word not in stopwords]) 

    # 对评论内容进行处理
    comments_text = " ".join(comments.apply(lambda x: filter_stopwords_and_single_char_words(str(x))))

    wordcloud = WordCloud(
        font_path='simhei.ttf', 
        width=1000,  
        height=600,  
        background_color='white', 
        max_words=150,  
        max_font_size=100,  # 最大字体大小
        random_state=42,  # 随机种子，保证每次生成相同
        contour_color='black',  # 边界颜色
        contour_width=2  
    ).generate(comments_text)

    # 显示词云图
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off') 

    # 自动保存为PNG图片
    wordcloud.to_file(cloud_path)

    # plt.show()
