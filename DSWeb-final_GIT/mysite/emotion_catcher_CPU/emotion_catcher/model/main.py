from CNN_for_batch import final_CNN
from CNN_with_emotion_for_batch import final_CNN_with_emotion
import os

new_direct = os.getcwd()+"\\mysite\\emotion_catcher_CPU\\emotion_catcher\\model"
try:
    # 更改当前工作目录
    os.chdir(new_direct)
    print(f"当前工作目录已更改为: {os.getcwd()}")
except OSError as error:
    print(f"无法更改工作目录: {error}")

# print(os.getcwd())#测试

#
csv_path = "../test/筛选数据_“三权分立”变“三权合一”，特朗普2.0时代前瞻.csv"
temp_path = "../test/temp.csv"
save_path = "../test/result.csv"

final_CNN(csv_path=csv_path, tag="评论内容", save_path=temp_path)
final_CNN_with_emotion(csv_path=temp_path, tag="评论内容", save_path=save_path)
