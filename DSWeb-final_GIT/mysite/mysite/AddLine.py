from emotion_catcher_CPU.emotion_catcher.model.CNN_for_batch import final_CNN
from emotion_catcher_CPU.emotion_catcher.model.CNN_with_emotion_for_batch import final_CNN_with_emotion
import os


def EmotionAdder(name):
    #
    original_direct = os.getcwd()
    new_direct = os.getcwd()+"\\emotion_catcher_CPU\\emotion_catcher\\model"
    try:
        # 更改当前工作目录
        os.chdir(new_direct)
        print(f"当前工作目录已更改为: {os.getcwd()}")
    except OSError as error:
        print(f"无法更改工作目录: {error}")

    csv_path = "../test/"+name+"_raw.csv"
    temp_path = "../test/"+name+"_tmp.csv"
    save_path = "../../../static/csvs/"+name+"_result.csv"
    print("开始情感解析...")
    final_CNN(csv_path=csv_path, tag="评论内容", save_path=temp_path)
    final_CNN_with_emotion(csv_path=temp_path, tag="评论内容", save_path=save_path)
    print("情感解析完毕")
    os.chdir(original_direct)
    

