from django.http import HttpResponse
from django.shortcuts import render
from .Process_2 import Crawer_new
from .AddLine import EmotionAdder
from .printtable_new import PicGenerate
import time
import csv
import concurrent.futures
import os
def hello(request):
    return HttpResponse("Hello World!")

def SearchPage(request):
    return render(request, 'SearchPage.html')

async def ResultPage(request):
    if request.method == "POST":
        search_query = request.POST.get('search_query', '')  # 获取文本框内容
        options = request.POST.getlist('options')           # 获取勾选列表内容
        time_map={"全部":0,"一个月前":30,"三个月前":90,"半年前":183,"一年前":365}
        day_before = time_map[options[0]]
        #以下是需要更改的部分
        # await(Crawer_new(search_query,DAY_BEFORE=day_before))
        print("next...")
        # EmotionAdder(search_query)

        result_path = "./static/csvs/"+search_query+"_result.csv"
        picture_path = "./static/images/"+search_query+"_result.png"
        gender_path = "./static/images/"+search_query+"_result2.png"
        selected_emotions = set(options[1:])
        # print(selected_emotions)

        PicGenerate(result_path,selected_emotions,picture_path,gender_path)
        
        Picture_Name = search_query+"_result.png"
        Gender_Name = search_query+"_result2.png"
        Cloud_Name = search_query+"_result3.png"
        Result_File_Path = os.getcwd()+"/static/csvs/"+search_query+"_result.csv"


    else:
        search_query = request.GET.get('search_query', '')  # 如果是 GET 请求，获取查询参数
        options = request.GET.getlist('options')

    return render(request, 'ResultPage.html' ,{
        'search_query': search_query,
        'options': options,
        'Picture_Name': Picture_Name, #如果有多余的返回值就这样返回
        'Gender_Name': Gender_Name,
        'Result_File':Result_File_Path,
    })