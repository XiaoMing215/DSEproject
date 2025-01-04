from django.http import HttpResponse
from django.shortcuts import render
from .Process_2 import Crawer_new
from .AddLine import EmotionAdder
import time
import csv
import concurrent.futures
def hello(request):
    return HttpResponse("Hello World!")

def SearchPage(request):
    return render(request, 'SearchPage.html')

async def ResultPage(request):
    if request.method == "POST":
        search_query = request.POST.get('search_query', '')  # 获取文本框内容
        options = request.POST.getlist('options')           # 获取勾选列表内容
        #以下是需要更改的部分
        await(Crawer_new(search_query))
        print("next...")
        otherdata = "D:\\MyCode\\VScode\\DSWeb-final\\mysite\\emotion_catcher_CPU\\emotion_catcher\\test\\"+search_query+"_result.csv"
        EmotionAdder(search_query)
    else:
        search_query = request.GET.get('search_query', '')  # 如果是 GET 请求，获取查询参数
        options = request.GET.getlist('options')
    return render(request, 'ResultPage.html' ,{
        'search_query': search_query,
        'options': options,
        'other_data': otherdata #如果有多余的返回值就这样返回
    })