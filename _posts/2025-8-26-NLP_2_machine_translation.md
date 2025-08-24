---
title: 自然语言处理（2）机器翻译案例：英—法
tags: DL NLP
typora-root-url: ./..
---

实现自然语言处理中的机器翻译案例。

<!--more-->

##### 1.机器翻译简介

机器翻译（machine translation）指的是将序列从⼀种语⾔⾃动翻译成另⼀种语⾔。机器翻译有两大流派，一个是统计机器翻译（statistical machine translation），另一个是神经机器翻译（neural machine translation），本篇及之后都以神经机器翻译为主。

##### 2.下载和预处理数据集

###### 2.1 数据集下载

数据集下载网址：http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip

下载后，用zipfile库函数导入压缩包里面的数据。

~~~
# 获取数据
import zipfile

# 压缩包路径（根据实际文件位置修改）
zip_path = "fra-eng.zip"
# 压缩包内目标文件的路径（注意目录结构是否正确）
file_in_zip = "fra-eng/fra.txt"

def read_data(zip_path, file_in_zip):
    # 打开压缩包并读取文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 检查目标文件是否存在于压缩包中
        if file_in_zip in zip_ref.namelist():
            # 读取文件内容（按utf-8编码解码）
            with zip_ref.open(file_in_zip, 'r') as file:
                return file.read().decode('utf-8')
        else:
            print(f"错误：压缩包中未找到 {file_in_zip} 文件")

raw_text = read_data(zip_path, file_in_zip)

# 打印前100个字符查看内容
print("文件内容预览：")
print(raw_text[:100])
~~~

###### 2.2 数据预处理

~~~
# 数据预处理
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使⽤空格替换不间断空格
    # 使⽤⼩写字⺟替换⼤写字⺟
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插⼊空格，防止计算机把单词+标点识别成一个整体
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)
    
text = preprocess_nmt(raw_text)
print(text[:80])
~~~

##### 3.词元化

