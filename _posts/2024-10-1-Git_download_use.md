---

title: Git安装及Github使用
tags: Tools
---

参考链接：[Git安装](https://blog.csdn.net/mukes/article/details/115693833)

参考视频：[Github Pages搭建个人博客](https://www.bilibili.com/video/BV1Xh411b7wh)

### 一、简介

最近在学习AI方向的知识，需要对理论知识和代码进行整理，故想要制作一个个人Blog来记录自己的学习过程，Git和Github自然是一个很好的选择，笔者记录下自己的安装上传过程，希望对其他人也有所帮助。

<!--more-->

### 二、Git的安装

前往Git的官方链接[Git官网](https://git-scm.com/)，在首页点击Dowmload for Windows。

![](/images/gitdown/one.png)

选择对应版本下载即可，笔者选的是Git for Windows/x64 Setup.，版本为2.50.1。

![](/images/gitdown/two.png)

下载后，进行软件安装，一直点击Next，选择默认选项即可。

### 三、Github的使用
点击进入[Github官网](https://github.com/)，注册一个账号即可。（由于网络波动。可能需要翻墙）

### 四、Git的使用
安装完Git后，鼠标右键桌面，找到open git bash here，打开git的命令窗口

接着，使用Git进行全局配置

首先配置名字
```
git config --global user.name "这里输入github注册时的用户名"
```

接着是邮箱
```
git config --global user.email "这里输入github注册时的邮箱"
```

设置密钥，由于笔者电脑的系统用户文件名包含中文，直接配置会出现乱码，这里进行强制配置
1.强制创建文件夹ssh
```
mkdir -p "$HOME/.ssh"
```
2.设置正确权限
```
chmod 700 "$HOME/.ssh"
```
3.找到C：/user/用户名/.ssh，在文件里面创建一个新的文本文件.txt，将该文本文件命名为id_ed25519，并且删除后缀文件类型.txt
4.生成新的SSH密钥
```
ssh-keygen -t ed25519 -C "这里输入github注册时的邮箱" -f "$HOME/.ssh/id_ed25519""
```
按enter至结束，如果success则成功

5.添加密钥到SSH代理

```
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/id_ed25519"
```
6.将公钥添加到 GitHub
```
cat "$HOME/.ssh/id_ed25519.pub"
```
登录 GitHub → Settings → SSH and GPG Keys → New SSH key

将内容粘贴到Key，title可以随表写，最后Add SSH Key即可

7.检查是否成功

```
ssh -T git@github.com
```
successfully出现则成功

### 五、Git上传代码
1.在github页面，点击Create repository，在Repository name写上名字，格式为：
~~~
github用户名.github.io
~~~
之后直接创建就行了

2.随便在电脑一个位置新建一个文件夹，例如D：/newdir，打开这个文件夹，鼠标右键选择open git bash here，打开git的命令窗口，输入
~~~
git init
~~~
初始化git

3.创建一个任意文件，例如text.py
~~~
git add text.py
git commit -m "上传内容提示"
git remote add origin 远程仓库地址
git push
~~~
选择SSH地址

第一行为添加文件，如果有很多，可以输入

~~~
git add .
~~~
第二行为输入提示，说明上传的是什么文件

第三行是连接远程仓库，只在第一次连接时需要

第四行上传

连接仓库后，之后每次更新只需要输入一、二、四行即可。

如果需要制作github pages，可以参考开头的视频。
