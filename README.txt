天池python评测程序说明

# 原理介绍
赛事举办方根据demo，编写评测程序，并重新打包为zip，提交给天池运营方。

## 输入和输出参数
当赛事举办时，选手提交了结果文件，天池平台的评测服务，会这样触发调用：
sh py_entrance.sh input_param.json eval_result.json

第一个参数名字固定，叫input_param.json，里面的内容是动态生成的，主要是传入了标准答案的路径，和选手提交的结果文件的路径，内容示例如下（该文件请勿自行增加字段）：

{
  "fileData":{
    "evaluatorDir":"",
    "evaluatorPath":"",
    "standardFileDir":"",
    "standardFilePath":"评测答案文件路径，比如answer.zip",
    "userFileDir":"",
    "userFilePath":"需要被评测的文件路径，比如submit.zip"
  }
}
第二个参数名字也是固定，表示评测程序应该把结果写入这个文件。

但实际上评测程序（evaluate.py）不需要关注具体的名称，而直接取值即可：

input_file  = open(sys.argv[1])
input_param = json.load(input_file)
standard_file = input_param['fileData']['standardFilePath'] # 答案文件路径
user_file     = input_param['fileData']['userFilePath']     # 用户提交文件路径

同样，针对结果文件：

output_file = open(sys.argv[2], 'w')

## 输出结果的规范
评测成功的输出：

{
  "score": 1.0, # 这个score是必须的，请勿删除并改为其他名称
  "scoreJson": {
    "score": 1.0 # 这里的key也不一定是score，取决于大赛的配置
  },
  "success": true
}

评测错误的输出：

{
  "errorDetail": "user input is wrong, please check !",
  "errorMsg": "user input is wrong, please check !",
  "score": 0,
  "scoreJson": {
  },
  "success": false
}


# 编写一个评测的步骤
注意评测系统的python版本为3.6，所以需要使用python 3的语法格式。

修改evaluate.py，完善评测逻辑，以及错误时的错误描述逻辑；
本地测试评测程序，具体方法见后面说明；
重新打包，比如zip evaluate.zip evaluate.py py_entrance.sh
将打包文件提交给天池同学

# 评测程序注意事项
1. 需要解压答案和选手文件的情况
目前评测程序在容器运行的逻辑是，下载评测代码、标准答案、选手答案，为root用户运行；而运行评测代码时，是emotion这个用户。所以解压的时候，无法解压到当前目录，建议解压到/home/emotion目录；同时针对选手答案的解压，如果发现该目录已经存在，一定要先删除并再解压！示例代码如下：

import zipfile
import os
import logging
import shutil
......

# standard_file 代表标准答案的路径
if os.path.isdir('/home/emotion/standard') and len(os.listdir('/home/emotion/standard')) > 0:
    logging.info("no need to unzip %s", standard_file)
else:
    with zipfile.ZipFile(standard_file, "r") as zip_ref:
        zip_ref.extractall("/home/emotion/standard")
        zip_ref.close()

# submit_file 表示选手提交的文件路径
if os.path.isdir("/home/emotion/submit"):
    shutil.rmtree("/home/emotion/submit")
with zipfile.ZipFile(submit_file, "r") as zip_data:
    zip_data.extractall("/home/emotion/submit")
    zip_data.close()

## 评测程序的本地测试
1、在刚才的目录，创建一个文件input_param.json，内容如下：
{
  "fileData":{
    "evaluatorDir":"",
    "evaluatorPath":"",
    "standardFileDir":"",
    "standardFilePath":"standard.txt",
    "userFileDir":"",
    "userFilePath":"submit.txt"
  }
}
注意：这里的standard.txt、submit.txt只是示例，分别代表答案文件和用户提交的结果文件。

2、新增两个刚才input_param.json里面提到的文件，比如standard.txt、submit.txt。
3、运行命令：
python3 evaluate.py input_param.json eval_result.json
观察是否运行正常，查看eval_result.json是否产生了结果；并且出错情况下，eval_result.json里的内容是否是约定的错误描述结果。

