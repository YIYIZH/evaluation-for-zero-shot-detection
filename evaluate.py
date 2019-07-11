# coding=utf-8

import json
import sys
import os
import numpy as np
import time
from eval import *

# 错误字典，这里只是示例
error_msg={
    1: "Bad input file",
    2: "Wrong input file format",
}

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)

def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict, out_p)

def report_score(score, out_p):
    result = dict()
    result['success']=True
    result['score'] = score
    result['scoreJson'] = {'score': score} # 这里{}里面的score也可以是其他的key，取决于大赛配置
    dump_2_json(result, out_p)

if __name__=="__main__":
    '''
      online evaluation
      
    '''
    in_param_path = "input_param.json" #sys.argv[1]
    out_path = "eval_result.json" #sys.argv[2]

    # read submit and answer file from first parameter
    with open(in_param_path, 'r') as load_f:
        input_params = json.load(load_f)

    # 标准答案路径
    standard_path=input_params["fileData"]["standardFilePath"]
    print("Read standard from %s" % standard_path)

    # 选手提交的结果文件路径
    submit_path=input_params["fileData"]["userFilePath"]
    print("Read user submit file from %s" % submit_path)

    try:
        # TODO: 执行评测逻辑
        # NOTICE: 这个是示例
        # score = 0.8
        # report_score(score, out_path)
        score = evaluate(standard_path, submit_path)
        report_score(score, out_path)

    except Exception as e:
        # NOTICE: 这个只是示例
        check_code = 1
        report_error_msg(error_msg[check_code], error_msg[check_code], out_path)
