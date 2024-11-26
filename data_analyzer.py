"""
本程序独立于整个项目
用于读入对应路径下的全部json文件（需要是插件收集的数据）
统计：
    1. 文件数量
    2. 日志项数量
    3. 日志项类型和对应数量
    4. 工件类型数量
方便开发者对数据进行全局观察
为数据预处模块提供参考
"""

import argparse
import json
import os

fileCount = 0
logCount = 0
logItemCount = {}
artifactList = {}

def getArtifactShortName(name):
    if '/' in name:
        name = name.split('/')[-1] # 处理文件类型工件
        if '?' in name:
            name = name.split('?')[0] # 处理 git 类型工件
    return name

def analyzeSingleJsonFile(filePath):
    global logCount
    global logItemCount
    with open(filePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logCount += len(data) # 统计记录 log-item 的数量
    for item in data:
        # 统计不同事件的数量
        logItemCount[item['eventType']] = logItemCount.get(item['eventType'], 0) + 1 
        # 统计包含的工件的数量
        if 'hierarchy' in item['artifact']: # 多层级工件
            name = ''
            for i in range(len(item['artifact']['hierarchy'])-1):
                if i==0:
                    name = getArtifactShortName(item['artifact']['hierarchy'][i]['name'])
                else:
                    name = name + '->' + getArtifactShortName(item['artifact']['hierarchy'][i]['name'])
                artifactList[name] = artifactList.get(name, 0) + 1
        else: # 单层级工件
            name = getArtifactShortName(item['artifact']['name'])
            artifactList[name] = artifactList.get(name, 0) + 1

def anaylzeFileAndFoler(path): # 递归遍历当前文件夹
    if os.path.isfile(path):
        _, file_extension = os.path.splitext(path)
        if file_extension != '.json':
            return
        print(f'Analyzing {path}...')
        global fileCount
        fileCount += 1
        analyzeSingleJsonFile(path)
    elif os.path.isdir(path):
        for file in os.listdir(path):
            anaylzeFileAndFoler(os.path.join(path, file))
    else:
        print(f'{path} is invalid!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A toy analyzer for collected data by VirtualMe')
    parser.add_argument('-p', '--path', type=str, default='./dataset', help='path of data to be analyzed(folder or file)')
    args = parser.parse_args()
    path = args.path

    print()
    anaylzeFileAndFoler(path)
    
    print(f'\nTotal files number = {fileCount}')
    print(f'Total logs number = {logCount}')
    
    print('-' * 32, 'Log Item Count', '-' * 32)
    sorted_items_desc = sorted(logItemCount.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_items_desc:
        print(f'{key.rjust(20)}: {value}')
    
    print('-' * 32, 'Artifact Count', '-' * 32)
    sorted_artifacts_desc = sorted(artifactList.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_artifacts_desc:
        print(f'{key}: {value}')
