import json
import os

def readFile(filePath: str) -> dict | list:
    """读取json文件"""
    with open(filePath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def readFolderFile(folderPath: str) -> dict | list:
    """读取文件夹下所有json文件，不会读取子文件夹的内容"""
    data = []
    for file in os.listdir(folderPath):
        if file.endswith('.json'):
            with open(os.path.join(folderPath, file), 'r', encoding='utf-8') as file:
                data.append(json.load(file))
    return data