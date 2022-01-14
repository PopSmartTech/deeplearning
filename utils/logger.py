# -*- coding: UTF-8 -*-
# @Time        :   1:57
# @Author      :  Huangxiao
# @application :  
# @File        :  test.py
import logging

class Logger(object):
    def __init__(self,filePath):

        self.fileName = filePath
        self.logger = logging.getLogger(self.fileName)
        self.logger.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()
        showFormat = logging.Formatter('%(levelname)s [%(asctime)s] || %(funcName)s %(message)s')
        self.ch.setFormatter(showFormat)
        self.formatter = logging.Formatter('%(levelname)s [%(asctime)s] || %(funcName)s %(message)s')
        self.fh = logging.FileHandler(self.fileName)
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

    def info(self,message):
        self.logger.info(message)

    def dataPrint(self,data):
        logText = "DATA OUTPUT || DataType: %s - Intermediate Data: %s"
        if type(data) == str:
            logText = logText %(type(data),data)
        elif type(data) in (dict,set,list,tuple):
            logText = logText % (type(data),str(data))
        else :
            logText = "DATA OUTPUT || DataType: {0} - Intermediate Data: {1}".format(type(data), data)
        self.logger.info(logText)


    def error(self,message):
        self.logger.error(message)

    def warning(self,message):
        self.logger.warning(message)

    def debug(self,message):
        self.logger.debug(message)

if __name__ == '__main__':
    import json
    logger = Logger('test')
    logger.info('info')
    logger.error('error')
    logger.warning('error')
    logger.debug('error')
    data = {
        'test':'info'
    }
    data = json.dumps(data, ensure_ascii=False).encode('utf-8')
    logger.dataPrint(data)
    # print(data)