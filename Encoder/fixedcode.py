#-*- coding: utf-8
from decimal import Decimal
def bTod(n, pre):
    '''
    把一个带小数的二进制数n转换成十进制
    小数点后面保留pre位小数
    '''
    string_number1 = str(n)          #number1 表示二进制数，number2表示十进制数
    if string_number1[0] == '0':     #判断正负，负数flag1=True
        flag1 = True
    else:
        flag1 = False
    string_number1 = string_number1[1:]    #去除符号位
    #print(string_number1)
    if string_number1=='':
        return 0
    decimal = 0  #小数部分化成二进制后的值
    flag = False
    for i in string_number1: #判断是否含小数部分
        if i == '.':
            flag = True
            break
    if flag: #若二进制数含有小数部分
        string_integer, string_decimal = string_number1.split('.') #分离整数部分和小数部分
        for i in range(len(string_decimal)):
            decimal += 2**(-i-1)*int(string_decimal[i])  #小数部分化成二进制
        if string_integer == '':
            number2 = 0 + decimal
        else:
            number2 = int(str(int(string_integer, 2))) + decimal
        if flag1 == True:
            number2 = -number2
        return round(number2, pre)
    else: #若二进制数只有整数部分
        if flag1 == True:
            string_number1 = -string_number1
        return int(string_number1, 2)#若只有整数部分 直接一行代码二进制转十进制 python还是骚

def dTob(n, pre):
    '''
    把一个带小数的十进制数n转换成二进制
    小数点后面保留pre位小数
    '''
    flag1 = False
    if n<0:
        n = -n              #将负数转换为正数进行编码，若为正数falg1=True
        flag1 = True
    string_number1 = str(n) #number1 表示十进制数，number2表示二进制数
    flag = False
    for i in string_number1: #判断是否含小数部分
        if i == '.':
            flag = True
            break
    if flag:
        string_integer, string_decimal = string_number1.split('.') #分离整数部分和小数部分
        integer = int(string_integer)
        decimal = Decimal(str(n)) - integer
        l1 = [0,1]
        l2 = []
        decimal_convert = ""
        while True:
           if integer == 0: break
           x,y = divmod(integer, 2)  #x为商，y为余数
           l2.append(y)
           integer = x
        string_integer = ''.join([str(j) for j in l2[::-1]])  #整数部分转换成二进制
        i = 0
        while decimal != 0 and i < pre:
            result = int(decimal * 2)
            decimal = decimal * 2 - result
            decimal_convert = decimal_convert + str(result)
            i = i + 1
        if flag1 ==True:
            string_number2 = '0'+string_integer + '.' + decimal_convert   #设置符号位，0表示负数，1表示正数
        else:
            string_number2 = '1'+string_integer + '.' + decimal_convert
        return string_number2
    else: #若十进制只有整数部分
        l1 = [0,1]
        l2 = []
        while True:
           if n == 0: break
           x,y = divmod(n, 2)  #x为商，y为余数
           l2.append(y)
           n = x
        string_number = ''.join([str(j) for j in l2[::-1]])
        if flag1 ==True:
            string_number = '0'+string_number
        else:
            string_number = '1'+string_number
        return string_number



