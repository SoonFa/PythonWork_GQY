# -*- coding: utf-8 -*-
# @Time    : 2020年11月2020/11/29日16:23
# @Author  : SoonCj
# @Email   : F_aF_a@163.com
# @File    : GUI.py
# @Software: PyCharm
"""
说明:图形界面

"""
import tkinter as tk
from tkinter import *


def _change_content(s):
    var.set(s)

def upload_file():
    selectFile = tk.filedialog.askopenfilename()
    # askopenfilename 1次上传1个
    # askopenfilenames1次上传多个
    the_button = Button(frame2,
                        text='下一句',
                        command=_change_content  # 点击时调用的函数
                        )
    the_button.pack()

root = tk.Tk()
frame1 = Frame(root)
frame2 = Frame(root)

# Label显示的文字要是会变化的话，只接受这种类型的变量
var = StringVar()
var.set(" ")

frm = tk.Frame(root)
frm.grid(padx='20', pady='30')
btn = tk.Button(frm, text='1.上传图片', command=upload_file)
btn.grid(row=0, column=0, ipadx='3', ipady='3', padx='10', pady='20')
label1 = tk.Label(frm, width='40')
label1.grid(row=0, column=1)

root.mainloop()