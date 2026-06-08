# -*- coding: utf-8 -*-
import logging

# 配置日志
logger = logging.getLogger(__name__)

import re
import six
import warnings
from functools import wraps
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable

import pandas as pd
from datetime import datetime

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def send_email(sender, password, recipient, subject, body, attachment=None):
    """
    发送邮件的函数

    :param sender: 发件人邮箱地址
    :param password: 发件人邮箱密码
    :param recipient: 收件人邮箱地址，支持以下格式：
                     - 单个邮箱地址字符串: "user@example.com"
                     - 多个邮箱地址字符串（逗号分隔）: "user1@example.com,user2@example.com"
                     - 邮箱地址列表: ["user1@example.com", "user2@example.com"]
                     - 邮箱地址元组: ("user1@example.com", "user2@example.com")
    :param subject: 邮件主题
    :param body: 邮件正文
    :param attachment: 附件文件路径
    :raises: Exception: 当邮件发送失败时抛出异常
    """
    # 处理收件人地址，统一转换为列表格式
    if isinstance(recipient, str):
        # 如果是字符串，按逗号分割
        recipients = [email.strip() for email in recipient.split(',')]
    elif isinstance(recipient, (list, tuple)):
        # 如果是列表或元组，直接使用
        recipients = list(recipient)
    else:
        # 其他情况，转换为字符串列表
        recipients = [str(recipient)]
    
    # 过滤空字符串
    recipients = [email for email in recipients if email.strip()]
    
    if not recipients:
        raise ValueError("收件人地址不能为空")
    
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)  # 在邮件头中显示所有收件人
    msg['Subject'] = subject

    # 添加邮件正文
    body_text = MIMEText(body, 'plain', 'utf-8')
    msg.attach(body_text)

    # 添加附件
    if attachment is not None:
        with open(attachment, 'rb') as f:
            attachment_file = MIMEApplication(f.read())
            attachment_file.add_header('Content-Disposition', 'attachment', filename=attachment)
            msg.attach(attachment_file)

    # 发送邮件
    try:
        server = smtplib.SMTP_SSL('smtp.126.com', 465)
        server.login(sender, password)
        # 发送给所有收件人
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        print('send email success!!!')
    except Exception as e:
        print(f'send email error!!! {e}')
        raise  # 重新抛出异常，让调用方知道发送失败

# def send_email(sender, password, recipient, subject, body, attachment=None):
#     """
#     发送邮件的函数

#     :param sender: 发件人邮箱地址
#     :param password: 发件人邮箱密码
#     :param recipient: 收件人邮箱地址
#     :param subject: 邮件主题
#     :param body: 邮件正文
#     :param attachment: 附件文件路径
#     """
#     # 创建邮件对象
#     msg = MIMEMultipart()
#     msg['From'] = sender
#     msg['To'] = recipient
#     msg['Subject'] = subject

#     # 添加邮件正文
#     body_text = MIMEText(body, 'plain', 'utf-8')
#     msg.attach(body_text)

#     # 添加附件
#     if attachment is not None:
#         with open(attachment, 'rb') as f:
#             attachment_file = MIMEApplication(f.read())
#             attachment_file.add_header('Content-Disposition', 'attachment', filename=attachment)
#             msg.attach(attachment_file)

#     # 发送邮件
#     try:
#         server = smtplib.SMTP_SSL('smtp.126.com', 465)
#         server.login(sender, password)
#         server.sendmail(sender, recipient, msg.as_string())
#         server.quit()
#         print('send email success!!!')
#     except Exception as e:
#         print(f'send email error!!! {e}')
#         raise  # 重新抛出异常，让调用方知道发送失败