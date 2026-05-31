# -*- coding: utf-8 -*-
"""
为参考资料的大 PDF 生成文本索引，方便快速 Grep 检索。
用法（在 study_rl 目录下运行）：
    python 参考资料/_index/build_index.py
PDF 更新后重新运行本脚本即可刷新索引。
"""
import os
import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, '..', '深度学习入门4：强化学习.pdf')
FULLTEXT = os.path.join(HERE, '全文.txt')
TOC = os.path.join(HERE, '目录.md')
LOG = os.path.join(HERE, '_build.log')


def build():
    log = []
    pdf = pdfplumber.open(PDF)
    n = len(pdf.pages)

    # 1) 全文提取，每页加页码标记（PDF_PAGE 为内部索引，从 0 开始）
    parts = []
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ''
        parts.append('=== PDF_PAGE %d ===\n%s' % (i, t))
    with open(FULLTEXT, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(parts))
    log.append('全文.txt 写入完成，PDF 共 %d 页' % n)

    # 2) 目录：扫描前 20 页，凡含点引导线（"...."）的行都视为目录行。
    #    pdfplumber 对部分目录页提取会出现乱码（U+FFFD），这些行直接跳过，
    #    避免污染目录文件。
    BAD = chr(0xFFFD)
    toc_lines = []
    toc_pages = []
    seen = set()
    for i in range(min(20, n)):
        t = pdf.pages[i].extract_text() or ''
        page_has = False
        for line in t.split('\n'):
            s = line.strip()
            if '....' in s and BAD not in s and s not in seen:
                toc_lines.append(s)
                seen.add(s)
                page_has = True
        if page_has:
            toc_pages.append(i)

    with open(TOC, 'w', encoding='utf-8') as f:
        f.write('# 《深度学习入门4：强化学习》目录索引\n\n')
        f.write('> 自动从 PDF 提取。右侧数字是书内印刷页码；\n')
        f.write('> 在 `全文.txt` 中检索时，请用 `=== PDF_PAGE N ===` 标记定位实际页。\n')
        f.write('> （印刷页码通常 = PDF_PAGE 索引 + 4 左右，按实际微调。）\n\n')
        f.write('```\n')
        f.write('\n'.join(toc_lines))
        f.write('\n```\n')
    log.append('目录.md 写入完成，目录页=%s，共 %d 行' % (toc_pages, len(toc_lines)))

    with open(LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))


if __name__ == '__main__':
    build()
