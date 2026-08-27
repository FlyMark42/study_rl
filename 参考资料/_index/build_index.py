# -*- coding: utf-8 -*-
"""
为参考资料生成文本索引，方便快速 Grep 检索。

支持两类资料：
  1) 有文字层的 PDF —— 用 pdfplumber 直接提取
     （如《深度学习入门4：强化学习》，斋藤康毅）。
  2) 扫描版 PDF —— 无文字层，需先用 RapidOCR 转成 md（脚本见项目根目录
     pdf_ocr_to_md.py），本脚本再从该 md 读取
     （如《深度强化学习》，王树森）。

用法（在 study_rl 目录下运行）：
    python 参考资料/_index/build_index.py            # 重建全部
    python 参考资料/_index/build_index.py --only wang # 只重建王树森
    python 参考资料/_index/build_index.py --only saito# 只重建斋藤

资料更新后重新运行本脚本即可刷新索引。
"""
import os
import re
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.path.join(HERE, '..'))

# ---- 书目配置 ----
SAITO_PDF = os.path.join(REF, '深度学习入门_斋藤康毅', '深度学习入门4：强化学习.pdf')
SAITO_FULLTEXT = os.path.join(HERE, '全文.txt')
SAITO_TOC = os.path.join(HERE, '目录.md')

WANG_MD = os.path.join(REF, '深度强化学习_王树森', '深度强化学习_王树森.md')
WANG_FULLTEXT = os.path.join(HERE, '全文_王树森.txt')
WANG_TOC = os.path.join(HERE, '目录_王树森.md')

LOG = os.path.join(HERE, '_build.log')


# ============ 1. 有文字层的 PDF（斋藤康毅） ============
def build_saito(log):
    import pdfplumber
    pdf = pdfplumber.open(SAITO_PDF)
    n = len(pdf.pages)

    # 全文提取，每页加页码标记（PDF_PAGE 从 0 开始）
    parts = []
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ''
        parts.append('=== PDF_PAGE %d ===\n%s' % (i, t))
    with open(SAITO_FULLTEXT, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(parts))
    log.append('[斋藤] 全文.txt 写入完成，PDF 共 %d 页' % n)

    # 目录：扫描前 20 页，凡含点引导线（"...."）的行视为目录行
    BAD = chr(0xFFFD)
    toc_lines, toc_pages, seen = [], [], set()
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
    with open(SAITO_TOC, 'w', encoding='utf-8') as f:
        f.write('# 《深度学习入门4：强化学习》目录索引\n\n')
        f.write('> 自动从 PDF 提取。右侧数字是书内印刷页码；\n')
        f.write('> 在 `全文.txt` 中检索时，请用 `=== PDF_PAGE N ===` 标记定位实际页。\n')
        f.write('> （印刷页码通常 = PDF_PAGE 索引 + 4 左右，按实际微调。）\n\n')
        f.write('```\n')
        f.write('\n'.join(toc_lines))
        f.write('\n```\n')
    log.append('[斋藤] 目录.md 写入完成，目录页=%s，共 %d 行' % (toc_pages, len(toc_lines)))


# ============ 2. 扫描版（王树森，从 OCR md 读取） ============
PAGE_RE = re.compile(r'^##\s*第\s*(\d+)\s*页\s*$')
CHAP_RE = re.compile(r'^第\s*(\d{1,2})\s*章\s*(\S.*)$')
PART_RE = re.compile(r'^第\s*([一二三四五六七八九十]+)\s*部分\s*(.*)$')
SEC_RE = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s*([一-鿿].*)$')
# 句子标点：真正的标题不会含这些，用来滤掉前言描述句与正文交叉引用
SENT_PUNC = set('，。、；：？！．…（）()《》【】')
CN2NUM = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
          '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}


def _chap_title(s):
    """是干净的章标题则返回标题文本，否则 None（前言句/交叉引用句会被排除）。"""
    m = CHAP_RE.match(s)
    if not m:
        return None
    title = m.group(2).strip()
    if 1 <= len(title) <= 15 and not (set(title) & SENT_PUNC):
        return title
    return None


def _part_title(s):
    """是分隔页里的部分标题则返回 (中文序号, 同行标题or'')，前言描述句返回 None。"""
    m = PART_RE.match(s)
    if not m:
        return None
    rest = m.group(2).strip()
    if rest and (rest[0] == '是' or (set(rest) & SENT_PUNC) or len(rest) > 8):
        return None
    return m.group(1), rest


def _is_div_heading(s):
    """下一行是否为‘标题行’（章 or 部分分隔），用于跳过分隔页里的罗列。"""
    return _chap_title(s) is not None or _part_title(s) is not None


def build_wang(log):
    with open(WANG_MD, encoding='utf-8') as f:
        lines = f.read().split('\n')

    # 2.1 全文（把 "## 第 N 页" 统一成 "=== PDF_PAGE N ==="，与斋藤格式一致）
    out = [
        '# 《深度强化学习》（王树森）全文检索文件',
        '# 本书为扫描版，正文由 RapidOCR 识别；页标记 N = PDF 第 N 页（从 1 起）。',
        '# 公式与图表 OCR 可能有误，阅读这两类内容时请对照原书 PDF。',
        '',
    ]
    for line in lines:
        m = PAGE_RE.match(line.strip())
        out.append('=== PDF_PAGE %s ===' % m.group(1) if m else line)
    with open(WANG_FULLTEXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    log.append('[王树森] 全文_王树森.txt 写入完成，共 %d 行' % len(lines))

    # 2.2 目录：抽取“部分/章/节”标题并映射到页码
    seq, page = [], 0            # seq: [(文本, 所在页), ...]，已剔除页标记与空行
    for line in lines:
        s = line.strip()
        m = PAGE_RE.match(s)
        if m:
            page = int(m.group(1))
        elif s:
            seq.append((s, page))

    parts, cand, secs = {}, {}, {}   # cand: 章号 -> [(页, 标题), ...] 所有候选位置
    for i, (s, pg) in enumerate(seq):
        nxt = seq[i + 1][0] if i + 1 < len(seq) else ''
        pt = _part_title(s)
        if pt is not None:
            cn, inline = pt
            name = inline or (nxt if nxt and not _is_div_heading(nxt) else '')
            parts.setdefault(CN2NUM.get(cn, 99), (pg, cn, name))
            continue
        ct = _chap_title(s)
        if ct is not None:
            # 跳过“部分分隔页”里的章节罗列（其后紧跟另一个标题行）
            if _is_div_heading(nxt):
                continue
            num = int(CHAP_RE.match(s).group(1))
            cand.setdefault(num, []).append((pg, ct))
            continue
        sm = SEC_RE.match(s)
        if sm:
            key, title = sm.group(1), sm.group(2).strip()
            if title[:1] != '节' and 2 <= len(title) <= 30 \
                    and not (set(title) & SENT_PUNC):
                secs.setdefault(key, (pg, title))

    # 章页码单调递增选择：每章取第一个比上一章更大的候选页，自动跳过分隔页罗列
    chaps, prev = {}, 0
    for num in sorted(cand):
        lst = sorted(cand[num])
        chaps[num] = next(((pg, t) for pg, t in lst if pg > prev), lst[-1])
        prev = chaps[num][0]

    # 结构校验：小节页码必须落在本章页码区间内，滤掉 OCR 误配（如 “2.4…第230页”）
    order = sorted(chaps)
    span = {}
    for j, num in enumerate(order):
        start = chaps[num][0]
        end = chaps[order[j + 1]][0] if j + 1 < len(order) else 10 ** 9
        span[num] = (start, end)

    def _sec_ok(key, pg):
        top = int(key.split('.')[0])
        if top not in span:
            return False
        start, end = span[top]
        return start - 1 <= pg < end
    secs = {k: v for k, v in secs.items() if _sec_ok(k, v[0])}

    def sec_key(k):
        return [int(x) for x in k.split('.')]

    with open(WANG_TOC, 'w', encoding='utf-8') as f:
        f.write('# 《深度强化学习》（王树森）目录索引\n\n')
        f.write('> 扫描版经 RapidOCR 识别后自动抽取，页码为 PDF 第 N 页（从 1 起），\n')
        f.write('> 可能有 ±1~2 页误差。在 `全文_王树森.txt` 中用 `=== PDF_PAGE N ===` 定位。\n\n')
        if parts:
            f.write('## 分部\n\n')
            for k in sorted(parts):
                pg, cn, name = parts[k]
                f.write('- 第%s部分 %s … 第 %d 页\n' % (cn, name, pg))
            f.write('\n')
        f.write('## 章 / 节\n\n')
        for num in order:
            pg, title = chaps[num]
            f.write('- **第%d章 %s** … 第 %d 页\n' % (num, title, pg))
            for key in sorted((k for k in secs if k.split('.')[0] == str(num)),
                              key=sec_key):
                spg, stitle = secs[key]
                f.write('  - %s %s … 第 %d 页\n' % (key, stitle, spg))
        f.write('\n')
    log.append('[王树森] 目录_王树森.md 写入完成，分部 %d，章 %d，节 %d'
               % (len(parts), len(chaps), len(secs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['saito', 'wang', 'all'], default='all')
    args = ap.parse_args()

    log = []
    if args.only in ('saito', 'all'):
        build_saito(log)
    if args.only in ('wang', 'all'):
        build_wang(log)
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))
    print('\n'.join(log))


if __name__ == '__main__':
    main()
