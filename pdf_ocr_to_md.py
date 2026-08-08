"""
用 RapidOCR 对【扫描版 PDF】逐页做中文 OCR，输出 Markdown。
用法:
    python pdf_ocr_to_md.py <输入.pdf> [输出.md] [--dpi 200] [--start 1] [--max N]

说明:
  - 用 pypdfium2 把每页渲染成图片，再用 RapidOCR 识别文字。
  - 结果按页写入，每页一个 "## 第 N 页" 小标题；边识别边落盘，中途中断也能保留已完成部分。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("md", nargs="?")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--start", type=int, default=1, help="起始页(从1计)")
    ap.add_argument("--max", type=int, default=0, help="最多处理多少页, 0=全部")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    md_path = Path(args.md).expanduser().resolve() if args.md \
        else pdf_path.with_suffix(".md")

    pdf = pdfium.PdfDocument(str(pdf_path))
    total = len(pdf)
    start_idx = args.start - 1
    end_idx = total if args.max == 0 else min(total, start_idx + args.max)

    print(f"📄 PDF: {pdf_path.name}  共 {total} 页, 本次处理 第{start_idx+1}~{end_idx}页, DPI={args.dpi}")
    print("🔧 初始化 RapidOCR 引擎...")
    engine = RapidOCR()
    scale = args.dpi / 72.0

    t0 = time.time()
    # 追加模式：第一页时清空重建
    mode = "w" if start_idx == 0 else "a"
    with open(md_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# {pdf_path.stem}\n\n> 由 RapidOCR 从扫描版 PDF 识别生成（DPI={args.dpi}）。\n\n")
        for i in range(start_idx, end_idx):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            img = np.asarray(bitmap.to_pil().convert("RGB"))
            result, _ = engine(img)
            lines = [item[1] for item in result] if result else []
            f.write(f"## 第 {i+1} 页\n\n")
            f.write("\n".join(lines) if lines else "*(本页未识别到文字)*")
            f.write("\n\n")
            f.flush()

            done = i - start_idx + 1
            avg = (time.time() - t0) / done
            remain = (end_idx - i - 1) * avg
            print(f"  第{i+1:>3}/{end_idx}页 ✓  识别{len(lines):>3}行  "
                  f"平均{avg:.1f}s/页  预计剩余{remain/60:.1f}分", flush=True)

    size = md_path.stat().st_size
    print(f"\n✅ 完成，用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"📦 Markdown 大小: {human_size(size)}  ({size:,} 字节)")


if __name__ == "__main__":
    main()
