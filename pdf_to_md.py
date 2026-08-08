"""
使用 markitdown 将 PDF 转换为 Markdown。
用法:
    python pdf_to_md.py <输入.pdf> [输出.md]
若不指定输出路径，则在同目录下生成同名 .md 文件。
"""
import sys
import time
from pathlib import Path

from markitdown import MarkItDown


def human_size(num_bytes: int) -> str:
    """把字节数转成易读的 KB / MB。"""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        print(f"❌ 找不到文件: {pdf_path}")
        sys.exit(1)

    # 输出路径：未指定则用同名 .md
    md_path = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 \
        else pdf_path.with_suffix(".md")

    print(f"📄 输入 PDF : {pdf_path}  ({human_size(pdf_path.stat().st_size)})")
    print(f"📝 输出 MD  : {md_path}")
    print("⏳ 正在转换（大文件可能需要几分钟）...")

    start = time.time()
    md = MarkItDown()
    result = md.convert(str(pdf_path))
    md_path.write_text(result.text_content, encoding="utf-8")
    elapsed = time.time() - start

    size = md_path.stat().st_size
    print(f"✅ 转换完成，用时 {elapsed:.1f} 秒")
    print(f"📦 Markdown 文件大小: {human_size(size)}  ({size:,} 字节)")
    print(f"🔤 字符数: {len(result.text_content):,}")


if __name__ == "__main__":
    main()
