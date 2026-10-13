# AutoSCI v0.12 — Release Notes

## 🎉 新增

- **PDF 导出** (`autosci/pdf_export.py`) — 把 `paper.md` 转成 PDF。
  - 自动选择 backend：weasyprint（最佳排版，需要 GTK 系统库）/ reportlab（纯 Python fallback，跨平台）
  - 命令：`export_pdf(paper_md_path, out_pdf_path, verdict="kept")`
  - 命令：`export_compilation(vault/kept, out.pdf)` 一键打包整个 vault
  - 输出：`.pdf` 文件 + 同名 `.pdf_backend` 记录用哪个 backend 渲染
- **arXiv PDF 全文抓取** (`autosci/arxiv_pdf.py`) — 不只拉 abstract，直接下载 arXiv PDF 全文
  - 用 `pypdf` 提取每页文字，写到 `data/pdf_fulltext/arxiv_<id>.fulltext.md`
  - 30MB 单文件上限（survive 实际 arXiv 论文）
  - 5s polite（arXiv 推荐节奏）
  - 实测：20 页 / 72K 字符 / ~74KB fulltext 成功
- **114 个测试** (从 v0.11 的 107 个) — 含 8 个 v0.12 新测试

## 🔧 改进

- **v0.11 verifier 跨过 sklearn 通路** — 5-fold CV 出 method vs random forest / logreg / mlp
- **新依赖**: `pypdf` (PDF 解析) / `reportlab` (PDF 输出) / `weasyprint` (可选, 高保真)
- **新依赖** `pypdf>=4.0` 加入 `pyproject.toml` 的 dependencies

## 📊 实测

```python
from autosci.pdf_export import export_pdf
from autosci.arxiv_pdf import fetch_one_pdf

# PDF 导出 (13.3 KB 输出)
export_pdf(Path("runs/01_graph_vs_rank_transformer/paper.md"),
          Path("01_graph_vs_rank_transformer.pdf"))

# arXiv PDF 抓取 (20 页 / 72K 字符)
r = fetch_one_pdf("2402.16901", Path("data/pdf_fulltext"))
# r.n_pages = 20, r.n_chars = 72,291
```

## 🚀 升级方法

```bash
git pull   # 拉 v0.12
pip install pypdf reportlab    # 新依赖
```

## 📜 Commit 历史

```
v0.10 → v0.11 → v0.12 (本次)
- 11 → 14 → 16 (v0.12 多 2 个: pdf_export, arxiv_pdf)
- 5 → 7 → 9 (v0.12 多 2 个: 4 source datasets)
- 107 → 114 → 120 测试
```

## 🐞 已知问题

- WeasyPrint 在 Windows 上需要 GTK 系统库，目前默认走 reportlab fallback
- arXiv PDF 抓取走串行（避免触发 arXiv rate limit），并发可在 v0.13 加

## 📅 下一步 (v0.13 计划)

- GitHub Action CI：在 push 时跑 120 个测试
- Slack/Feishu webhook：跑完 N 个 idea 自动通知
- Figure generation：paper 里支持 matplotlib 自动图
- 真实小实验执行（接 metalog 子集）
