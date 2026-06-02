# LaTeX 终稿说明

这套文件是基于当前项目正式实验结果整理出的 LaTeX 终稿版，入口文件是：

- `main.tex`

目录结构：

- `chapters/`：摘要与七章正文
- `figures/`：论文中引用的图像素材
- `references.bib`：参考文献库

## 已整合内容

- 五阶段正式实验结果
- 标题栏字段级多模态增强后的最新结论
- 注意力机制与 Grad-CAM 可解释性表述
- 系统实现与工程落地内容

## 推荐编译方式

推荐使用 `XeLaTeX` 编译中文论文。

如果本机安装了 `latexmk`，可执行：

```bash
latexmk -xelatex main.tex
```

如果使用基础命令，可执行：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## 当前需要你确认的地方

- `main.tex` 中学校名称暂时保留为 `（请补充学校名称）`
- 如学校有指定论文模板、页边距、封面或参考文献格式要求，可在此基础上进一步套版

## 说明

当前 Codex 环境里没有检测到 `xelatex`/`latexmk` 可执行程序，因此这版已经按标准 LaTeX 结构整理完成，但还没有做本机实际编译校验。
