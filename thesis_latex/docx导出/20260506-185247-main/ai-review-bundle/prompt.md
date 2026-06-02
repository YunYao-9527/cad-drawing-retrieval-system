# 天津大学本科毕业论文 AI 综合 Review Prompt

请按天津大学本科毕业设计（论文）常见规范，对这次导出的论文做一轮送审前综合 review。

## 输入材料
- DOCX 文件：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex\docx导出\20260506-185247-main\main.docx`
- LaTeX 项目：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex`
- 主 LaTeX：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex\main.tex`
- 自动格式检查：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex\docx导出\20260506-185247-main\review\report.md`
- DOCX 结构 JSON：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex\docx导出\20260506-185247-main\ai-review-bundle\docx-structure.json`
- LaTeX 源码摘录：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex\docx导出\20260506-185247-main\ai-review-bundle\latex-sources.md`
- 参考文献摘要：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex\docx导出\20260506-185247-main\ai-review-bundle\bibliography-summary.md`

## Review 要求
1. 先阅读 `review-context.md` 和 `review/report.md`，不要重复罗列已经明显正确的项目。
2. 重点检查标题层级、摘要和关键词、目录、图表题注、公式编号、参考文献、附录、致谢、页码/页眉页脚风险。
3. 检查 LaTeX 正文是否存在口语化、备忘录式表达、本地路径、内部文件名、未解释缩写、图表未在正文引用等送审风险。
4. 检查文内引用和参考文献是否存在不一致、缺失、顺序异常、英文作者格式异常。
5. 输出时按严重程度分组：必须修改、建议修改、提交前人工核对。
6. 每条问题尽量说明位置、原因和具体修改建议。

## 输出格式
请输出 Markdown：

```markdown
# 论文送审前 Review 报告

## 必须修改
- ...

## 建议修改
- ...

## 提交前人工核对
- ...

## 已检查但未发现明显问题
- ...
```
