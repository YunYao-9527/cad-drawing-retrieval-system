# LaTeX 源码摘录

- 项目目录：`D:\workspace\图纸检索项目\数据集\111_test\0version_1\thesis_latex`
- 收录文件数：13

## `main.tex`

```tex
\documentclass[UTF8,openany]{ctexbook}

\usepackage[a4paper,top=27.5mm,bottom=25.4mm,left=35.7mm,right=27.7mm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{array}
\usepackage{adjustbox}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{float}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{longtable}
\usepackage{makecell}
\usepackage{multirow}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{subcaption}
\usepackage{tabularx}
\usepackage{threeparttable}
\usepackage{xcolor}
\usepackage[nameinlink,noabbrev]{cleveref}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=blue!50!black,
  urlcolor=blue!60!black,
  pdftitle={基于预训练模型微调的工程图纸检索系统设计},
  pdfauthor={王登政}
}

\captionsetup{font=small,labelfont=bf}
\setlist[itemize]{leftmargin=2em}
\setlist[enumerate]{leftmargin=2.4em}
\renewcommand{\arraystretch}{1.25}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}
\crefname{table}{表}{表}
\Crefname{table}{表}{表}
\crefname{figure}{图}{图}
\Crefname{figure}{图}{图}
\crefname{equation}{公式}{公式}
\Crefname{equation}{公式}{公式}

\newcommand{\thesistitle}{基于预训练模型微调的工程图纸检索系统设计}
\newcommand{\englishtitle}{Design of an Engineering Drawing Retrieval System Based on Pretrained Model Fine-Tuning}
\newcommand{\universityname}{天津大学}
\newcommand{\schoolname}{自动化学院}
\newcommand{\majorname}{电子信息工程}
\newcommand{\studentname}{王登政}
\newcommand{\studentid}{3022234549}
\newcommand{\advisorname}{宋丹}
\newcommand{\gradeinfo}{2022级}
\newcommand{\finishdate}{2026年6月}

\fancypagestyle{thesisfancy}{
  \fancyhf{}
  \fancyhead[C]{\small 天津大学2026届本科生毕业设计（论文）}
  \fancyfoot[C]{\small \thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0pt}
}

\let\cleardoublepage\clearpage

\begin{document}

\begin{titlepage}
  \thispagestyle{empty}
  \centering
  \vspace*{2.5cm}
  {\zihao{2}\bfseries \universityname \par}
  \vspace{1.2cm}
  {\zihao{-0}\bfseries 本科毕业设计（论文）\par}
  \vspace{2cm}
  {\zihao{2}\bfseries \thesistitle \par}
  \vspace{2.2cm}
  \renewcommand{\arraystretch}{1.6}
  \begin{tabular}{p{3.5cm}p{8.5cm}}
    学\qquad 院： & \schoolname \\
    专\qquad 业： & \majorname \\
    年\qquad 级： & \gradeinfo \\
    姓\qquad 名： & \studentname \\
    学\qquad 号： & \studentid \\
    指导教师： & \advisorname \\
  \end{tabular}
  \vfill
  {\zihao{-3}\finishdate\par}
\end{titlepage}

\input{chapters/inner_cover}
\input{chapters/declaration}

\frontmatter
\pagestyle{plain}
\pagenumbering{Roman}
\input{chapters/abstract}
\tableofcontents

\mainmatter
\pagestyle{thesisfancy}
\input{chapters/01_introduction}
\input{chapters/02_related_work}
\input{chapters/03_system_design}
\input{chapters/04_method}
\input{chapters/05_experiments}
\input{chapters/06_implementation}
\input{chapters/07_conclusion}

\backmatter
\bibliographystyle{gbt7714-numerical}
\bibliography{references}
\input{chapters/appendix}
\input{chapters/acknowledgements}

\end{document}

```

## `chapters\01_introduction.tex`

```tex
\chapter{绪论}

绪论部分主要围绕课题提出背景、研究价值、相关研究现状以及本文拟解决的问题展开论述，其作用在于说明“为什么要做工程图纸相似检索”“现有方法还缺什么”以及“本文准备如何组织研究工作”。通过本章的铺垫，可以为后续系统设计、方法构建与实验分析提供清晰的问题定义和研究主线。

\section{研究背景}

随着机械制造、工艺设计和产品研发过程逐渐数字化，企业和科研机构积累了大量历史工程图纸。工程图纸不仅记录零件结构与装配关系，还包含图号、材料、比例、技术要求等重要信息。对于工程人员而言，从历史图库中快速检索出与当前图纸相似的图纸，能够显著提高设计复用效率，减少重复设计和试错成本。

传统工程图纸管理方式多依赖人工目录分类、图号检索或关键字搜索。这些方式虽然适合“已知图号”的场景，但在“只知道当前图纸外观、希望找到结构相近历史图纸”的任务中效果有限。尤其是在大量零件图、装配图和工序图混合存储的情况下，仅依赖文本信息难以完成高质量检索。

近年来，深度学习特别是视觉语义嵌入模型的发展为图像检索提供了新的思路。CLIP 作为典型的视觉--语言预训练模型，具备较强的跨任务泛化能力~\citep{radford2021clip}。然而工程图纸不同于自然图像，其主要信息由线条布局、几何轮廓、标题栏和尺寸标注构成，颜色和纹理特征极弱。因此，将通用视觉模型直接用于工程图纸检索往往会忽略结构主导信息。围绕这一问题开展有针对性的模型改进，具有明确的学术价值与工程意义。

\section{研究意义}

本文研究的意义主要体现在以下几个方面。

\begin{enumerate}
  \item 在应用层面，工程图纸相似检索有助于提升企业历史图纸复用效率和设计标准化水平。
  \item 在方法层面，工程图纸场景为 CLIP 等通用视觉模型的工业迁移提供了具有代表性的研究对象。
  \item 在工程层面，构建完整的检索服务、实验评测与可解释性工具链，有助于将算法研究转化为可落地系统。
  \item 在论文层面，围绕双主线检索框架、结构重排、标题栏字段级 OCR 辅助和 Grad-CAM 分析形成了较完整的方法设计与实验支撑。
\end{enumerate}

\section{研究目标与主要指标}

结合任务书与开题报告中对课题目标的要求，本文拟完成一个面向机械设计场景的工程图纸检索系统，并在算法效果与系统性能两个层面达到可验证的毕业设计目标。具体而言，本文希望在完成数据集清理与系统实现的基础上，重点达成以下目标：

\begin{enumerate}
  \item 构建面向工程图纸场景的专用图库，形成覆盖多类别机械零部件与装配图的标准化数据基础；
  \item 建立基于预训练模型迁移的视觉检索框架，并围绕工程图纸场景完成主体结构增强、结构重排与文本辅助检索的改进设计；
  \item 通过正式实验验证不同技术模块的作用差异，形成可复查、可比较的实验结论；
  \item 实现可运行的原型系统，使图纸上传、特征提取、相似检索、结果展示和图库管理形成完整闭环。
\end{enumerate}

从量化指标角度看，任务书与开题报告对系统提出了较明确的目标要求，即在大规模图纸数据上达到较高的 Top-1 检索准确率与较好的前列排序质量，同时保证端到端响应时间满足工程应用需求。本文在实际研究过程中延续了这一目标导向，并进一步采用 Recall@K、mAP@10、nDCG@10、FT、ST 和 ANMRR 等多项指标对系统进行综合评价，从而使实验结果不仅能够体现单点命中能力，也能够体现候选排序整体质量。

\section{国内外研究现状}

图像检索研究大体经历了手工特征方法、卷积神经网络方法以及视觉 Transformer 和多模态预训练模型方法的发展阶段。传统手工特征方法依赖边缘、纹理和形状描述子，但对复杂工程图纸的鲁棒性不足。深度特征方法通过学习统一嵌入空间提高了检索能力，但自然图像预训练特征在工程图纸场景中容易受模板边框和局部噪声影响~\citep{zhang2023cbirsurvey,liu2024matchingreview}。

近年来，CLIP 等多模态预训练模型在图像理解和检索任务中表现优异~\citep{radford2021clip,han2024crossmodalreview}，相关工作也证明了其在草图检索等结构主导任务中的潜力~\citep{sain2023clipforallthings}。与此同时，参数高效微调技术，如 CLIP-Adapter~\citep{gao2024clipadapter} 与 CoOp~\citep{zhou2022coop}，为将大模型迁移到垂直场景提供了可行路径。

在 CAD 图形与工程图纸检索方向，已有研究尝试结合轮廓、结构关系、图号和文本信息进行混合检索~\citep{mahajan2025orthocad,heidari2025geometriccad}。然而，多数工作要么依赖复杂规则，要么缺乏完整实验评测与系统实现验证。特别是如何将标题栏字段作为强语义信息融入视觉检索结果，并在不破坏召回性能的前提下改善前排排序，仍值得深入研究。

\section{研究内容与创新点}

本文围绕工程图纸相似检索任务，主要开展以下工作：

\begin{enumerate}
  \item 构建基于 CLIP 视觉嵌入的工程图纸检索基线。
  \item 将方法路线重构为 YOLO 清洗路线和无清洗注意力式视觉增强路线，并围绕两条主线组织正式对比实验。
  \item 设计掩码引导的局部主体增强机制与结构描述子重排序方法，提高结构相似检索能力。
  \item 设计 OCR 文本辅助方案，并进一步实现标题栏字段级多模态排序增强。
  \item 实现完整的系统原型、正式实验脚本、误检分析和 Grad-CAM 可解释性工具。
\end{enumerate}

本文的主要创新点如下：

\begin{enumerate}
  \item 提出了一种面向工程图纸的双主线检索组织方式，将原有串行堆叠流程重构为 YOLO 清洗路线与无清洗注意力式视觉增强路线，从而能够更清楚地比较显式清洗与编码阶段主体增强两种思路。
  \item 提出了一种掩码引导的局部主体增强与轻量结构描述子重排序组合方法，将主体区域强化和结构先验用于向量召回后的候选精排。
  \item 提出了一种标题栏字段级 OCR 多模态辅助方法，从标题栏中抽取图号、零件名、材料和比例等字段，作为候选重排序修正信号；同时通过正式实验如实验证其当前增益并不稳定。
  \item 提出并实现了基于 Grad-CAM 的成对检索可视化工具，用于解释模型在判定两张图纸相似时的关注区域。
\end{enumerate}

\section{论文结构安排}

全文共分为七章。第 1 章介绍研究背景、研究意义与创新点；第 2 章介绍相关理论与关键技术；第 3 章给出系统总体设计；第 4 章重点介绍改进检索方法；第 5 章介绍实验设计与结果分析；第 6 章介绍系统实现；第 7 章对全文进行总结并展望未来工作。

综上，本章完成了课题背景、研究问题与论文整体安排的说明，明确了本文以工程图纸相似检索为核心任务，以双主线方法设计和系统实现验证为主要研究路径。后续章节将在此基础上进一步展开相关理论、系统结构、方法细节与实验结果分析。

```

## `chapters\02_related_work.tex`

```tex
\chapter{相关理论与关键技术}

为了使后续方法设计建立在清晰、可追溯的技术基础之上，本章从工程图纸检索任务所涉及的几个核心方向出发，对本文所依赖的理论与关键技术进行梳理。具体包括 CLIP 视觉语义表示、参数高效微调、向量检索与重排、局部注意机制、OCR 字段级多模态融合以及 Grad-CAM 可解释性分析。通过对这些技术的归纳，可以明确本文方法设计的理论来源、工程可行性与改进空间。

\section{工程图纸检索任务特点}

工程图纸检索与自然图像检索存在显著差异。自然图像通常以纹理、颜色和语义对象为主要判别线索，而工程图纸更强调轮廓、投影视图、尺寸标注、标题栏字段和图面布局等结构性信息。因此，在工程图纸场景中，单纯依赖自然图像任务中常见的全局语义表征，往往难以充分区分外形相近但结构关系不同的图纸样本。

此外，工程图纸还具有三个典型特点。第一，图纸的类别边界往往由局部几何关系、剖视方式和投影组合决定，判别线索更细粒度。第二，标题栏中包含图号、零件名称、材料和比例等强语义信息，这些文本字段对排序具有补充价值。第三，图纸在来源转换过程中容易出现黑底、边框、截屏痕迹和模板差异，这些噪声会干扰视觉编码与文本提取。正是由于上述特点，工程图纸检索更适合采用“视觉召回 + 结构增强 + 文本辅助”的综合策略，而不是简单套用通用图像检索流程。

\section{CLIP 模型原理}

CLIP 通过大规模图文对对比学习，将图像与文本映射到统一嵌入空间中~\citep{radford2021clip}。对于输入图像 $I$ 和文本 $T$，图像编码器与文本编码器分别输出特征向量 $\bm{v}$ 和 $\bm{t}$，再通过余弦相似度进行匹配：
\begin{equation}
  s(I,T)=\frac{\bm{v}^{\top}\bm{t}}{\|\bm{v}\|_2\|\bm{t}\|_2}.
\end{equation}
该机制使 CLIP 具备较强的跨任务泛化能力，因此可以作为工程图纸检索的视觉主干。但对于结构主导型工程图纸，仅依赖全局语义表示仍难以充分刻画局部线稿结构特征。

\section{参数高效微调与 Adapter}

为了在保留大模型通用知识的同时适应特定领域，参数高效微调技术应运而生。CLIP-Adapter 通过在视觉或文本分支后附加轻量 MLP 实现快速领域适配~\citep{gao2024clipadapter}；CoOp 则通过可学习提示优化文本侧语义表达~\citep{zhou2022coop}。这类方法降低了全量微调的成本，也为垂直工业场景提供了更稳定的迁移策略。

在本文系统的前期实现中，曾参考“视觉主干 + 轻量适配”的思路，但在后续检索优化过程中，研究重点逐步转向结构建模与多模态重排。其主要原因在于，相较于重新训练更复杂的模型，在既有检索框架中引入结构信息与字段级语义信息，更有利于获得可解释、可验证且可复现的性能改进。

\section{工程图纸与 CAD 检索相关研究}

围绕工程图纸、草图和 CAD 数据的检索研究，近年来逐渐形成了两条较有代表性的思路。第一类工作关注跨模态或跨视图匹配，例如从二维正投影视图检索三维 CAD 模型、从草图检索图像或模型等。这类研究表明，预训练视觉语义模型与图结构描述在工程数据中具有良好的迁移潜力~\citep{sain2023clipforallthings,mahajan2025orthocad,heidari2025geometriccad}。第二类工作则更加关注内容检索流程本身，包括图像表示、相似度建模、候选重排与索引组织等~\citep{zhang2023cbirsurvey,liu2024matchingreview,han2024crossmodalreview}。

不过，已有研究大多聚焦公开草图数据集、自然图像检索任务或 CAD 模型对齐任务，而对工业企业内部常见的工程图纸数据关注较少。实际工程图纸往往具有模板差异大、转换噪声多、标题栏布局复杂、结构相似样本密集等特点，这使得现有方法在直接迁移时面临较强的领域鸿沟。本文工作正是在这样的背景下展开：一方面利用预训练模型微调后的视觉表征作为统一起点，另一方面结合结构重排序与字段级 OCR 辅助，面向真实工程图纸场景构建可运行的检索系统。

\section{向量检索与重排技术}

向量检索将输入图纸编码为高维嵌入向量，再通过近似最近邻搜索完成候选召回。分层小世界图是目前常用的近似最近邻索引结构之一，兼顾检索精度与效率~\citep{malkov2018hnsw}。向量数据库则在索引之上进一步提供元数据过滤、持久化和服务化能力~\citep{pan2024vectordbsurvey,qdrant2024site}。

然而，初始召回主要依赖全局特征，因此往往需要在候选阶段引入结构重排、多模态排序修正等机制，以提高前排结果质量。对于工程图纸任务，这一步尤其重要，因为许多零件图在全局轮廓上高度接近，仅靠单一视觉向量容易出现前排混淆。

在工程实现层面，向量检索系统不仅要追求召回质量，还要兼顾构建效率、更新成本和在线响应时间。面向大规模向量库的 GPU 加速索引和相似度计算研究表明，合理的近似搜索结构与批量编码方式能够显著降低系统延迟~\citep{johnson2019faiss}。因此，本文在算法设计之外，也将 GPU 化建库、离线缓存和候选重排机制纳入系统实现考虑范围。

\section{局部注意机制}

注意力机制最初用于序列建模任务，随后在视觉 Transformer 中被广泛用于图像 patch token 之间的关系建模~\citep{vaswani2017attention}。视觉 Transformer 将图像划分为 patch token 后进行全局建模，使模型能够在较大范围内捕获局部区域之间的依赖关系。本文引入的掩码引导局部注意机制并非重新设计复杂的注意力网络，而是利用图纸预处理阶段生成的掩码，将结构保留区域映射到 patch 空间，在特征聚合时对主体结构区域赋予更高权重。与直接增加深层注意力模块相比，该方案具有更低的额外计算开销，也更便于在现有工程系统中部署。

\section{OCR 与标题栏字段级多模态融合}

工程图纸标题栏通常包含图号、零件名、材料、比例和数量等关键信息。这些字段往往比普通说明文字更具类别区分性。传统 OCR 加分方案通常直接比较整段文本相似度，容易受到模板说明、技术要求等噪声影响。本文将 OCR 从“通用文本加分”升级为“标题栏字段级辅助”，使多模态信号更加聚焦于强语义字段，而非被大段说明文本干扰。

\section{Grad-CAM 可解释性分析}

Grad-CAM 通过对目标分数进行梯度回传，生成神经网络在当前任务中的关注热力图~\citep{selvaraju2017gradcam}。本文以查询图与候选图的嵌入相似度作为目标分数，计算 patch 级 Grad-CAM 热图，用于分析模型在相似检索中的决策依据。该工具一方面可以为论文中的注意力机制提供直观证据，另一方面也能辅助误检案例分析与后续模型诊断。

\section{本章评述}

综合已有研究可以看出，预训练视觉模型为工程图纸检索提供了良好的基础表征能力，参数高效微调降低了领域适配成本，向量检索与候选重排为大规模图库提供了可行的工程路径，OCR 和可解释性分析则为结果增强与误检诊断提供了补充工具。但同时也应看到，现有研究中仍存在三个不足：其一，对“显式清洗”与“无清洗主体增强”两种策略的系统对比不充分；其二，结构信息往往被简单视为附加特征，而缺乏与视觉召回紧密结合的精排机制；其三，多模态文本信号在工业图纸中的使用常常停留在通用文本比对层面，缺少针对标题栏字段的针对性设计。

\section{本章小结}

本章围绕本文研究所需的关键理论与技术基础进行了系统梳理。总体而言，CLIP 为工程图纸检索提供了统一的视觉表征起点，参数高效微调为领域适配提供了低成本路径，向量检索与候选重排构成了系统的基本工程框架，局部注意机制与结构增强为解决图纸主体表达不足问题提供了方法依据，OCR 字段级融合为引入强语义信息提供了补充路径，Grad-CAM 则为后续结果解释与误检分析提供了分析工具。上述内容共同构成了本文方法设计与系统实现的理论基础，下一章将在此基础上给出系统总体设计。

```

## `chapters\03_system_design.tex`

```tex
\chapter{系统总体设计}

在明确研究问题与相关技术基础之后，本文需要进一步回答方法如何在系统层面被组织、各模块如何协同工作以及实验平台如何支持双主线对比的问题。因此，本章从整体架构角度出发，对工程图纸检索系统的目标、层次结构、模块划分与运行流程进行说明，为后续方法实现与实验复现奠定系统基础。

\section{系统目标}

本文系统的目标是在输入一张工程图纸后，从图库中检索出与其结构和语义相似的其他图纸，并返回排序分数、结构分析信息、多模态辅助信号和可解释性可视化结果。系统既要满足算法研究的可验证性，也要满足工程部署的可运行性。

\section{系统功能需求与性能需求}

结合任务书和开题报告中对毕业设计的功能描述，本文系统的需求可以划分为功能性需求与性能性需求两个方面。

\subsection{功能性需求}

\begin{enumerate}
  \item \textbf{图纸上传与检索需求。} 用户应能够上传查询图纸，并获得与之结构或语义相近的候选图纸结果列表。
  \item \textbf{图库管理需求。} 系统应支持图库统计、类别浏览、图像入库与索引重建等基础管理功能。
  \item \textbf{结果解释需求。} 除返回相似度分数外，系统还应提供结构相似信息、文本辅助信息以及可解释性分析结果。
  \item \textbf{实验支撑需求。} 系统不仅服务于在线检索，还应支持离线评测、模式切换和实验结果复查。
\end{enumerate}

\subsection{性能性需求}

\begin{enumerate}
  \item \textbf{检索精度要求。} 系统需要在多类别工程图纸场景下保持较高的前列命中率与排序稳定性，以满足工程复用需求。
  \item \textbf{检索效率要求。} 在单次查询过程中，系统应尽量压缩图像加载、特征提取、向量召回与结果组织的总体耗时。
  \item \textbf{可扩展性要求。} 系统应支持不同检索模式、不同图库规模以及后续新模块的接入。
  \item \textbf{可复现性要求。} 系统应能够通过统一脚本完成建库、评测与结果导出，保证实验过程可追踪、可复查。
\end{enumerate}

\section{系统总体架构}

系统整体由图纸输入层、路线选择层、特征提取层、向量检索层、重排层和结果分析层组成。考虑到本文已经将方法重构为两条主线，系统在进入正式检索前会先根据当前模式决定走“YOLO 清洗路线”还是“无清洗注意力路线”。整体处理流程如下：

\begin{enumerate}
  \item 对输入图纸进行格式统一，并按当前模式决定是否执行 YOLO 清洗与保留掩码生成。
  \item 使用 CLIP 视觉编码器提取图像嵌入，并根据路线不同执行基础视觉编码或掩码引导的局部聚合增强。
  \item 从图纸中提取结构描述子，并在需要时提取 OCR 文本描述子。
  \item 将视觉向量送入 Qdrant 完成候选召回。
  \item 对候选结果进行结构重排，并在需要时于前排阶段引入标题栏字段级多模态增强。
  \item 输出最终检索结果，并支持 Grad-CAM 可视化和误检分析。
\end{enumerate}

从层次关系看，图纸输入层负责将用户上传图纸和图库图纸统一转化为系统可处理的 PNG 图像；路线选择层负责根据实验模式决定是否启用清洗、掩码聚合、结构重排和 OCR 辅助；特征提取层负责生成视觉向量、结构描述子与文本字段；向量检索层负责完成大规模候选召回；重排层负责对候选结果进行结构相似度和多模态分数修正；结果分析层则面向用户展示检索结果、统计信息和可解释性结果。通过这种分层方式，系统能够在保持在线检索流程简洁的同时，为论文实验提供较充分的模式切换能力。

\section{数据集组织与预处理设计}

根据任务书和开题报告中“构建一万张以上、二十类以上工程图纸数据集”的要求，本文将清理后的 \texttt{Dataset\_img3} 作为正式实验和系统建库数据。该数据集中的图纸均为 PNG 格式，来源于 DWG 图纸转换结果。由于原始转换过程中存在黑底、黑边和截屏边框等问题，系统在正式建库前首先进行统一背景处理，使图纸尽量保持白底线稿风格，从源头降低无关背景对视觉编码器的影响。

在数据组织上，系统按类别目录管理图纸文件，并在建库阶段写入图像路径、类别标签、文件名和结构描述等元数据。这样的组织方式有两点好处：一方面便于前端进行类别浏览和图库统计；另一方面便于离线评测脚本根据类别标签构造查询集合并计算 Recall、mAP、nDCG、FT、ST 和 ANMRR 等指标。对于后续扩展任务，系统也可以在保持目录结构不变的情况下继续加入新的图纸类别。

\section{双主线模式管理设计}

为了避免不同实验模式之间逻辑混杂，系统将检索模式抽象为若干可组合开关，包括是否启用 YOLO 清洗、是否启用掩码引导视觉增强、是否启用结构重排以及是否启用 OCR 辅助。论文中的 8 组实验模式均由这些开关组合得到。该设计使双主线实验具有统一实现基础，也便于在前端演示或离线评测时快速切换模式。

具体而言，\texttt{baseline} 仅使用 CLIP 视觉相似度；\texttt{cleaning\_only} 在视觉编码前增加清洗步骤；\texttt{masked\_pooling} 在清洗基础上加入局部视觉增强；\texttt{full\_model} 继续加入结构重排；\texttt{multimodal\_text} 在清洗和结构增强基础上加入 OCR 辅助。无清洗注意力路线则保留原图，不执行 YOLO 清洗，通过注意力式主体增强与结构重排构成 \texttt{attention\_visual} 和 \texttt{attention\_structure}，并在此基础上形成 \texttt{attention\_multimodal}。这种模式管理方式保证了论文实验结论能够直接追溯到系统运行逻辑。

\section{核心模块划分}

系统划分为以下核心模块：

\begin{enumerate}
  \item 图纸预处理模块：负责图纸格式统一、可选的 YOLO 清洗以及掩码生成。
  \item 特征提取模块：负责视觉嵌入、掩码引导聚合、OCR 文本提取和结构描述子生成。
  \item 向量检索模块：负责图纸向量入库、索引构建和候选召回。
  \item 结构重排模块：负责结构相似度计算与候选精排。
  \item 多模态排序模块：负责标题栏字段匹配和候选前排排序修正。
  \item 实验分析模块：负责正式评测、误检案例生成和可解释性可视化。
\end{enumerate}

\section{数据流与服务流}

从系统运行流程来看，输入图纸首先进入 FastAPI 服务，通过统一接口触发路线判断、特征提取与检索过程；随后视觉向量及相关元数据被送入 Qdrant 完成候选召回；最终结果再由业务层执行结构重排，并在启用多模态模式时叠加 OCR 辅助修正后返回。这种“向量召回 + 业务重排”的体系将高性能 ANN 检索与任务相关规则有效解耦，既有利于系统扩展，也有利于双主线实验的统一比较。

在离线实验流程中，系统不直接依赖在线接口返回结果，而是针对每一种模式分别生成图库特征缓存与查询特征缓存，再在统一评测脚本中完成相似度计算和指标统计。这样做可以避免在线参数切换带来的评测不一致问题，确保图库特征与查询特征始终来自同一检索模式。该流程虽然计算量更大，但更适合作为论文中的正式实验依据。

\section{系统部署思路}

从部署方式看，本文系统采用“算法服务 + 向量数据库 + 前端页面”的组织方式。其核心考虑在于，将模型推理、向量检索与页面展示三个层次解耦，可以降低系统维护复杂度，也有利于后续进行单独扩展。其中，FastAPI 负责统一对外提供检索与管理接口，Qdrant 负责存储向量及元数据，前端页面负责完成图纸上传、结果展示与统计浏览。该部署思路既满足了毕业设计的演示需求，也为实际应用场景中的服务化改造提供了基础框架。

\section{设计原则}

系统设计遵循以下原则：

\begin{enumerate}
  \item \textbf{视觉主导。} 检索结果首先由视觉向量召回保证全局可用性。
  \item \textbf{双路线兼容。} 系统既支持显式清洗路线，也支持无清洗注意力式增强路线，保证方法对比具有统一实现基础。
  \item \textbf{结构增强。} 结构先验作为工程图纸最重要的补充信息，用于提升 Top-K 排序质量。
  \item \textbf{多模态微调。} 标题栏字段仅作用于候选前排重排，避免 OCR 噪声对整体候选集合进行大范围改写。
  \item \textbf{工程可复现。} 所有实验均通过可运行脚本和服务接口完成，便于复现实验结果。
\end{enumerate}

\section{本章小结}

本章从系统角度给出了工程图纸检索平台的总体设计方案，明确了系统目标、整体架构、核心模块、数据流与设计原则。与传统单一路径系统不同，本文系统在架构层面即支持 YOLO 清洗路线与无清洗注意力路线两种模式，并通过统一的向量召回、结构重排和多模态增强框架完成结果输出。这种设计不仅保证了系统运行的完整性，也为后续方法对比实验提供了统一、可复现的实现基础。

```

## `chapters\04_method.tex`

```tex
\chapter{双主线工程图纸检索方法}

在系统总体结构确定之后，本文进一步从方法层面对工程图纸检索模型进行展开。本章重点回答两个问题：一是如何围绕工程图纸“结构主导、文本辅助”的特点构建更适合的检索路线；二是如何在统一实验框架下对显式清洗与无清洗主体增强两种思路进行可比较的设计。为此，本文将原有串行堆叠式流程重构为双主线方法体系，并在两条主线之上分别考察结构增强与 OCR 辅助的作用。

\section{方法设计思路}

针对工程图纸检索任务中存在的背景干扰、局部结构相似、标题栏文本信息分散等问题，本文不再沿用“清洗、局部增强、结构重排、OCR 融合”单链条串行堆叠的叙述方式，而是将整个方法重构为两条主线：
\begin{enumerate}
  \item \textbf{YOLO 清洗路线}：先通过目标检测清除无关区域，再在清洗结果上进行视觉建模、结构增强和 OCR 辅助检索；
  \item \textbf{无清洗注意力路线}：不显式执行 YOLO 清洗，而是直接利用掩码引导的局部注意力式建模突出主体结构，再叠加结构增强与 OCR 辅助检索。
\end{enumerate}

这样的设计可以更清楚地回答三个研究问题：第一，YOLO 清洗是否在新数据集上仍然有效；第二，不依赖显式清洗的注意力式建模是否能够取得更优结果；第三，两条路线在引入结构增强和 OCR 辅助后，哪一条路线的收益更稳定。

从算法组织角度看，本文方法包含“候选召回”和“候选精排”两个阶段。候选召回阶段主要依赖 CLIP 视觉向量，以保证系统能够在较大图库中快速获得相似候选；候选精排阶段再根据具体模式引入结构相似度和 OCR 字段相似度，对前列结果进行细粒度排序修正。这样的两阶段设计既保留了向量检索的效率优势，也为工程图纸中重要的局部结构和标题栏信息预留了融合空间。

\section{基础视觉检索框架}

本文首先构建基于 CLIP 视觉编码器的工程图纸检索基线。对于输入图纸 $I$，视觉编码器输出全局特征 $\bm{f}(I)$，归一化后得到检索向量：
\begin{equation}
  \bm{z}(I)=\frac{\bm{f}(I)}{\|\bm{f}(I)\|_2}.
  \label{eq:baseline-embedding-new}
\end{equation}

对于查询图纸 $I_q$ 与图库图纸 $I_i$，语义相似度定义为
\begin{equation}
  s_{\text{sem}}(I_q,I_i)=\bm{z}(I_q)^\top \bm{z}(I_i).
  \label{eq:semantic-score-new}
\end{equation}

该基线能够提供稳定的全局视觉召回能力，但对工程图纸中的细粒度结构布局、标题栏字段信息和局部主体区域关注不足，因此需要在其基础上进一步增强。

\section{YOLO 清洗路线}

\subsection{检测清洗与保留掩码}

在 YOLO 清洗路线中，首先利用目标检测模型识别图纸中的无关干扰区域，例如截图边框、注释噪声和非主体块区域。对于检测到的每个候选框，采用填白策略将对应区域置为背景，并同步生成保留掩码。设原图为 $I$，清洗后的图像为 $\widetilde{I}$，二值保留掩码为 $M$，则
\begin{equation}
  M(x,y)=
  \begin{cases}
    1, & (x,y)\ \text{属于保留区域},\\
    0, & (x,y)\ \text{属于清洗区域}.
  \end{cases}
\end{equation}

这里的掩码并不直接替代原始视觉编码，而是作为后续局部结构增强的先验信息。实验部分将专门验证：在新数据集已经完成黑底和黑边统一处理后，单独执行 YOLO 清洗是否仍然带来稳定收益。

\subsection{清洗路线的局部视觉增强}

在完成清洗后，本文进一步在视觉编码阶段使用掩码引导的 patch 聚合。设视觉编码器输出的 patch 特征为 $\{\bm{h}_p\}_{p=1}^{P}$，将图像级掩码映射到 patch 空间，得到 patch 权重 $m_p$，则局部增强后的特征写为
\begin{equation}
  w_p=\frac{m_p+\epsilon}{\sum_{j=1}^{P}(m_j+\epsilon)},
\end{equation}
\begin{equation}
  \bm{z}_{\text{mask}}=\sum_{p=1}^{P}w_p\bm{h}_p.
  \label{eq:masked-pooling-new}
\end{equation}

其中 $\epsilon$ 为平滑项。该机制可降低大面积空白区域与边框区域对全局特征的干扰，使模型更加关注轮廓、剖视、连接关系等主体结构信息。

\section{无清洗注意力路线}

\subsection{无清洗条件下的主体增强}

与 YOLO 清洗路线不同，无清洗注意力路线不显式对图纸做目标检测清除，而是直接从原始图纸出发，通过掩码引导的局部聚合机制突出主体结构区域。其核心思想是：即使不执行显式清洗，只要视觉编码阶段能够更聚焦主体区域，仍然可能取得优于清洗路线的检索性能。

在当前实现中，该路线采用 \textbf{mask-guided pooling / attention-like} 机制，即利用图像前景区域和局部结构分布构造轻量化的注意力式聚合，而非单独训练新的端到端空间注意力模型。也就是说，本文实现并验证的是“无清洗注意力式视觉增强路线”，而非严格意义上重新训练得到的 \texttt{spatial\_attention} 新模型。

\subsection{当前实现边界}

需要指出的是，当前实现支持将掩码引导聚合与局部 patch 表征结合，从而形成一条可运行的“无清洗注意力路线”。该方案能够作为论文中的对比路线，并用于回答“是否必须依赖显式清洗”的研究问题。

但如果要构建严格意义上的独立 \texttt{spatial\_attention} 模型，还需要单独训练与其匹配的权重文件。本文当前实验并未完成这一额外训练，因此在论文表述中应明确：本文验证的是“无清洗注意力式增强思路”，而非“重新训练的独立空间注意力模型”。

\section{结构相似性增强与重排序}

工程图纸检索与自然图像检索的重要区别在于：许多样本的局部纹理差异较小，但整体几何布局、投影轮廓和前景分布具有较强判别性。为此，本文设计了轻量级结构描述符 $\bm{g}(I)$，包括网格占用分布、水平投影、垂直投影和前景比例等信息。对于查询图纸和候选图纸，其结构相似度定义为
\begin{equation}
  s_{\text{str}}(I_q,I_i)=\operatorname{cos}\bigl(\bm{g}(I_q),\bm{g}(I_i)\bigr).
  \label{eq:structure-score-new}
\end{equation}

在候选重排序阶段，将视觉语义相似度与结构相似度融合，得到最终的结构增强评分：
\begin{equation}
  s_{\text{rerank}}=\lambda s_{\text{sem}}+(1-\lambda)s_{\text{str}},
  \label{eq:rerank-score-new}
\end{equation}
其中 $\lambda$ 为融合权重。

该设计有两个优点：一方面不会显著增加大规模图库召回成本，因为结构相似度主要作用于候选精排阶段；另一方面可以有效提升前列结果的结构一致性，尤其适用于零件图、装配图和工序图之间存在局部轮廓相似但布局关系不同的场景。

在实际计算中，结构描述符并不追求完整表达所有 CAD 几何细节，而是强调低成本、可复现和对检索排序有效。网格占用分布用于描述主体结构在图面中的空间位置，水平投影和垂直投影用于刻画轮廓延展方向，前景比例用于区分结构密集图纸与结构稀疏图纸。上述信息虽然简单，但能够补足 CLIP 全局向量在二维工程图纸布局表达上的不足。

\section{OCR 文字辅助检索}

\subsection{标题栏字段建模}

在视觉相似度与结构相似度之外，工程图纸还包含标题栏、图号、零件名称、比例和材料等强语义信息。本文没有将 OCR 输出简单视为一段普通文本，而是进一步面向标题栏抽取以下关键字段：
\begin{enumerate}
  \item 图号 $drawing\_no$；
  \item 零件名称 $part\_name$；
  \item 材料 $material$；
  \item 比例 $scale$；
  \item 数量 $quantity$。
\end{enumerate}

在此基础上，先计算通用 OCR 文本相似度 $s_{\text{ocr}}$，再计算字段级匹配得分：
\begin{equation}
  s_{\text{field}}=\alpha_1 s_{\text{id}}+\alpha_2 s_{\text{name}}+\alpha_3 s_{\text{mat}}+\alpha_4 s_{\text{scale}},
  \label{eq:field-score-new}
\end{equation}
其中 $s_{\text{id}}$、$s_{\text{name}}$、$s_{\text{mat}}$ 和 $s_{\text{scale}}$ 分别表示图号、零件名称、材料和比例的匹配得分，$\alpha_k$ 为对应权重。由于图号和零件名称通常具有更强区分度，因此其权重高于材料和比例。

\subsection{多模态融合策略}

在重排序基础上，定义 OCR 辅助后的最终得分为
\begin{equation}
  s_{\text{final}} = s_{\text{rerank}} + \beta_1 s_{\text{ocr}} + \beta_2 s_{\text{field}},
  \label{eq:final-score-new}
\end{equation}
其中 $\beta_1$ 和 $\beta_2$ 分别控制通用文本相似度和字段级匹配得分的影响。

为了避免 OCR 噪声破坏已有视觉排序，本文采用“候选重排而非全图库覆盖”的策略，仅在前列候选中启用 OCR 增强，并设置字段质量门控机制。由此，OCR 并不作为独立检索主干，而是作为视觉检索的辅助微调模块。

\section{双主线实验矩阵}

基于上述设计，本文将正式实验组织为 8 个模式：
\begin{enumerate}
  \item \texttt{baseline}；
  \item \texttt{cleaning\_only}；
  \item \texttt{masked\_pooling}；
  \item \texttt{full\_model}；
  \item \texttt{multimodal\_text}；
  \item \texttt{attention\_visual}；
  \item \texttt{attention\_structure}；
  \item \texttt{attention\_multimodal}。
\end{enumerate}

其中前 5 个模式构成 YOLO 清洗路线，后 4 个模式构成无清洗注意力路线，\texttt{baseline} 作为两条路线的共同参考点。这样的实验组织方式可以从方法设计上清楚划分“显式清洗”和“无清洗注意力式建模”两种思路，并进一步比较结构增强和 OCR 辅助在两条路线中的作用差异。

\section{方法流程总结}

综合以上设计，本文方法的完整流程可以概括为以下步骤：
\begin{enumerate}
  \item 输入查询图纸，并根据所选模式决定是否执行 YOLO 清洗；
  \item 提取查询图纸的 CLIP 视觉特征，在需要时执行掩码引导聚合或无清洗注意力式主体增强；
  \item 在 Qdrant 中利用视觉向量完成候选召回，获得初始 Top-K 候选集合；
  \item 对候选集合计算结构描述子相似度，并与视觉相似度融合得到结构重排分数；
  \item 当模式启用 OCR 时，抽取标题栏字段并计算字段级文本匹配分数；
  \item 综合视觉、结构和 OCR 分数得到最终排序结果，并输出相似图纸及辅助分析信息。
\end{enumerate}

该流程体现了本文“视觉为主、结构增强、文本辅助”的方法原则。其中视觉特征负责提供稳定召回，结构相似性负责改善前列候选的几何一致性，OCR 字段级信息则作为可扩展的辅助信号参与排序修正。通过双主线实验设计，本文能够进一步分析不同模块在真实工程图纸数据集上的实际贡献。

\section{可解释性分析}

为了辅助分析误检原因和局部关注区域，本文在检索结果分析中引入 Grad-CAM 可解释性工具。具体做法是对查询图与候选图的相似度响应执行梯度回传，在 patch 级特征上生成热力分布，从而观察模型究竟关注的是主体轮廓、剖视区域，还是无关边框与空白背景。

Grad-CAM 本身并不参与检索得分计算，但可为“局部注意力式建模是否真正聚焦主体结构”提供定性证据，也为后续误检分析与模型改进提供依据。

\section{本章小结}

本章围绕工程图纸检索任务给出了双主线方法设计。首先构建基于 CLIP 的基础视觉检索框架，然后分别提出 YOLO 清洗路线与无清洗注意力式视觉增强路线，并在此基础上引入结构重排序、OCR 字段级辅助和可解释性分析机制。通过这种组织方式，本文不仅完成了方法层面的模块化设计，也为下一章的正式实验提供了清晰的对比对象与评测维度。

```

## `chapters\05_experiments.tex`

```tex
\chapter{实验设计与结果分析}

为了客观评估本文所提出双主线方法的有效性，本章围绕数据集设置、评价指标、实验方案与结果分析展开论述。与早期基于小规模测试集的验证不同，本文正式实验全部建立在清理后的大规模数据集之上，并采用离线模式对齐评测，确保图库特征与查询特征在同一模式下生成，从而提高实验结论的可信度与答辩时的说服力。

\section{实验环境与数据集设置}

本文实验环境为 Windows + Python 3.10，特征提取与模型推理在 CUDA 设备上完成，视觉编码主干采用微调后的 CLIP 检索模型，向量索引部分使用 Qdrant 完成图库构建。本文正式实验不再采用旧版约 2454 张图纸的小规模测试集，而是全部切换至清理后的 \texttt{Dataset\_img3} 数据集。

新的正式数据集具有以下特点：
\begin{enumerate}
  \item 图库规模为 13880 张 PNG 工程图纸；
  \item 数据覆盖 22 个类别；
  \item 原始 DWG 转 PNG 过程中产生的黑底、黑边和截图边框干扰已被统一清理；
  \item 为避免“图库特征与查询模式不一致”带来的评估偏差，本文采用离线模式对齐评测，即在每种模式下分别对图库和查询图纸提取对应特征，再统一计算检索指标。
\end{enumerate}

在正式评测阶段，本文采用按类别均衡抽样策略，每类最多选取 20 张图纸作为查询样本，共得到 440 个正式查询。为了保证前列排序与深度排序指标都具有可比性，本文设置 Top-$K=10$，并在评测中额外采用 depth$=1000$ 的候选深度用于 FT、ST 和 ANMRR 计算。

\section{评价指标}

本文采用 Recall@K、mAP@10、nDCG@10、First Tier（FT）、Second Tier（ST）和 ANMRR 作为评价指标。设查询样本对应的相关集合为 $R_q$，前 $K$ 个检索结果为 $\mathcal{L}_q^K$，则 Recall@K 定义为
\begin{equation}
  \text{Recall@}K = \frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}} \mathbf{1}\!\left(\mathcal{L}_q^K \cap R_q \neq \emptyset\right).
\end{equation}

平均精度定义为
\begin{equation}
  AP@K(q)=\frac{1}{|R_q|}\sum_{k=1}^{K} P_q(k)\cdot rel_q(k),
\end{equation}
对所有查询取平均可得 mAP@10。

nDCG@10 反映前列排序质量，其定义为
\begin{equation}
  DCG@K = \sum_{k=1}^{K}\frac{2^{rel_q(k)}-1}{\log_2(k+1)},\qquad
  nDCG@K = \frac{DCG@K}{IDCG@K}.
\end{equation}

FT 和 ST 分别用于衡量类别规模相关深度下的召回能力，ANMRR 用于衡量相关样本整体排序位置，其值越小表示整体排序越优。由于工程图纸检索不仅关注 Top-1 命中，还关注前列候选的整体可用性，因此本文同时保留 Recall、mAP、nDCG 和 ANMRR 等多种指标。

\section{双主线实验设计}

与旧版“五阶段串行消融”不同，本文正式实验按两条主线组织：
\begin{enumerate}
  \item \textbf{YOLO 清洗路线}：包含 \texttt{baseline}、\texttt{cleaning\_only}、\texttt{masked\_pooling}、\texttt{full\_model} 和 \texttt{multimodal\_text} 五种模式；
  \item \textbf{无清洗注意力路线}：包含 \texttt{baseline}、\texttt{attention\_visual}、\texttt{attention\_structure} 和 \texttt{attention\_multimodal} 四种模式。
\end{enumerate}

其中：
\begin{enumerate}
  \item \texttt{baseline} 表示仅使用 CLIP 视觉语义相似度；
  \item \texttt{cleaning\_only} 表示仅执行 YOLO 清洗；
  \item \texttt{masked\_pooling} 表示在清洗路线中加入掩码引导的局部视觉增强；
  \item \texttt{full\_model} 表示在清洗路线中进一步加入结构重排序；
  \item \texttt{multimodal\_text} 表示在清洗路线中进一步加入 OCR 辅助检索；
  \item \texttt{attention\_visual} 表示不执行 YOLO 清洗，直接使用无清洗注意力式视觉增强；
  \item \texttt{attention\_structure} 表示在无清洗注意力路线中加入结构重排序；
  \item \texttt{attention\_multimodal} 表示在无清洗注意力路线中进一步加入 OCR 辅助检索。
\end{enumerate}

需要指出的是，当前“注意力路线”采用的是可运行的掩码引导聚合方案，用于验证无清洗主体增强思路；本文并未额外训练独立的 \texttt{spatial\_attention} 新模型。因此，实验结论针对的是“无清洗注意力式增强路线”的有效性，而非新权重模型本身。

\section{正式实验结果}

\subsection{8 组模式总体结果}

8 个模式的总体结果如\cref{tab:all-eight-modes}所示。

\begin{table}[htbp]
  \centering
  \caption{双主线 8 组正式实验结果}
  \label{tab:all-eight-modes}
  \begin{adjustbox}{width=\textwidth}
    \small
    \setlength{\tabcolsep}{3.5pt}
    \begin{tabular}{p{4.4cm}cccccccc}
      \toprule
      方法 & Recall@1 & Recall@5 & Recall@10 & mAP@10 & nDCG@10 & FT & ST & ANMRR \\
      \midrule
      Baseline & 0.8682 & 0.9023 & 0.9159 & 0.6191 & 0.6814 & 0.3676 & 0.4079 & 0.5773 \\
      YOLO Cleaning Only & 0.8682 & 0.9023 & 0.9159 & 0.6191 & 0.6814 & 0.3676 & 0.4079 & 0.5773 \\
      YOLO Cleaning + Masked Visual & 0.8614 & 0.8909 & 0.9045 & 0.5374 & 0.6128 & 0.3016 & 0.3414 & 0.6442 \\
      YOLO Cleaning + Structure & 0.8795 & 0.9114 & 0.9205 & 0.5629 & 0.6396 & 0.3026 & 0.3458 & 0.6409 \\
      YOLO Cleaning + Structure + OCR & 0.8773 & 0.9114 & 0.9205 & 0.5657 & 0.6421 & 0.3026 & 0.3458 & 0.6409 \\
      Attention Visual (No Cleaning) & 0.8750 & 0.9114 & 0.9205 & 0.6031 & 0.6714 & 0.3648 & 0.4099 & 0.5757 \\
      Attention + Structure (No Cleaning) & \textbf{0.8977} & \textbf{0.9318} & \textbf{0.9500} & \textbf{0.6141} & \textbf{0.6882} & 0.3655 & \textbf{0.4158} & 0.5727 \\
      Attention + Structure + OCR (No Cleaning) & 0.8705 & 0.9273 & 0.9477 & 0.6116 & 0.6845 & \textbf{0.3655} & \textbf{0.4158} & \textbf{0.5726} \\
      \bottomrule
    \end{tabular}
  \end{adjustbox}
  \vspace{0.2em}
  \parbox{\textwidth}{\footnotesize 注：表中粗体表示对应列中的最优结果。ANMRR 越小越好，其余指标越大越好。}
\end{table}

\subsection{YOLO 清洗路线结果}

YOLO 清洗路线的对比结果如\cref{tab:yolo-route-results}所示。

\begin{table}[htbp]
  \centering
  \caption{YOLO 清洗路线正式实验结果}
  \label{tab:yolo-route-results}
  \begin{adjustbox}{width=\textwidth}
    \small
    \setlength{\tabcolsep}{3.8pt}
    \begin{tabular}{p{4.4cm}cccccccc}
      \toprule
      方法 & Recall@1 & Recall@5 & Recall@10 & mAP@10 & nDCG@10 & FT & ST & ANMRR \\
      \midrule
      Baseline & 0.8682 & 0.9023 & 0.9159 & 0.6191 & 0.6814 & 0.3676 & 0.4079 & 0.5773 \\
      YOLO Cleaning Only & 0.8682 & 0.9023 & 0.9159 & 0.6191 & 0.6814 & 0.3676 & 0.4079 & 0.5773 \\
      YOLO Cleaning + Masked Visual & 0.8614 & 0.8909 & 0.9045 & 0.5374 & 0.6128 & 0.3016 & 0.3414 & 0.6442 \\
      YOLO Cleaning + Structure & 0.8795 & 0.9114 & 0.9205 & 0.5629 & 0.6396 & 0.3026 & 0.3458 & 0.6409 \\
      YOLO Cleaning + Structure + OCR & 0.8773 & 0.9114 & 0.9205 & 0.5657 & 0.6421 & 0.3026 & 0.3458 & 0.6409 \\
      \bottomrule
    \end{tabular}
  \end{adjustbox}
\end{table}

从\cref{tab:yolo-route-results}可以得到以下结论：
\begin{enumerate}
  \item \textbf{单独执行 YOLO 清洗没有带来增益。} \texttt{YOLO Cleaning Only} 与 \texttt{Baseline} 在 8 个指标上完全一致，说明在新数据集已经完成黑底与边缘统一处理后，显式清洗本身不再是主要增益来源。
  \item \textbf{仅加入清洗后的局部视觉增强反而会降低整体性能。} 清洗加局部视觉增强模式的 Recall@1 从 0.8682 降至 0.8614，mAP@10 从 0.6191 降至 0.5374，说明在当前数据条件下，清洗后的局部聚合可能削弱了原始全局语义信息。
  \item \textbf{结构重排序能够部分修复清洗路线的性能损失。} 清洗加结构增强模式将 Recall@1 提升到 0.8795，较清洗加局部视觉增强模式提高了 0.0181；nDCG@10 从 0.6128 提高到 0.6396，说明结构相似性信息对于工程图纸精排仍然有效。
  \item \textbf{OCR 对清洗路线的帮助有限。} 清洗加结构增强与 OCR 模式是在清洗加结构增强模式基础上继续叠加 OCR 后得到的结果。其 mAP@10 仅从 0.5629 提高到 0.5657，nDCG@10 从 0.6396 提高到 0.6421，但 Recall@1 从 0.8795 略降到 0.8773，说明 OCR 文本信息在该路线下未形成稳定的前列排序增益。
\end{enumerate}

\subsection{无清洗注意力路线结果}

无清洗注意力路线的对比结果如\cref{tab:attention-route-results}所示。

\begin{table}[htbp]
  \centering
  \caption{无清洗注意力路线正式实验结果}
  \label{tab:attention-route-results}
  \begin{adjustbox}{width=\textwidth}
    \small
    \setlength{\tabcolsep}{3.8pt}
    \begin{tabular}{p{4.8cm}cccccccc}
      \toprule
      方法 & Recall@1 & Recall@5 & Recall@10 & mAP@10 & nDCG@10 & FT & ST & ANMRR \\
      \midrule
      Baseline & 0.8682 & 0.9023 & 0.9159 & 0.6191 & 0.6814 & 0.3676 & 0.4079 & 0.5773 \\
      Attention Visual (No Cleaning) & 0.8750 & 0.9114 & 0.9205 & 0.6031 & 0.6714 & 0.3648 & 0.4099 & 0.5757 \\
      Attention + Structure (No Cleaning) & 0.8977 & 0.9318 & 0.9500 & 0.6141 & 0.6882 & 0.3655 & 0.4158 & 0.5727 \\
      Attention + Structure + OCR (No Cleaning) & 0.8705 & 0.9273 & 0.9477 & 0.6116 & 0.6845 & 0.3655 & 0.4158 & 0.5726 \\
      \bottomrule
    \end{tabular}
  \end{adjustbox}
\end{table}

由\cref{tab:attention-route-results}可知：
\begin{enumerate}
  \item \textbf{无清洗注意力式视觉增强整体优于纯基线。} 无清洗注意力视觉增强模式将 Recall@1 从 0.8682 提升到 0.8750，Recall@5 从 0.9023 提升到 0.9114，说明即使不做显式清洗，只要在视觉编码阶段强化主体区域，依然可以获得更好的检索效果。
  \item \textbf{结构增强在该路线中效果最稳定。} 无清洗注意力加结构增强模式在 Recall@1、Recall@5、Recall@10 和 nDCG@10 上均达到全表最优，分别为 0.8977、0.9318、0.9500 和 0.6882，说明“无清洗注意力式视觉增强 + 结构重排”是当前最有效的组合。
  \item \textbf{OCR 融合未进一步提升该路线的主指标。} 无清洗注意力加结构增强与 OCR 模式是在无清洗注意力加结构增强模式基础上继续叠加 OCR 后得到的结果。虽然其 ANMRR 略优，但 Recall@1 从 0.8977 下降到 0.8705，nDCG@10 也从 0.6882 降至 0.6845，说明 OCR 融合对该路线同样没有形成稳定的正增益。
\end{enumerate}

\subsection{两条主线的横向比较}

为了更清楚地比较两条主线，本文进一步从三个角度进行横向分析。

\subsubsection{YOLO 清洗是否仍然必要}

实验结果表明，\texttt{YOLO Cleaning Only} 与 \texttt{Baseline} 完全一致，而 \texttt{YOLO Cleaning + Masked Visual} 甚至在多数指标上明显下降。这表明，在已经完成黑底、黑边统一清理的大规模新数据集上，YOLO 清洗已不再是决定性能上限的关键因素。数据集预处理已消除了大量显式背景干扰，额外的检测清洗因而未能形成新的稳定收益。

\subsubsection{无清洗注意力路线是否更优}

从总体结果来看，答案是肯定的。相比清洗路线中的最佳视觉版本 \texttt{YOLO Cleaning + Structure}，\texttt{Attention + Structure (No Cleaning)} 在 Recall@1、Recall@5、Recall@10、mAP@10、nDCG@10 和 ANMRR 上均表现更优。这表明，对于当前任务而言，相较于在前处理阶段显式移除区域，在特征编码阶段直接增强主体结构表达更符合工程图纸检索的需求。

\subsubsection{OCR 辅助是否稳定有效}

本文在两条主线中都测试了 OCR 辅助检索。结果显示：
\begin{enumerate}
  \item 在 YOLO 清洗路线中，OCR 仅带来了极小的 mAP 和 nDCG 提升，但未提升 Recall@1；
  \item 在无清洗注意力路线中，OCR 反而使 Recall@1 和 nDCG@10 略有下降；
  \item 两条路线中 OCR 的增益都不稳定，说明当前 OCR 文本质量、标题栏模板差异和字段抽取置信度仍然限制了多模态策略的作用发挥。
\end{enumerate}

因此，在本文当前系统设置下，OCR 更适合作为可扩展增强模块，而非主要性能来源。

\section{补充分析与讨论}

\subsection{为什么清洗路线没有优势}

清洗路线在旧数据条件下本应更有潜力，因为旧版数据包含大量黑底、黑边和截图噪声。但在本文最终正式实验中，数据集已经过统一清理，显式 YOLO 清洗面对的是“已大幅净化后的图纸”，因此其边际收益明显减弱。此外，若清洗策略过强，还可能将部分结构细节和辅助线一并覆盖，从而降低局部形状辨识能力。这也是 \texttt{YOLO Cleaning + Masked Visual} 性能下降的重要原因。

\subsection{为什么无清洗注意力路线更适合当前任务}

工程图纸不同于自然图像，其主体通常由轮廓线、剖切线、尺寸线和局部结构组合构成。显式检测清洗更适合“前景与背景界限清晰”的场景，而对于线框类图纸，直接在视觉编码阶段引导模型关注主体结构，往往比先执行区域级清除更稳健。当前结果表明，无清洗注意力路线既保留了原图整体布局，又强化了局部主体区域，因此在新数据集上取得了最佳性能。

\subsection{OCR 模块的当前局限}

虽然工程图纸的标题栏字段具备较强语义信息，但 OCR 模块当前仍受到以下因素影响：
\begin{enumerate}
  \item 图号、零件名称和材料字段在不同模板中位置不统一；
  \item 部分旧图纸的标题栏分辨率较低或文本模糊；
  \item OCR 结果在候选重排阶段只能作为辅助信号，难以稳定压过强视觉匹配结果；
  \item 某些查询的标题栏信息与文件名、结构布局之间并不完全一致，容易引入轻微排序扰动。
\end{enumerate}

因此，OCR 多模态结果在本次实验中没有形成稳定正增益，这也是本文必须如实报告的结论之一。

\section{本章小结}

本文基于清理后的 \texttt{Dataset\_img3} 数据集，完成了双主线、8 组模式的正式实验。实验结果表明：在当前大规模新数据集上，YOLO 清洗本身不再构成主要收益来源；无清洗注意力式视觉增强路线整体优于 YOLO 清洗路线；结构重排序依然是最稳定的增强模块；OCR 文本辅助检索尚未在当前设置下形成稳定正增益。

综合来看，\texttt{Attention + Structure (No Cleaning)} 是当前表现最优的方案。该结果表明，对于已经完成统一清理的工程图纸数据集，直接增强主体结构感知能力较继续依赖显式清洗更适合作为本文的主要方法路线。由此，本章不仅完成了对 8 组模式的系统比较，也给出了后续系统实现与全文总结所依据的核心实验结论。

```

## `chapters\06_implementation.tex`

```tex
\chapter{系统实现}

在完成方法设计与正式实验之后，还需要说明本文提出的检索方案如何被实现为可运行、可复现、可演示的工程系统。因此，本章从工程实现角度对后端服务、索引构建、离线评测、前端交互以及系统可运行性进行说明，展示本文工作不仅停留在算法设计层面，而且已经形成较完整的原型系统。

\section{实现环境}

系统后端采用 Python 3.10 开发，模型推理运行于 CUDA 环境，服务框架为 FastAPI，向量数据库使用 Qdrant，OCR 模块采用 RapidOCR 方案。围绕本文所提出的方法体系，系统实现已完成以下几项关键功能建设：
\begin{enumerate}
  \item 支持面向 \texttt{Dataset\_img3} 的全图库重建；
  \item 支持独立建库脚本 \texttt{build\_index.py}，可显式指定 \texttt{CUDA} 设备并按需关闭 OCR；
  \item 支持离线模式对齐评测脚本，用于在不同模式下分别重建图库和查询特征；
  \item 支持前后端一体化检索服务与结果展示界面。
\end{enumerate}

\section{核心模块实现}

当前系统主要由以下模块构成：
\begin{enumerate}
  \item \texttt{feature\_service.py}：负责视觉特征提取、掩码引导聚合、结构描述符构建以及 OCR 辅助所需的基础特征组织；
  \item \texttt{cleaning\_service.py}：负责基于 YOLO 的图纸清洗和保留掩码生成；
  \item \texttt{ocr\_service.py}：负责 OCR 文本抽取、标题栏字段结构化、字段相似度计算；
  \item \texttt{qdrant\_db.py}：负责图库向量写入、候选召回、结构重排序和在线检索结果组织；
  \item \texttt{build\_index.py}：负责 GPU 化独立建库，避免在服务启动阶段直接触发高负载全量建库；
  \item \texttt{evaluate\_visual\_offline.py}：负责视觉主线离线对齐评测；
  \item \texttt{evaluate\_multimodal\_offline.py}：负责 OCR 辅助模式离线评测；
  \item \texttt{visualize\_pair\_similarity.py}：负责相似案例分析与 Grad-CAM 可视化。
\end{enumerate}

各模块之间通过配置文件和统一服务类进行衔接。配置文件负责管理数据集路径、模型路径、向量库参数和检索模式参数；服务类负责封装图像读取、特征提取、OCR 调用和结果组装逻辑。这样可以避免算法逻辑直接散落在前端接口中，也便于后续对某一模块进行替换。例如，当后续训练出独立的 \texttt{spatial\_attention} 权重后，只需要在特征提取模块中替换对应的主体增强实现，而不必改动向量数据库和前端展示逻辑。

\section{关键接口与服务组织}

为了使系统既能支持在线演示，也能支撑毕业设计中的实验复现，本文在 FastAPI 服务中设计了若干核心接口。其组织方式遵循“检索接口、管理接口、监控接口”三类职责划分。

\begin{enumerate}
  \item \textbf{检索接口。} 以 \texttt{/search} 为核心，用于接收用户上传的查询图纸，完成特征提取、向量召回和结果返回。
  \item \textbf{基础状态接口。} 包括 \texttt{/health}、\texttt{/ready}、\texttt{/live} 和 \texttt{/metrics}，分别用于健康检查、就绪检测、存活检测和监控指标输出。
  \item \textbf{图库管理接口。} 包括 \texttt{/stats}、\texttt{/api/images}、\texttt{/api/categories}、\texttt{/api/initialize}、\texttt{/api/rebuild} 等，用于图库浏览、图像管理与索引维护。
\end{enumerate}

通过上述接口组织方式，系统能够同时满足答辩演示中的交互需求和工程运行中的可维护性需求，也使系统实现与实验脚本之间形成了相对统一的接口层。

\section{索引构建实现}

早期版本中，建库流程与服务启动过程耦合较紧，且 OCR 文本抽取阶段容易造成较大的 CPU 开销。为解决这一问题，本文重新设计了独立建库流程。当前建库逻辑分为三步：
\begin{enumerate}
  \item 加载配置与模型，并显式确认视觉编码设备为 CUDA；
  \item 根据运行参数决定是否启用 YOLO 清洗和 OCR；
  \item 扫描图纸目录，按批次提取视觉特征并写入 Qdrant。
\end{enumerate}

其中，为提高 GPU 利用率，本文进一步将视觉特征提取改写为批量前向方式，不再逐张图纸单独执行 CLIP 编码。这样既提高了显卡利用效率，也降低了全量重建时的无效 CPU 占用。在正式实验前，本文已基于 \texttt{CUDA + 关闭 OCR} 的方式完成 \texttt{Dataset\_img3} 全图库重建，最终索引数量达到 13880/13880。

在工程实践中，OCR 是全量建库阶段较容易造成 CPU 压力的环节。为避免系统在首次建库时因 OCR 抽取而长时间占用 CPU，本文将视觉索引构建与 OCR 辅助评测进行解耦：视觉索引优先使用 GPU 批量完成，OCR 仅在多模态实验或需要字段辅助的候选阶段启用。该设计既保证了主检索链路的可运行性，也使不同实验模式之间的计算开销更加可控。

\section{在线检索流程实现}

系统在线检索流程如下：
\begin{enumerate}
  \item 用户上传查询图纸；
  \item 系统根据当前检索模式决定是否启用 YOLO 清洗、掩码引导聚合和结构重排序；
  \item 提取查询图纸的视觉特征与结构描述符；
  \item 在 Qdrant 中完成向量召回；
  \item 对候选结果进行结构增强与必要的 OCR 辅助重排序；
  \item 返回相似图纸路径、排序分数、结构相似度及文本辅助分析结果。
\end{enumerate}

该流程兼容论文中的双主线实验设计，即既可以运行 YOLO 清洗路线，也可以运行无清洗注意力路线。

\section{离线正式评测实现}

为了避免“图库特征来自一种模式，而查询特征来自另一种模式”导致的评测偏差，本文没有继续使用旧版仅依赖在线接口切换参数的实验方式，而是重新实现了离线模式对齐评测方案。

具体而言：
\begin{enumerate}
  \item \texttt{evaluate\_visual\_offline.py} 负责对视觉主线进行离线评测。脚本会在不同模式下分别构建图库特征缓存和查询特征，再统一计算 Recall@K、mAP@10、nDCG@10、FT、ST 和 ANMRR；
  \item \texttt{evaluate\_multimodal\_offline.py} 在视觉缓存基础上，对候选结果执行 OCR 辅助重排，从而评测 \texttt{multimodal\_text} 和 \texttt{attention\_multimodal} 两个模式；
  \item 评测脚本最终输出 JSON、CSV 和 Markdown 三种格式结果，便于论文表格回填与结果复查。
\end{enumerate}

基于该离线评测机制，本文完成了双主线 8 个模式的正式实验，并将所得结果统一回填至论文之中。

与在线检索相比，离线评测更强调公平性和可复现性。在线服务需要优先保证交互响应，而离线评测需要保证每一组实验的图库特征、查询特征和排序逻辑严格一致。因此，本文将正式指标计算全部放在线下脚本中完成，并保留原始 JSON 结果文件。这样既便于论文表格回填，也便于在答辩或复查阶段追溯每一组数值的来源。

\section{前端与交互实现}

系统前端采用 HTML 页面结合后端接口实现，支持图纸上传、相似图纸检索、图库浏览、统计信息查看和分类结果展示等功能。相较于早期简化界面，当前前端已恢复为完整展示版本，可较为完整地呈现系统功能与检索结果。用户可在页面中直接查看图库规模、检索结果、类别统计和相似样本信息，后端则负责完成特征提取、排序计算与结果返回。

在交互流程上，用户首先通过页面上传查询图纸，系统返回相似图纸缩略图、文件路径、相似度分数和相关类别信息。对于图库管理部分，页面提供类别浏览与统计入口，使用户能够了解当前图库规模和类别分布。对于系统维护部分，前端通过后端管理接口触发初始化或重建索引操作。这样的设计能够覆盖毕业设计演示中常见的三类场景：单张图纸检索、图库状态展示和系统维护操作。

\section{异常处理与运行保障}

为了提高系统运行稳定性，后端在图像读取、模型推理、向量库访问和结果返回等环节均设置了基础异常处理。当上传文件格式不符合要求、向量库尚未完成初始化或模型推理过程出现异常时，系统会返回明确的错误信息，避免前端页面长时间无响应。同时，健康检查接口可以用于判断服务是否正常启动，就绪接口可以用于判断向量库与模型是否已经完成加载。

在资源使用方面，本文将全量建库与在线查询分离。全量建库通过独立脚本执行，适合在 GPU 可用且系统空闲时运行；在线查询则只处理单张或少量图纸，避免与大规模索引构建过程相互干扰。这种实现方式能够降低系统演示过程中的不确定性，也更符合实际部署中“离线建库、在线查询”的常见工作模式。

\section{工程可运行性}

截至论文撰写完成阶段，系统已经具备较完整的工程可运行性，主要体现在：
\begin{enumerate}
  \item 已完成新数据集清理与全图库索引重建；
  \item 已支持 GPU 化建库与批量视觉编码；
  \item 已支持双主线在线检索与离线评测；
  \item 已支持 OCR 辅助检索、结构重排序和 Grad-CAM 可解释性分析；
  \item 已形成前后端联动的完整系统原型。
\end{enumerate}

上述结果表明，本文工作并未停留于算法设计层面，而是进一步形成了较为完整的系统实现与实验复现基础。

\section{本章小结}

本章从工程实现角度介绍了本文方法的落地方式。通过将数据清理、GPU 化建库、视觉特征提取、结构增强、OCR 辅助检索、离线评测和前端展示统一组织为模块化系统，本文不仅实现了工程图纸检索方法的完整运行链路，也为后续扩展独立注意力模型、优化 OCR 融合策略和部署实际系统提供了明确的实现基础。

```

## `chapters\07_conclusion.tex`

```tex
\chapter{总结与展望}

在完成方法构建、系统实现与正式实验之后，本文需要对整体研究工作进行归纳，并对当前工作的边界与后续发展方向作出总结。因此，本章从工作总结、不足分析与未来展望三个方面，对全文研究内容进行收束，以形成完整的论文结论。

\section{工作总结}

本文围绕工程图纸相似检索任务，完成了从数据清理、系统实现、方法设计到正式评测的完整研究流程。与旧版工作相比，本文将原先基于小规模旧测试集的实验体系重构为基于 \texttt{Dataset\_img3} 的双主线实验体系，并在 13880 张图纸、22 个类别、440 个正式查询样本上完成了新的正式实验。

在方法上，本文不再采用“清洗、局部增强、结构重排、OCR 融合”单链条递进式写法，而是构建了两条主线：
\begin{enumerate}
  \item YOLO 清洗路线；
  \item 无清洗注意力式视觉增强路线。
\end{enumerate}
在两条主线之上，本文分别考察了结构相似性增强与 OCR 文字辅助检索的作用，从而形成了 8 组正式实验模式。

实验结果表明：
\begin{enumerate}
  \item 在经过统一黑底和边框清理的大规模新数据集上，\texttt{YOLO Cleaning Only} 与 \texttt{Baseline} 指标一致，说明单独执行 YOLO 清洗已不再带来稳定收益；
  \item 结构重排序仍然是最稳定有效的增强模块；
  \item 无清洗注意力路线整体优于 YOLO 清洗路线；
  \item 当前最佳模式为 \texttt{Attention + Structure (No Cleaning)}，其 Recall@1 达到 0.8977，Recall@5 达到 0.9318，Recall@10 达到 0.9500，nDCG@10 达到 0.6882；
  \item OCR 辅助检索在两条路线中均未形成稳定正增益，说明当前标题栏字段抽取与文本融合策略仍有优化空间。
\end{enumerate}

因此，本文得到的核心结论是：对于已经完成统一预处理的工程图纸大规模数据集，\textbf{在视觉编码阶段直接增强主体结构表达，相较于继续依赖显式清洗，更适合作为工程图纸检索的主要技术路线}。

\section{不足分析}

尽管本文完成了方法重构和正式实验，但仍存在以下不足：
\begin{enumerate}
  \item 当前“注意力路线”采用的是可运行的 mask-guided pooling / attention-like 方案，尚未单独训练独立的 \texttt{spatial\_attention} 新模型；
  \item OCR 标题栏字段抽取仍然受到模板差异、分辨率和文本质量影响，因此多模态结果不够稳定；
  \item 结构描述符虽然能够改善候选重排序，但对零件图、装配图和工序图之间更高层级的语义差异建模仍不充分；
  \item 目前的多模态融合仍属于后处理式重排，尚未实现视觉与文本信息的端到端联合优化。
\end{enumerate}

\section{未来展望}

后续工作可从以下几个方向继续展开：
\begin{enumerate}
  \item 单独训练与当前路线匹配的 \texttt{spatial\_attention} 模型，进一步验证“无清洗注意力路线”在端到端训练条件下的潜力；
  \item 引入更稳定的标题栏检测与字段解析模型，提高图号、零件名称和材料字段的抽取精度；
  \item 将 OCR 融合从后处理重排升级为训练期联合约束，使文本语义信息更早参与特征学习；
  \item 引入更精细的图纸层级标签和部件级标注，增强系统对装配图、零件图和工序图之间语义层次差异的辨别能力；
  \item 继续扩充标准化工程图纸数据集，提升实验结论的泛化性与学术说服力。
\end{enumerate}

总体而言，本文围绕工程图纸相似检索问题完成了较系统的研究与实现工作，既给出了可运行的工程方案，也形成了基于大规模清理数据集的正式实验结论。虽然当前工作仍存在若干局限，但相关方法、系统与实验框架已经为后续深入研究奠定了较为扎实的基础。

```

## `chapters\abstract.tex`

```tex
\chapter*{摘要}
\addcontentsline{toc}{chapter}{摘要}

工程图纸检索是制造业设计复用、工艺继承和知识管理中的关键问题。传统基于人工分类、图号索引或关键词匹配的检索方式，难以满足大规模工程图纸库中按结构相似性和语义相关性快速检索历史图纸的需求。针对工程图纸线条密集、结构主导、文本信息与图形信息紧密耦合的特点，本文以 CLIP 检索模型为基础，研究并实现了一套面向工程图纸的相似检索方法与系统。

在方法设计上，本文不再沿用单链条模块堆叠的方式，而是将整体方案重构为两条主线：一条是 YOLO 清洗路线，另一条是无清洗注意力式视觉增强路线。在此基础上，本文进一步分别考察结构相似性增强和 OCR 文字辅助检索对两条主线的影响。针对视觉表征部分，本文构建了基于 CLIP 的基础视觉检索框架，并设计掩码引导的局部主体增强机制，以提升模型对工程图纸主体结构区域的关注能力。针对候选精排阶段，本文设计了轻量级结构描述符和结构重排序策略，用于增强对轮廓布局和投影分布的区分能力。针对标题栏信息，本文实现了字段级 OCR 文本抽取与辅助检索机制，用于提取图号、零件名称、材料和比例等强语义字段，并将其用于候选排序修正。为了提高方法的可解释性，本文还实现了基于 Grad-CAM 的可视化分析工具，用于展示模型在检索判定中所关注的局部区域。

在实验设计上，本文使用清理后的 \texttt{Dataset\_img3} 数据集进行正式评测。该数据集包含 13880 张 PNG 工程图纸，覆盖 22 个类别。正式实验采用按类别均衡抽样方式，共选取 440 个查询样本，并以 Recall@K、mAP@10、nDCG@10、FT、ST 和 ANMRR 作为评价指标。实验结果表明，在当前数据条件下，单独引入 YOLO 清洗并未带来稳定收益；无清洗注意力路线整体优于 YOLO 清洗路线；结构重排序仍然是最稳定有效的增强模块；当前最佳模式为 \texttt{Attention + Structure (No Cleaning)}，其 Recall@1 达到 0.8977，Recall@5 达到 0.9318，Recall@10 达到 0.9500，nDCG@10 达到 0.6882。相比之下，OCR 文字辅助检索在两条主线中的增益均不稳定，说明其仍然更适合作为可扩展增强模块而非当前阶段的主要性能来源。

本文完成了从数据清理、GPU 化建库、离线对齐评测到前后端检索服务的完整工程实现，为后续工程图纸检索研究与系统部署提供了较为完整的实验基础与实现方案。

\noindent\textbf{关键词：} 工程图纸检索，CLIP，参数高效微调，无清洗注意力路线，结构重排序，OCR，多模态检索

\chapter*{ABSTRACT}
\addcontentsline{toc}{chapter}{ABSTRACT}

Engineering drawing retrieval is an important problem in design reuse, process inheritance, and knowledge management in manufacturing. Traditional retrieval methods based on manual categorization, drawing numbers, or keyword matching are insufficient for large-scale engineering drawing repositories, where users need to retrieve historically relevant drawings according to structural similarity and semantic relevance. To address this problem, this thesis develops and implements a CLIP-based retrieval framework for engineering drawings.

Instead of describing the method as a single serial pipeline, this thesis reorganizes the overall solution into two main routes: a YOLO-cleaning route and a no-cleaning attention-style visual enhancement route. On top of these two routes, structure-aware enhancement and OCR-assisted multimodal retrieval are further studied. For visual representation, a CLIP-based baseline retrieval framework is built, and a mask-guided local enhancement strategy is introduced to focus more on the main structural regions of engineering drawings. For candidate refinement, a lightweight structural descriptor and a structure-aware re-ranking strategy are designed to better distinguish layout and projection patterns. For title-block semantics, a field-aware OCR-assisted retrieval mechanism is implemented to extract strong semantic cues such as drawing number, part name, material, and scale. In addition, a Grad-CAM-based visualization tool is implemented to improve interpretability and support failure-case analysis.

Formal experiments are conducted on the cleaned \texttt{Dataset\_img3}, which contains 13,880 PNG engineering drawings from 22 categories. A total of 440 balanced query samples are used for evaluation, and Recall@K, mAP@10, nDCG@10, FT, ST, and ANMRR are adopted as the evaluation metrics. The results show that, under the cleaned large-scale dataset setting, YOLO cleaning alone does not bring stable gains. The no-cleaning attention-style route consistently outperforms the YOLO-cleaning route, while structure-aware re-ranking remains the most effective enhancement module. The best-performing setting is \texttt{Attention + Structure (No Cleaning)}, which achieves Recall@1 of 0.8977, Recall@5 of 0.9318, Recall@10 of 0.9500, and nDCG@10 of 0.6882. In contrast, OCR-assisted multimodal retrieval does not produce stable gains on either route, indicating that OCR should currently be viewed as an auxiliary enhancement rather than the main source of performance improvement.

Finally, this thesis completes a full engineering workflow including dataset cleaning, GPU-based indexing, offline aligned evaluation, and an end-to-end retrieval service, which provides a practical basis for follow-up research and deployment of engineering drawing retrieval systems.

\noindent\textbf{KEY WORDS:} Engineering Drawing Retrieval; CLIP; Parameter-Efficient Fine-Tuning; No-Cleaning Attention Route; Structure-Aware Re-Ranking; OCR; Multimodal Retrieval

```

## `chapters\acknowledgements.tex`

```tex
\chapter*{致谢}
\addcontentsline{toc}{chapter}{致谢}

本课题的完成离不开学校、学院、指导教师以及同学和家人的支持与帮助。首先，衷心感谢指导教师宋丹老师在课题选题、技术路线设计、实验组织和论文写作过程中给予的持续指导与耐心帮助。老师严谨的治学态度和细致的修改意见，使我在毕业设计过程中不断完善研究思路并提升工程实现质量。

同时，感谢自动化学院提供的学习环境与实验条件，使本课题能够顺利完成数据清理、模型训练、系统开发和实验验证等工作。感谢在项目实施过程中给予我建议和帮助的同学与朋友，你们在数据整理、系统测试和结果讨论中的支持，对本论文的完成具有重要意义。

最后，感谢家人在毕业设计期间给予的理解、鼓励与支持。正是因为这些关心与帮助，我才能够较为顺利地完成本科阶段的毕业设计（论文）工作。在此一并表示诚挚的谢意。

```

## `chapters\appendix.tex`

```tex
\chapter*{附录}
\addcontentsline{toc}{chapter}{附录}

\section*{附录A 主要系统接口说明}
\addcontentsline{toc}{section}{附录A 主要系统接口说明}

结合系统实现与演示需求，本文工程原型对外提供了若干关键接口。其主要接口及功能说明如\cref{tab:appendix-api}所示。

\begin{table}[H]
  \centering
  \caption{主要系统接口说明}
  \label{tab:appendix-api}
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabularx}{\textwidth}{lllX}
    \toprule
    序号 & 方法 & 路径 & 功能说明 \\
    \midrule
    1 & GET & \texttt{/} & 返回系统主页，用于前端检索界面展示。 \\
    2 & GET & \texttt{/health} & 返回服务健康状态，用于基础可用性检查。 \\
    3 & GET & \texttt{/metrics} & 返回系统监控指标，用于性能统计与运维观测。 \\
    4 & POST & \texttt{/search} & 上传查询图纸并执行相似检索，返回候选结果与相似度分数。 \\
    5 & GET & \texttt{/stats} & 获取当前图库规模、类别统计等信息。 \\
    6 & GET/POST & \texttt{/api/images} & 支持图库图像查询与新增写入。 \\
    7 & POST & \texttt{/api/initialize} & 执行系统初始化与基础数据准备。 \\
    8 & POST & \texttt{/api/rebuild} & 触发图库索引重建。 \\
    9 & GET & \texttt{/api/categories} & 返回类别列表与分类浏览信息。 \\
    \bottomrule
  \end{tabularx}
\end{table}

\section*{附录B 正式实验模式说明}
\addcontentsline{toc}{section}{附录B 正式实验模式说明}

为保证双主线实验过程的可复查性，本文对正式实验所涉及的 8 个模式进行归纳，具体如\cref{tab:appendix-modes}所示。

\begin{table}[H]
  \centering
  \caption{正式实验模式说明}
  \label{tab:appendix-modes}
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabularx}{\textwidth}{llX}
    \toprule
    模式标识 & 对应名称 & 说明 \\
    \midrule
    \texttt{baseline} & 基础视觉检索 & 仅采用 CLIP 视觉向量进行相似度召回。 \\
    \texttt{cleaning\_only} & YOLO Cleaning Only & 在基线基础上增加显式清洗，不进行结构重排。 \\
    \texttt{masked\_pooling} & YOLO Cleaning + Masked Visual & 在清洗后加入掩码引导的局部视觉增强。 \\
    \texttt{full\_model} & YOLO Cleaning + Structure & 在清洗路线中进一步引入结构重排序。 \\
    \texttt{multimodal\_text} & YOLO Cleaning + Structure + OCR & 在清洗与结构增强基础上叠加 OCR 辅助重排。 \\
    \texttt{attention\_visual} & Attention Visual (No Cleaning) & 不执行显式清洗，直接进行无清洗注意力式视觉增强。 \\
    \texttt{attention\_structure} & Attention + Structure (No Cleaning) & 在无清洗注意力路线中引入结构重排序。 \\
    \texttt{attention\_multimodal} & Attention + Structure + OCR (No Cleaning) & 在无清洗注意力与结构重排基础上叠加 OCR 辅助重排。 \\
    \bottomrule
  \end{tabularx}
\end{table}

```

## `chapters\declaration.tex`

```tex
\clearpage
\chapter*{独创性声明}
\thispagestyle{empty}

本人声明：所呈交的毕业设计（论文），是本人在指导教师指导下进行研究工作所取得的成果。除文中已经注明引用的内容外，本毕业设计（论文）中不包含任何他人已经发表或撰写过的研究成果。对本毕业设计（论文）所涉及的研究工作做出贡献的其他个人和集体，均已在文中作了明确说明。本毕业设计（论文）原创性声明的法律责任由本人承担。

\vspace{2cm}
\noindent 论文作者签名：\hspace{4cm}

\vspace{1cm}
\noindent 年\hspace{1cm}月\hspace{1cm}日

\vspace{2.2cm}
本人声明：本毕业设计（论文）是本人指导学生完成的研究成果，已经审阅过论文的全部内容。

\vspace{2cm}
\noindent 论文指导教师签名：\hspace{3.2cm}

\vspace{1cm}
\noindent 年\hspace{1cm}月\hspace{1cm}日
\clearpage

```

## `chapters\inner_cover.tex`

```tex
\clearpage
\thispagestyle{empty}
\begin{center}
  \vspace*{2.2cm}
  {\zihao{-0}\bfseries 天津大学本科生毕业设计（论文）\par}
  \vspace{2.2cm}
  {\zihao{2}\bfseries \thesistitle\par}
  \vspace{2.4cm}
\end{center}

\renewcommand{\arraystretch}{1.6}
\begin{center}
  \begin{tabular}{p{3.5cm}p{8.5cm}}
    学\qquad 院： & \schoolname \\
    专\qquad 业： & \majorname \\
    年\qquad 级： & \gradeinfo \\
    姓\qquad 名： & \studentname \\
    学\qquad 号： & \studentid \\
    指导教师： & \advisorname \\
  \end{tabular}
\end{center}

\vfill
\begin{center}
  {\zihao{-3}\finishdate\par}
\end{center}
\clearpage

```
