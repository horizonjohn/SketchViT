# SketchMLP
Fine-Grained Sketch-Based Image Retrieval


## 1. Experimental results

<div class="center">

| Column 1 | Column 2      |
|:--------:| -------------:|
| centered 文本居中 | right-aligned 文本居右 |

</div>

\begin{table}[]
\caption{Comparative results of our model against other methods on QMUL-Chair-V2}
\label{tab:Compare_Chair}
\begin{tabular}{c|ccc}
\hline
\multirow{2}{*}{Methods} & \multicolumn{3}{c}{QMUL-Chair-V2(\%)} \\ \cline{2-4} 
 & \multicolumn{1}{c|}{Acc.@1} & \multicolumn{1}{c|}{Acc.@5} & Acc.@10 \\ \hline
Triplet-SN \cite{yu2016sketch} & \multicolumn{1}{c|}{33.75} & \multicolumn{1}{c|}{65.94} & 79.26 \\
Triplet-Att-SN \cite{song2017deep} & \multicolumn{1}{c|}{37.15} & \multicolumn{1}{c|}{67.80} & 82.97 \\
OnTheFly \cite{bhunia2020sketch} & \multicolumn{1}{c|}{39.01} & \multicolumn{1}{c|}{75.85} & 87.00 \\
CMHM-SBIR \cite{sain2020cross} & \multicolumn{1}{c|}{51.70} & \multicolumn{1}{c|}{80.50} & 88.85 \\
SketchAA \cite{yang2021sketchaa} & \multicolumn{1}{c|}{52.89} & \multicolumn{1}{c|}{73.80} & 94.88 \\
Semi-Sup \cite{bhunia2021more} & \multicolumn{1}{c|}{60.20} & \multicolumn{1}{c|}{78.10} & 90.81 \\
StyleMeUp \cite{sain2021stylemeup} & \multicolumn{1}{c|}{62.86} & \multicolumn{1}{c|}{79.60} & 91.14 \\
NT-SBIR \cite{bhunia2022sketching} & \multicolumn{1}{c|}{64.80} & \multicolumn{1}{c|}{79.10} & - \\ \hdashline
\textbf{Ours} & \multicolumn{1}{c|}{\textbf{67.62}} & \multicolumn{1}{c|}{\textbf{91.10}} & \textbf{95.37} \\ \hline
\end{tabular}
\end{table}
