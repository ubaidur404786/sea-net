# figures/

Put the report's own images here (PDF preferred, PNG is fine).

Right now every figure in the report is a grey `\figph{...}` placeholder box that
describes the picture to draw. When an image is ready, drop it in this folder and
replace the placeholder line:

```latex
% before
\figph{Overall SEA-Net architecture: ...}

% after
\includegraphics[width=\linewidth]{figures/seanet_architecture}
```

Leave the file extension off in `\includegraphics` — LaTeX then picks the `.pdf`
version, which stays sharp at any zoom.

Figures already produced by the code (`python main.py paper`) do **not** need to be
copied here. They can be used straight from the project with their normal path, e.g.
`results/paper_figures/01_main_figures/pareto_web_acc_vs_params`, because
`preamble.tex` adds the project root to the image search path.

Naming suggestion, so the list stays readable:

```
fig01_teaser.pdf
fig02_mil_view.pdf
fig03_aopcr_curve.pdf
fig04_seanet_architecture.pdf     <- the new architecture diagram
fig05_multiscale_block.pdf
fig06_pooling_architecture.pdf
fig07_pipeline.pdf
...
```
