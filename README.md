# Representations as Language: An Information Theoretic Framework for Interpretability

[view paper](https://arxiv.org/pdf/2406.02449)

Initial code is available! To run a model on the CFQ dataset run ```python code/train.py``` from this directory. Additional documentation for general entropy estimation and support for setting up dependencies coming soon.

Additional visuals for experiments from the paper can be found in visuals. Sorted by dataset.

Large scale neural models show impressive performance across a wide array of linguistic tasks. Despite this they remain, largely, black-boxes - inducing vector-representations of their input that prove difficult to interpret. We introduce a novel approach to interpretability that looks at the mapping a model learns from sentences to representations as a kind of language in its own right. In doing so we introduce a set of information-theoretic measures that quantify how structured a model's representations are with respect to its input, and when during training that structure arises. Our measures are fast to compute, grounded in linguistic theory, and can predict which models will generalise best based on their representations. We use these measures to describe two distinct phases of training a transformer: an initial phase of in-distribution learning which reduces task loss, then a second stage where representations becoming robust to noise. Generalisation performance begins to increase during this second phase, drawing a link between generalisation and robustness to noise. Finally we look at how model size affects the structure of the representational space, showing that larger models ultimately compress their representations more than their smaller counterparts.

