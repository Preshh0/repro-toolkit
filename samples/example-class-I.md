# Sentiment Classification with a Small Transformer

## Abstract
We present a compact transformer for sentiment classification and release everything needed to reproduce it.

## Introduction
Sentiment analysis is widely used. This report documents our model, data and results in full. We ask whether a 6-layer transformer encoder can match a larger pre-trained baseline on binary sentiment classification of English product reviews. The primary success metric, accuracy on the held-out test set, was fixed before any training began. Scope: English reviews from one retail platform, collected between 2021 and 2023.

## Related Work

The motivation for this work comes from practical deployments where teams repeatedly rebuild similar systems without sharing details. In our experience, the main obstacles to reuse are missing documentation, undocumented preprocessing and unclear evaluation protocols. We therefore describe each step of the pipeline carefully and explain why alternative designs were rejected. The remainder of this report follows the usual structure: we describe the problem setting and the related literature, then the method and the data, and finally the results, a discussion of the limitations and our conclusions. Throughout the report we try to be explicit about what was measured, how it was measured and what we could not measure, so that readers can judge the strength of each claim for themselves. 

The motivation for this work comes from practical deployments where teams repeatedly rebuild similar systems without sharing details. In our experience, the main obstacles to reuse are missing documentation, undocumented preprocessing and unclear evaluation protocols. We therefore describe each step of the pipeline carefully and explain why alternative designs were rejected. The remainder of this report follows the usual structure: we describe the problem setting and the related literature, then the method and the data, and finally the results, a discussion of the limitations and our conclusions. Throughout the report we try to be explicit about what was measured, how it was measured and what we could not measure, so that readers can judge the strength of each claim for themselves. 
Prior work used recurrent networks and large pre-trained encoders.

## Methods
**Architecture.** The model is a transformer encoder with 6 layers, 8 attention heads, hidden size 512 and 45M parameters.
**Training.** We split the data into train/validation/test sets (80/10/10) and trained for 20 epochs using seeds 1-5. All hyperparameters (learning rate 3e-4, batch size 64, 20 epochs, dropout 0.1, Adam optimizer) are stored in config.yaml, not hard-coded. Hyperparameters were tuned with a random search of 30 trials using 5-fold cross-validation on the training set; the search took about 6 GPU hours. The test set was used only once, after all modelling decisions were final.
**Code.** The training code, including the data split script, is available at https://github.com/example/sentiment under the MIT license. Results in this report were produced with commit 3f9a2c1d (tag v1.0.0). The same repository contains inference code (predict.py, with a usage example in the README) and evaluation code (evaluate.py). A single command (make reproduce) regenerates all tables and figures; see the README for how to run it.
**Environment.** Dependencies are pinned in requirements.txt (torch==2.1.0, numpy==1.26.0, scikit-learn==1.3.2) and a Dockerfile is provided.
**Metric.** Accuracy is defined as the fraction of reviews whose predicted label equals the gold label (see metrics.py). The metric is computed on the test set for each seed and averaged.

## Data
The dataset was collected from a public review API (dataset version v1.2; the sha-256 checksum is published with the release). Inclusion criteria: English reviews with at least 20 words; exclusion criteria: duplicates and reviews flagged as spam. Preprocessing (cleaning, tokenization, normalization) is done by preprocess.py in the repository. Columns: text, label (0 negative, 1 positive), source, date. Bias analysis: we checked label balance and performance by product category. Privacy: usernames were removed and the data is anonymised, so re-identification risk is low; consent follows the API terms. Sharing strategy: the anonymised data is released openly because no personal information remains. The dataset is publicly available at https://zenodo.org/records/1234567 (doi.org/10.5281/zenodo.1234567) under CC-BY-4.0. The held-out test set is released with it. A data card is included at https://huggingface.co/datasets/example/reviews.

## Results
Evaluation on the test set: accuracy 91.2 ± 0.4 % and F1 0.89 ± 0.01 over five seeds. Sample outputs and failure cases are shown in Table 3. The larger baseline did not improve on our model by more than 0.5 points, which we report as a null result for model size.

## Release
The final weights are released at https://huggingface.co/example/sentiment-small under Apache-2.0. Intermediate checkpoints saved every 1000 steps are available at https://zenodo.org/records/7654321. The model card (https://huggingface.co/example/sentiment-small) lists intended use, out-of-scope use, limitations and risks. Machine-readable metadata (Croissant) is in metadata.json.

## Process
Design decisions were logged in DECISIONS.md as they were made. Trade-offs: we chose a small model over a larger one because of the compute budget.

## Discussion
Performance drops on sarcastic reviews.

## Limitations
The model was trained on English product reviews from one platform and should not be assumed to transfer to other domains or languages. Sarcasm and domain shift remain failure cases.

## Conclusion
We release a fully reproducible sentiment model.

## References
[1] Vaswani et al. 2017. Attention is all you need. arXiv:1706.03762.
