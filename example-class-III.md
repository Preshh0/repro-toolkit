# Fraud detection report

## Abstract
This technical report describes a fraud detection model and the data behind it.

## Introduction
Fraud costs money. We ask whether a small neural network can flag fraudulent card transactions better than a rule baseline. The primary success metric (F1 on the held-out test set) was fixed before training. Scope: completed card transactions between 2022 and 2023 from one payment processor.

## Related Work

The motivation for this work comes from practical deployments where teams repeatedly rebuild similar systems without sharing details. In our experience, the main obstacles to reuse are missing documentation, undocumented preprocessing and unclear evaluation protocols. We therefore describe each step of the pipeline carefully and explain why alternative designs were rejected. The remainder of this report follows the usual structure: we describe the problem setting and the related literature, then the method and the data, and finally the results, a discussion of the limitations and our conclusions. Throughout the report we try to be explicit about what was measured, how it was measured and what we could not measure, so that readers can judge the strength of each claim for themselves. 

The motivation for this work comes from practical deployments where teams repeatedly rebuild similar systems without sharing details. In our experience, the main obstacles to reuse are missing documentation, undocumented preprocessing and unclear evaluation protocols. We therefore describe each step of the pipeline carefully and explain why alternative designs were rejected. The remainder of this report follows the usual structure: we describe the problem setting and the related literature, then the method and the data, and finally the results, a discussion of the limitations and our conclusions. Throughout the report we try to be explicit about what was measured, how it was measured and what we could not measure, so that readers can judge the strength of each claim for themselves. 
Earlier work used rules and logistic regression.

## Methods
The architecture is a feed-forward neural network with 4 layers, hidden size 256 and 0.3M parameters. Hyperparameters (learning rate 1e-3, batch size 128, 30 epochs, Adam optimizer) are stored in a YAML config file. Hyperparameters were tuned with a grid search of 24 configurations using 5-fold cross-validation; it took about 2 hours. F1 is defined as the harmonic mean of precision and recall, computed on the test set. The test set was used only once, after all decisions were final. Results are averaged over 5 seeds (seeds 1-5). Design decisions were logged in DECISIONS.md. Trade-offs: we chose a small network over gradient boosting because of latency constraints.

## Data
The dataset was collected between 2022 and 2023 (version v2) from the processor's transaction logs. Inclusion criteria: completed transactions; exclusion criteria: test accounts. Preprocessing: cleaning and normalisation. Columns: amount, merchant, label (1 = fraud). Bias analysis: we checked error rates by region and merchant type. Privacy: all identifiers were anonymised, so re-identification risk is low; consent was obtained and the licence is CC-BY-4.0. Data card: https://example.org/datacard. The README explains how to run the model.

## Results
Test accuracy 96.1 % and F1 0.78 ± 0.02 on the held-out set. Sample outputs and failure cases are shown in Table 2. Rare fraud types were missed, which we report as a negative result.

## Release
Final weights are released at https://huggingface.co/example/fraud under Apache-2.0. The model card at https://huggingface.co/example/fraud lists intended use, out-of-scope use, limitations and risks.

## Limitations
The model was trained on one processor's data and has not been tested on other markets.

## Conclusion
A useful baseline.

## References
[1] Dal Pozzolo 2015.
