Câu truyện cho phần intro có thể kể là classifier chains tận dụng dependencies để lấy hiệu suất và có thể làm explanation (ví dụ Shapley chains). Nhưng dùng 1 fully connected DAG trên labels nên mô hình phức tạp (tính exact BOPs là NP-hard kể cả BOPs của Hamming accuracy). Chia labels về IL và DL giúp giảm độ phức tạp mô hình => tính BOPs nhanh hơn và dễ explain hơn.

Tham khảo mẫu bài báo ở "https://www.sciencedirect.com/journal/expert-systems-with-applications"



Role: You are an expert AI/Computer Science researcher and IEEE Senior Member with extensive experience publishing in top-tier IEEE conferences and transactions.

Task: Generate a comprehensive, high-quality, full-length academic paper draft adhering strictly to the official IEEE two-column conference/journal format.

Target Format: Valid, compilable LaTeX code using the standard `IEEEtran` document class.

Paper Specification:
- Title: Greedy Selection of Independent Labels for Multi-Label Classification with Partial Abstention
- Research Field: Machine Learning
- Problem Statement & Research Gap: 
  Multi-label classification (MLC) involves assigning each instance to one or more relevant labels from a predefined set. Unlike single-label tasks, MLC acknowledges that instances often exhibit multiple attributes simultaneously, necessitating models capable of capturing label correlations to maximize predictive accuracy. However, modeling these complex label dependencies introduces significant computational challenges. One of the most widely studied families of models, Classifier Chains (CC) [1]–[4], addresses this by leveraging label correlations through an ordered chain structure, where each classifier’s prediction is conditionally dependent on the labels predicted by the preceding classifiers. This approach allows CC to effectively exploit the conditional dependencies between labels, often achieving superior performance compared to memory-less models that treat labels independently. Furthermore, the sequential nature of CC enables the generation of label-wise explanations, as the contribution of each label to the final prediction can be traced through the chain [5], [6]. Despite these advantages, standard CC suffers from several critical limitations that hinder its scalability and interpretability. Firstly, the inference process requires sequential predictions, making the computational cost dependent on the chain order [7]. More importantly, determining the optimal ordering of labels to minimize prediction errors is an NP-hard problem, even for evaluating simple metrics like exact match accuracy (EMA) [8]. This complexity arises because the full joint distribution of labels is often intractable, necessitating the use of approximate or heuristic ordering strategies. Secondly, the reliance on a fixed linear chain can propagate errors, where an incorrect prediction early in the chain may lead to compounding errors in subsequent predictions, degrading overall performance [9], [10]. This sensitivity to ordering and error propagation limits the practical applicability of CC in large-scale or high-stakes domains where prediction robustness is paramount.

- Proposed Method/Contribution: 
  To overcome the limitations of standard CC, we propose a novel approach that combines independent label (IL) learning with deep learning (DL) and leverages greedy selection to optimize model complexity and inference efficiency. Our method decomposes the MLC problem into two distinct components: (1) modeling label dependencies through a deep neural network that captures complex interactions between labels, and (2) selecting a subset of informative independent labels that serve as proxies for the full label set. By incorporating partial abstention, we allow the model to abstain from predicting certain labels when confidence is low, thereby reducing error propagation and improving robustness. The core of our contribution lies in the greedy selection algorithm that identifies the most informative independent labels to include in the chain, effectively reducing the model's complexity while maintaining high predictive accuracy. This approach allows us to approximate the optimal label ordering problem, which is intractable for the full label set, by focusing on a smaller subset of informative labels. Furthermore, by leveraging deep learning for label dependency modeling, we can capture complex non-linear interactions that are difficult to model with traditional methods. The combination of greedy selection and deep learning enables us to achieve a favorable trade-off between model complexity, inference efficiency, and predictive accuracy.

- Experiments & Datasets: 
  We evaluate our proposed method on several benchmark multi-label classification datasets, including PASCAL VOC [11], COCO [12], and EUREX [13]. These datasets vary in terms of the number of instances, the number of labels, and the label correlation structure, providing a comprehensive testbed for evaluating the method's effectiveness. We compare our method against several baseline models, including standard Classifier Chains (CC) [1]–[4], Random Classifier Chains (RCC) [14], and Ensemble of Classifier Chains (ECC) [15]. We evaluate the models using standard multi-label classification metrics, including Exact Match Accuracy (EMA) [8], Hamming Accuracy (HA) [16], F1-score [17], and Area Under the ROC Curve (AUC) [18]. We also evaluate the inference efficiency of the models by measuring the average inference time per instance.

- Key Results: 
  The experimental results demonstrate that our proposed method achieves competitive performance compared to the baseline models. Specifically, our method achieves an average EMA of 0.58, HA of 0.72, F1-score of 0.65, and AUC of 0.78. These results are comparable to the baseline models, which achieve average EMA of 0.56, HA of 0.70, F1-score of 0.63, and AUC of 0.76. Furthermore, our method achieves significantly faster inference times compared to the baseline models. Specifically, our method achieves an average inference time of 0.12 seconds per instance, compared to 0.18 seconds for CC, 0.20 seconds for RCC, and 0.25 seconds for ECC. These results demonstrate that our proposed method achieves a favorable trade-off between model complexity, inference efficiency, and predictive accuracy.

Structural & Writing Requirements:
1. Academic Tone: Third-person, formal, rigorous, concise, objective, and grammatically impeccable. Avoid conversational language or hand-waving claims.
2. Abstract: Exactly 150–200 words summarizing problem context, core limitation, proposed solution, key technical contribution, and quantifiable outcome.
3. Index Terms: 4–6 relevant IEEE keywords.
4. Section Breakdown:
   - I. INTRODUCTION: Problem motivation, literature landscape, identified gap, clearly bulleted key contributions (3-4 bullet points), paper organization outline.
   - II. RELATED WORK: Systematic categorization of prior literature with critical contrast against the proposed solution.
   - III. METHODOLOGY: Thorough formal mathematical formulation (using rigorous LaTeX equations, clearly defined variables, and algorithmic logic). Include pseudo-code inside an `algorithm` or `algorithmic` environment where relevant.
   - IV. EXPERIMENTS AND RESULTS: Experimental setup (datasets, evaluation metrics, implementation details, baseline models), empirical results formatted in IEEE LaTeX tables (`\begin{table}`), ablation study, and discussion of performance.
   - V. CONCLUSION AND FUTURE WORK: Summary of technical achievements, practical limitations, and potential directions for future exploration.
   - REFERENCES: 8–12 realistic, representative IEEE-formatted citations (`\begin{thebibliography}`).

LaTeX Formatting Rules:
- Enclose the entire output in a single valid LaTeX code block.
- Use `\documentclass[conference]{IEEEtran}`.
- Include standard packages: `cite`, `amsmath`, `amssymb`, `amsfonts`, `algorithmic`, `graphicx`, `textcomp`, `xcolor`, `booktabs`.
- Do not output placeholders like "[insert details here]"—flesh out complete theoretical explanations, math derivations, and detailed prose.