<div align="center">

## 🔍 Linguistic Minimal Pairs Elicit Linguistic Similarity in Large Language Models

[Xinyu Zhou](https://www.linkedin.com/in/xinyu-zhou2000/)
<img src="doc/cuhk.png" alt="Logo" width="16">, &nbsp; &nbsp;
[Delong Chen](https://chendelong.world/)
<img src="doc/hkust.png" alt="Logo" width="12">, &nbsp; &nbsp;
[Samuel Cahyawijaya](https://samuelcahyawijaya.github.io/)
<img src="doc/hkust.png" alt="Logo" width="12">, &nbsp; &nbsp;

[Xufeng Duan](https://xufengduan.github.io/)
<img src="doc/cuhk.png" alt="Logo" width="16">, &nbsp; &nbsp;
[Zhenguang G. Cai](https://sites.google.com/site/zhenguangcai/)
<img src="doc/cuhk.png" alt="Logo" width="16">


<img src="doc/cuhk_with_text.png" alt="Logo" width="150">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img src="doc/hkust_with_text.png" alt="Logo" width="90"> 


[ArXiv]() |
[Data]()


</div>

This project introduces a novel analysis that leverages linguistic minimal pairs to probe the internal linguistic representations of Large Language Models (LLMs). By measuring the similarity between LLM activation differences across minimal pairs, we quantify the linguistic similarity and gain insight into the linguistic knowledge captured by LLMs.

Our large-scale experiments, spanning 100+ LLMs and 150k minimal pairs in three languages, reveal properties of linguistic similarity from four key aspects:

1. **Consistency across LLMs**: Linguistic similarity is significantly influenced by training data exposure, leading to higher cross-LLM agreement in higher-resource languages.

2. **Relation to theoretical categorizations**: Linguistic similarity strongly aligns with fine-grained theoretical linguistic categories but weakly with broader ones.

3. **Dependency to semantic context**: Linguistic similarity shows a weak correlation with semantic similarity, showing its context-dependent nature.

4. **Cross-lingual alignment**: LLMs exhibit limited cross-lingual alignment in their understanding of relevant linguistic phenomena.


## News
- **2024.09.20**. Our paper is available on arXiv! [(arxiv link)]()


## Methodology

<p align="center">
<img src="doc/method_figure.png" width="90%">
</p>

We extract LLM activations for sentences in linguistic minimal pairs and compute their differences. Since the sentences differ solely in a specific linguistic phenomenon, the resulting difference only contains information about that phenomenon. We then measure the similarity between these activation differences, which we refer to as linguistic similarity.
