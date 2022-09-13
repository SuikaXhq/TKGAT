# Tag-aware Knowledge Graph Attention Network

This is the PyTorch implementation for the paper Tag-aware Knowledge Graph Attention Network in Knowledge-Based Systems.

## Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
* torch == 1.3.1
* dgl-cu90 == 0.4.1
* numpy == 1.15.4
* pandas == 0.23.1
* scipy == 1.1.0
* sklearn == 0.20.0

## Run the Codes
* TKGAT
```bash
python main_tkgat.py --data_name movielens
```

## Related Papers
* FM
    * Proposed in [Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002), SIGIR2011.

* NFM
    * Proposed in [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777), SIGIR2017.

* BPRMF
    * Proposed in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=1795167), UAI2009.
    * Key point: 
        * Replace point-wise with pair-wise.

* ECFKG
    * Proposed in [Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation](https://arxiv.org/abs/1805.03352), Algorithm2018.
    * Implementation by the paper authors: [https://github.com/evison/KBE4ExplainableRecommendation](https://github.com/evison/KBE4ExplainableRecommendation)
    * Key point: 
        * Introduce Knowledge Graph to Collaborative Filtering

* CKE
    * Proposed in [Collaborative Knowledge Base Embedding for Recommender Systems](https://dl.acm.org/citation.cfm?id=2939673), KDD2016.
    * Key point: 
        * Leveraging structural content, textual content and visual content from the knowledge base.
        * Use TransR which is an approach for heterogeneous network, to represent entities and relations in distinct semantic space bridged by relation-specific  matrices.
        * Performing knowledge base embedding and collaborative filtering jointly.

* KGAT
    * Proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD2019.
    * Implementation by the paper authors: [https://github.com/xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)
    * Key point:
        * Model the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.
        * Train KG part and CF part in turns.
        


