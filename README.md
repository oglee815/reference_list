# Reference List

## Keyword
- ![](https://img.shields.io/badge/NLU-green) ![](https://img.shields.io/badge/NLG-orange) ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/Discrete_Prompt-red) ![](https://img.shields.io/badge/Survey-grey) ![](https://img.shields.io/badge/STS-purple)

## Contrastive

- **2005 Cert: Contrastive self-supervised learning for language understanding**
  - *Hongchao Fang et al., UC San Diego*
  - Augment 용으로 Back-Translation 사용
- **2005 Pretraining with Contrastive Sentence Objectives Improves Discourse Performance of Language Models**
  - *ACL 2020, Dan Iter et al., Stanford, Google Research*
  - Anchor Sentence 주변 문장을 예측하는 방식으로 Contrastive learning, discourse relationship에 초점을 맞춤
  - DiscoEval , RTE, COPA and ReCoRD tasks
- **2006 DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations**
  - *ACL2021, John Giorgi et al., Toronto Univ.*
  - Paragraph 단위로 인접하거나, 포함된 span을 positive로 MLM과 함께 학습
  - CR, MR, MPQA, SUBJ, SST2, SST5, TREC, MRPC, SNLI, Task
- **2009 SEMANTIC RE-TUNING WITH CONTRASTIVE TENSION ![](https://img.shields.io/badge/STS-purple)**  
  - *ICLR2021, Fredrik Carlsson et al., RISE NLU group*
  - a new self-supervised method called Contrastive Tension (CT) : a noise-contrastive task
  - Fine Tuning 전에는 마지막 레이어보다 앞선 레이어의 **mean pooling**이 STS 점수가 더 높다
  - eng wiki 사용, batch 16. 동일한 문장으로 positive set을 만드는데, simcse와 다른 점은 다른 두개의 모델을 사용하는것(초기화는 동일)

  
## Prompt Learning

- **2101 Prefix-Tuning : Optimizing Continuous Prompts for Generation ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/NLG-orange)**
  - *ACL2021, Li and Liang, Stanford*
  - Encoder와 decoder 입력 앞에 **virtual token(prefix)** 을 붙여서, 해당 토큰과 관련된 파라미터만 학습함(Continuous vector)
  - prefix 토큰의 임베딩 메트릭스를 곧바로 학습하지 않고 reparameterize를 시켜서 학습함. p = MLP(p'), inference 시에는 p만 사용
  - Task : Table-To-Text, Abstractive Sum 
  - 흥미로운 결과: Prefix를 랜덤보다 그냥 banana라도 real word로 하는게 성능이 좋음, prefix vs infix 끼리 비교도 함 
  - P-tuning하고 유사한 continuous prompt learning이지만 이 페이퍼는 prefix를 real token으로 사용하는 차이가 있는 것 같음
- **2103 P-tuning: GPT Understands, Too ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/NLU-green)**
  - *Preprint, Liu et al, Tsinghua*
  - Discrete Prompt Engineering의 한계를 극복하기 위해 Continuous Prompt Learning의 최초 제안 
  - we propose a novel method P-tuning  to automatically search prompts in the continuous space to bridge the gap between GPTs and NLU applications
  - Downstream tasks : LAMA, SuperGlue Task 
  - Motivation: these models(large lm) are still too large to memorize the fine-tuning samples 
  - Prompt Encoder : bi-LSTM + 2LayerMLP(ReLU)
- **2104 PROMPT TUNING: The Power of Scale for Parameter-Efficient Prompt Tuning ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/NLG-orange)**
  - *Lester et al, Google*
  - simplification of Prefix-tuning 
  - T5 활용한게 특징 
  - We freeze the entire pre-trained model and only allow an additional k tunable tokens per downstream task to be prepended to the input text 
  - We are the first to show that prompt tuning alone (with no intermediate-layer prefixes or task-specific output layers) 
  - We cast all tasks as text generation. 
  - 뭔가 방법적으로 특별한게 뭔지 모르겟음
- **2104 SOFT PROMPTS: Learning How to Ask: Querying LMs with Mixtures of Soft Prompts ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/NLU-green)**
  - *Qin and Eisner, Johns Hopkins*
  - We explore the idea of learning prompts by gradient descent—either fine-tuning prompts taken from previous work, or starting from random initialization
  - Our prompts consist of “soft words,” i.e., continuous vectors that are not necessarily word type embeddings from the language model 
  - binary relation관계에 있는 x(입력), y(정답)을 맞추는 문제에 적용(LAMA, T-REx 등)
  - Deep Perterbation 적용 
  - BERT, BART 등 사용
- **2107 Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. ![](https://img.shields.io/badge/Survey-grey)**
  - *Preprint, Liu et al.,* 
  - dddd
- **2110 P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/NLU-green)**
  - *Liu et al, Tsinghua*
  - normal size model, hard sequence labeling task(like SQuAD)에 p-tuning 적용
  - prefix-tuning, soft prompt를 NLU에 적용한 버전
- **Prompt Tuning for Discriminative Pre-trained Language Models**
  - https://arxiv.org/pdf/2205.11166.pdf

## ETC

- **2011 On the sentence embeddings from pre-trained language models ![](https://img.shields.io/badge/NLU-green)**
  - *Li et al., Carnegie Mellon University*
  - BERT-flow 논문
  - reveal the theoretical connection between the masked lanugae model pre-training objective and the semantic similarity task theoretically.
  - **BERT always induces a non-smooth anisotropic(이방성) semantic space of sentences,** which harms its performance of semantic similarity.
  - anisotropic : word embedding 이 narrow cone 처럼 되어 있는 현상 말하는듯??
  - average token embedding
  - mlm loss 간단하게 설명해야하면 이 페이퍼 참조해도 될듯
  - PMI is a common mathematical surrogate to approximate word-level semantic similarity.
  - word embeddings can be biased to word frequency
