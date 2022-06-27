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
- **2104 SimCSE** 
- **2012 Universal Sentence Representation Learning with Conditonal Masked Language Model**
  - *Yang et al., Stanford, Google Research*
  - propose Conditional Masked Language Modeling(CMLM), to learn sentence representation on large scale unlabeld corpora, fully unsupervised
  - combinating next sentence prediction with MLM training
  - 와.. 미친 이 페이퍼 diffcse랑 너무 비슷한데? MLM을 하는데 주변 문장들의 semantic을 diffcse의 cls_input 처럼 넘겨준다;;
  - Multilingual로도 테스트
  - skip-though 랑 비슷
  - avg pooling
  - sentence embedding을 3 layer projection을 통해서 H x N dim으로 변환하는데, 나도 SE와 Discriminator를 다른 모델을 쓴다면 이런 방식을 사용할 수도 있겠다.
    - 여기서 N이 마치 Prompt의 Length처럼 느껴진다.
  - inference 할 때는 First Encoder만 사용
  - Common Crawl dumps, MLM 약 30% 적용하여 more challenging 하게 만듬
- **2106 Self-guided contrastive learning for BERT sentence Representation ![](https://img.shields.io/badge/STS-purple)**
  - *Taeuk et .al, NAVER*
  - fine-tunes BERT in a self-supervised fashion, does not rely on data augmentation, redesign the contrastive learning objective 
  - Conflate(융합) the knowledge stored in *BERT's different layers* to produce sentence embeddings
  - BERT(Fixed), BERT(Train) 두개의 센텐스 아웃풋을 positive sample로 사용. 이 때, Fixed BERT의 각 레이어의 아웃풋 중 max pooling 하여 사용
- **2107 CLINE:contrastive learning with semantic negative emxaples for natural language understanding**
  - *Wang et al., 칭화대*
  - using wordnet to generate adversarial and contrastive examples
  - 이 친구들도 RTD를 이용함. 다만 BERT가 아니라 spacy와 wordnet으로 하는듯

- **2109 ESimCSE: Enhance sample building method for contrastive learning of unsupervised sentence embedding ![](https://img.shields.io/badge/STS-purple)**
  - *Wu et al, Chinese Academy of Sciences*
  - positive pair와 negative pair의 sentence length 차이에 집중
  - 랜덤한 단어를 반복해서 넣음으로써, 의미를 훼손 시키지 않은채로 문장의 길이를 변경함 --> 조금 더 robust 한 모델을 만들때 괜찮은데?
  - momentum contrast 사용
- **2201 SNCSE: contrastive learning for unsupervised sentence embedding with soft negative samples**
  - *Wang et al, sensetime*
  - 기존 모델들은 feature supperssion이 발생하여, textual similarity가 높은 문장들에 대해서 overestimate 하는 경향이 있다. 따라서 negation(부정) 샘플들을 활용하여 이러한 문제를 극복하고자 하는 논문
  - soft negative sample : 의미는 다르지만 textual 하게 비슷한 형태를 가지므로 완전히 negative라고 말할수는 없는 문장
  - negated sentences are generated through a rule-based method(spacy tool)
  - H를 얻고자, prompt를 사용 "X" means [MASK] 에서 MASK의 hidden output을 사용
  - 1M 위키 샘플을 사용하긴 했는데, Spacy로 negative sample을 만들었으니 SimCSE와 Fair한 비교는 아니라고 봐야하지 않나?

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
- **2201 PromptBERT: Improving BERT Sentence Embeddings with Prompts**
  - *Jiang et al.,, 베이징대*
  - Reforming the sentence embeddings task as the fillin-the-blanks promblem.
  - Two prompt representing method, 3 prompt searching methods
  - Anisotropy makes the token embeddings cuupy a narrow cone, resuling in a high similarity between any sentence pair
  - 이방성이 poor sentence representaiton의 문제인줄 알았으나, 이방성이 상대적으로 적은 마지막 layer의 average pooling이 이방성이 더 큰 static token averaging 보다 더 sentence representaiton 이 구린 것을 봐서는, 이방성 만이 문제는 아닌것으로 봄
  - high frequency subwords나 punctuation을 제거한 뒤 token averaging을 s.r 로 사용. token bias를 줄임
  - Anisotropy를 구하는 공식
    - ![image](https://user-images.githubusercontent.com/18374514/175939045-d0dabd46-d844-43c6-abb4-d911cf1ddd18.png)
    - 

## ETC

- **2011 On the sentence embeddings from pre-trained language models ![](https://img.shields.io/badge/NLU-green)**
  - *Li et al., Carnegie Mellon University*
  - BERT-flow 논문
  - reveal the theoretical connection between the masked lanugae model pre-training objective and the semantic similarity task theoretically.
  - **BERT always induces a non-smooth anisotropic(이방성) semantic space of sentences,** which harms its performance of semantic similarity.
  - anisotropic : word embedding 이 narrow cone 처럼 되어 있는 현상 말하는듯??
  - average token embedding
  - mlm loss 간단하게 설명해야하면 이 페이퍼 참조해도 될듯.
  - PMI is a common mathematical surrogate to approximate word-level semantic similarity.
  - word embeddings can be biased to word frequency
