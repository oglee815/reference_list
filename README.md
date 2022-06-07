# Reference List

## Keyword
- ![](https://img.shields.io/badge/NLU-green) ![](https://img.shields.io/badge/NLG-orange) ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/Discrete_Prompt-red)

## Contrastive


## Prompt Learning

- **2101 Prefix-Tuning : Optimizing Continuous Prompts for Generation ![](https://img.shields.io/badge/Continuous_Prompt-blue) ![](https://img.shields.io/badge/NLG-orange)**
  - *ACL2021, Li and Liang, Stanford*
  - Encoder와 decoder 입력 앞에 virtual token(prefix)을 붙여서, 해당 토큰과 관련된 파라미터만 학습함(Continuous vector)
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
- **2107 Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.**
  - *Preprint, Liu et al.,* 
  - dddd
