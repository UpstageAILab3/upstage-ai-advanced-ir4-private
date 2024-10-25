[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tm6AYAOm)

# '디지털 보물찾기(Digital Treasure Quest)' 팀의 '과학 지식 질의 응답 시스템 구축(Scientific Knowledge Question Answering)' 경연 도전기

## Team

| ![박석](https://avatars.githubusercontent.com/u/5678836?v=4) | ![백경탁](https://avatars.githubusercontent.com/u/62689715?v=4) | ![위효연](https://avatars.githubusercontent.com/newwhy) | ![남상이](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박석](https://github.com/parksurk)             |            [백경탁](https://github.com/ktbaek72)             |            [위효연](https://github.com/newwhy)             |            [남상이](https://github.com/UpstageAILab)             |
|                            Lead, R&D                             |                            R&D                             |                            R&D                             |                            R&D                             |

![박석](./images/team-member-ps.png)
![백경탁](./images/team-member-pkt.png)
![위효연](./images/team-member-why.png)
![남상이](./images/team-member-nse.png)


## 1. Competition Info

### 1.1. Overview

#### 대회 개요

##### 목표
과학 상식 기반의 질문에 대해 적절한 답변을 제공하는 RAG(Retrieval Augmented Generation) 시스템을 개발합니다.

##### 소개
'과학 지식 질의응답 시스템 구축' 경진대회는 LLM(Large Language Model)이 가지는 한계를 극복하고, 과학 지식에 대한 질문에 적합한 문서를 추출한 후 이를 기반으로 답변을 생성하는 RAG 시스템을 구축하는 대회입니다. 특히, 과학 상식을 기반으로 한 질문을 판단하고, 검색엔진을 활용해 적절한 문서를 찾아 답변을 제공하는 것이 핵심입니다. LLM이 잘못된 정보를 생성하는 문제를 해결하고, 사용자에게 신뢰할 수 있는 정보를 제공하는 시스템을 개발하는 것이 목표입니다.

##### 제공 데이터셋
1. **과학 상식 문서**: 약 4,200개의 과학 상식 문서가 사전 색인되어 제공됩니다.
2. **평가 데이터**: 사용자의 질문 및 그에 대한 정답 여부를 판단할 수 있는 평가 데이터

##### 사용 가능한 알고리즘
RAG 시스템 개발을 위해 다양한 검색 엔진과 LLM을 활용할 수 있습니다. LLM은 주어진 질문에 적합한 문서를 기반으로 답변을 생성하며, 정보의 정확성과 신뢰성을 중시하는 시스템을 구축할 수 있도록 알고리즘을 조합할 수 있습니다.

##### 시스템 개발 목표
참가자들은 과학 상식 문서를 바탕으로 질문에 적합한 정보를 검색하고, 이를 바탕으로 신뢰성 있는 답변을 생성하는 RAG 시스템을 구축해야 합니다. 이 과정에서 검색된 문서의 적합성과 답변의 정확성을 평가하는 것이 주요 목표입니다.

##### 제출 형식
JSON 형식의 결과물을 제출합니다.

- **Input**: 사용자 질문
- **Output**: 추출된 문서 및 생성된 답변

자세한 내용은 대회 페이지에서 확인하세요.

### 1.2. Timeline

#### 프로젝트 전체 기간
- **2024년 10월 2일 (수) ~ 2024년 10월 24일 (목)**

#### 주요 일정
- **프로젝트 시작**: 2024년 10월 2일 (수)
- **팀 병합 마감**: 2024년 10월 3일 (목) 16:00
- **최종 제출**: 2024년 10월 24일 (목) 19:00
- **GPU 서버 운영 기간**: 2024년 10월 2일 (수) ~ 2024년 11월 8일 (금) 16:00

#### 상세 일정

1. **2024년 10월 2일 (수)**: 프로젝트 시작 및 데이터셋 준비
   - 데이터 탐색 및 전처리 시작
   - 팀 구성 및 역할 분담
   - 초기 베이스라인 시스템 논의 시작

---

##### Week 1 (2024년 10월 2일 ~ 10월 8일)

2. **2024년 10월 3일 (목)**: 베이스라인 시스템 설계
   - RAG 시스템 구조 설계
   - 검색 엔진 세팅 및 검색 알고리즘 결정
   - 팀 병합 완료 및 역할 확정

3. **2024년 10월 4일 (금)**: 검색 시스템 구축
   - 초기 검색 시스템 구축 및 테스트
   - 적합한 문서 추출 성능 검토
   - 성능 평가 기준 논의

4. **2024년 10월 5일 (토)**: 답변 생성 모델 구축
   - LLM 기반 답변 생성 시스템 구현
   - 검색된 문서를 기반으로 한 초기 답변 생성 실험

5. **2024년 10월 6일 (일)**: 시스템 통합 및 초기 테스트
   - 검색 및 답변 생성 시스템 통합
   - 토이 데이터셋을 통한 초기 테스트
   - 성능 평가 및 개선 방향 논의

6. **2024년 10월 7일 (월)**: 에러 분석 및 개선 작업
   - 검색 및 생성 결과에 대한 에러 분석
   - 시스템 개선을 위한 피드백 반영

7. **2024년 10월 8일 (화)**: 베이스라인 시스템 개선
   - 하이퍼파라미터 튜닝 및 성능 최적화 작업
   - 추가 문서 및 데이터 분석 진행

---

##### Week 2 (2024년 10월 9일 ~ 10월 15일)

8. **2024년 10월 9일 (수)**: 모델 고도화 및 성능 최적화
   - RAG 시스템 고도화 작업
   - 전체 데이터셋을 이용한 성능 개선 실험

9. **2024년 10월 10일 (목)**: 답변 생성 모델 고도화
   - 다양한 LLM 모델 실험 및 성능 비교
   - 답변 생성 품질 개선 작업

10. **2024년 10월 11일 (금)**: 성능 평가 및 분석
    - MAP 기반 성능 평가 및 분석
    - 에러 사례 분석 및 추가 개선

11. **2024년 10월 12일 (토)**: 데이터 증강 및 피처 엔지니어링
    - 추가적인 데이터 증강 시도
    - 성능 최적화를 위한 피처 엔지니어링 작업

12. **2024년 10월 13일 (일)**: 시스템 성능 검토
    - 최종 성능 검토 및 개선사항 논의
    - 최종 테스트 계획 수립

13. **2024년 10월 14일 (월)**: 결과 분석 및 문서화
    - 최종 성능 분석 및 결과 도출
    - 개발 과정 및 성능 결과 문서화 시작

14. **2024년 10월 15일 (화)**: 최종 점검 및 제출 준비
    - 최종 모델 점검 및 성능 테스트
    - 최종 제출물 준비 완료

---

##### Week 3 (2024년 10월 16일 ~ 10월 23일)

15. **2024년 10월 16일 (수)**: Public 테스트 및 제출 준비
    - 공개된 테스트 데이터로 최종 성능 평가
    - 최종 결과물 점검 및 제출 준비

16. **2024년 10월 17일 (목)**: 최종 모델 테스트
    - Private 테스트 데이터를 사용해 최종 시스템 성능 검토

17. **2024년 10월 18일 (금)**: 최종 모델 제출 준비
    - 최종 결과물 점검 및 최종 제출물 준비

18. **2024년 10월 19일 (토)**: 최종 백업 및 검토
    - 서버 백업 및 최종 검토 작업 완료

---

##### Week 4 (2024년 10월 24일 목요일): 대회 종료

19. **2024년 10월 24일 (목)**: 최종 모델 제출 및 대회 종료
    - 최종 제출 및 리더보드 결과 확인


### 1.3. Evaluation

#### 평가방법

이번 대회는 RAG 시스템을 통해 과학 상식 기반의 질문에 적합한 문서를 얼마나 잘 추출했는지를 평가하는 대회입니다. **MAP(Mean Average Precision)** metric을 기준으로 문서 추출의 적합성을 평가하며, 질문에 따라 문서를 추출할 필요가 없는 경우도 고려하여 수정된 MAP 로직이 적용됩니다.

##### MAP란?
MAP는 다수의 질의에 대해 평균적인 정밀도를 계산하는 평가 지표로, 특정 질의에 대해 문서가 얼마나 적절하게 추출되었는지를 평가합니다. 

##### MAP 계산 방법
1. 문서가 잘 추출된 경우, Precision-recall curve에 기반하여 정밀도를 계산합니다.
2. 과학 상식 질문이 아닌 경우, 검색된 문서가 없으면 1점을 부여하고, 그렇지 않으면 0점을 부여합니다.
3. 모든 질의에 대한 평균 정밀도를 계산하여 최종 MAP 점수를 산출합니다.

자세한 내용은 대회 제공 코드를 참고하세요.

#### Public 및 Private 평가
- **Public 평가**: 제출된 결과물 중 일부를 기반으로 리더보드에 반영됩니다.
- **Private 평가**: 나머지 테스트 데이터를 사용하여 최종 리더보드 점수가 산출됩니다.

자세한 내용은 대회 페이지에서 확인할 수 있습니다.

## 2. Winning Strategy

### 2.1. DTQ Team's Pros and Cons
#### Pros
- 다양한 경력과 경험을 가진 팀원들
- 다양한 뷰 포인트와 색깔를 가진 팀원들
- AI Assistant에 대한 수용력이 높음

#### Cons 
- Git을 활용한 팀단위의 R&D 경험 수준 낮음
- Python기반 R&D 경험 수준 낮음
- 머신러닝/딥러닝 R&D 경험 수준 낮음
- 경연 주제와 관련된 도메인 지식이 낮음
- Career Path에 대한 개인적인 목표가 모두 다름 

### 2.2. DTQ Team's strategic approach
- 첫째, Jupyter notebook이 아닌 커멘드라인 인터페이스로 동작하는 팀 베이스라인 코드를 구축하여 팀 단위 협업을 가능하게 한다.
- 둘째, ChatGPT와 같은 AI활용 도구를 적극 활용하여 EDA 와 Model Selection 의 방향성을 잡는다.
- 셋째, 팀원별 서로 다른 머신러닝 모델링을 각 팀원별 수준에 맞게 진행한다.

### 2.3. DTQ Team's culture & spirit
- 경연 참가의 목적은 개인별 학습을 통해 머신러닝 R&D에 필요한 지식과 경험을 얻는 것에 있다.
- 팀원 각각이 처한 상황을 서로 이해하고 인정하고 Respect 한다.
- AI Assistant를 적극적으로 활용하여 개인별 생산성을 극대화 한다.
- 팀 전체 목표을 위해 팀원 개개인의 스케쥴이나 리소스를 희생해서는 안된다.
- 팀원별로 최소한 한번의 제출은 해 본다.

## 3. Components

### 3.1. Directory

- code : 팀원별 실험 소스 코드 및 관련 문서
  - tm1 : 팀원(박석) 실험 소스 코드 및 관련 문서
  - tm2 : 팀원(백경탁) 실험 소스 코드 및 관련 문서
  - tm3 : 팀원(위효연) 실험 소스 코드 및 관련 문서
  - tm4 : 팀원(남상이) 실험 소스 코드 및 관련 문서
  - team : 팀전체 협업을 위한 실험 소스 코드 및 관련 문서
- docs : 팀 문서(발표자료, 참고자료 등)
  - presentation : 발표자료
  - reference : 참고자료
- images : 첨부 이미지
- README.md : 디지털 보물찾기(Digital Treasure Quest)' 팀의 '과학 지식 질의 응답 시스템 구축' 경연 도전기 Readme.md

```markdown
## 4. Data description

### 4.1. Dataset overview

#### 학습 데이터
| **Desc**       | **Details**                                                   |
|----------------|---------------------------------------------------------------|
| File name      | documents.jsonl                                                    |
| Rows           | 4,200+                                                        |
| Columns        | 3                                                             |
| docid          | 문서 고유 ID (UUID 형식)                                        |
| src            | 문서의 출처 (예: ko_mmlu__nutrition__test, ko_ai2_arc__ARC_Challenge__test) |
| content        | 문서의 내용 (과학 상식 관련 정보)                                |

#### 평가 데이터
| **Desc**       | **Details**                                                   |
|----------------|---------------------------------------------------------------|
| File name      | eval.jsonl                                                    |
| Rows           | 220                                                           |
| Columns        | 2                                                             |
| eval_id        | 평가 메시지 고유 ID                                            |
| msg            | 사용자와 시스템 간의 대화 메시지 리스트 (멀티턴 포함)             |

#### 테스트 데이터
- **비공개 테스트 데이터는 제공되지 않습니다.**  
  테스트 데이터는 대회에서 평가 목적으로만 사용되며, 참가자는 최종 성능을 제출하기 전에는 해당 데이터를 사용할 수 없습니다.

### 4.2. EDA

#### Feature Description

| **Feature Name** | **Index** | **Description**                                               |
|------------------|-----------|---------------------------------------------------------------|
| docid            | 1         | 문서 고유 ID (UUID 형식)                                       |
| src              | 2         | 문서 출처 (MMLU, ARC 등 과학 상식 데이터 출처)                   |
| content          | 3         | 과학 상식 내용이 포함된 문서 텍스트                              |
| eval_id          | 4         | 평가 메시지 고유 ID                                            |
| msg              | 5         | 사용자와 시스템 간의 대화 내용 (자연어 질문 및 멀티턴 대화 포함) |

#### EDA 결과

1. **문서 내용 분석**: 각 문서에는 과학 상식이 포함되어 있으며, 문서의 길이는 다소 다양합니다. 과학 상식 주제는 영양학, 물리학 등 여러 분야에 걸쳐 있으며, 데이터는 MMLU 및 ARC 데이터 출처로 분류됩니다.
   
2. **출처 분석**: 데이터는 두 가지 주요 출처에서 제공되며, `ko_mmlu__nutrition__test`와 같은 형식으로 출처가 명시되어 있습니다. 이는 문서의 신뢰성과 출처를 추적하는 데 도움이 됩니다.

3. **평가 메시지 분석**: 220개의 평가 메시지는 사용자와 시스템 간의 대화 형식으로 이루어져 있으며, 멀티턴 대화 20개, 일반 대화 20개가 포함되어 있습니다. LLM을 활용해 각 대화에서 적절한 질의를 추출해야 하는 것이 핵심 과제입니다.

4. **테스트 데이터 부재**: 학습 또는 평가 목적으로 사용할 수 있는 공개된 테스트 데이터가 없습니다. 이로 인해 모델 성능을 사전 검증하는 데 어려움이 따르며, 최종 평가에서 공개되지 않은 테스트 데이터를 사용하기 때문에 모델의 일반화 능력을 사전 평가하기 위한 추가적인 접근 방식이 필요합니다. 
   - **필요성**: 공개된 테스트 데이터가 없는 상황에서 사전 평가를 위해선 토이 데이터나 유사한 과학 상식 데이터를 활용하여 RAG 시스템을 실험하고 최적화하는 방법이 필요합니다. 또한, 평가 메시지에 대한 다각적 분석 및 LLM 프롬프트 최적화를 통해 시스템의 성능을 극대화하는 것이 중요합니다.

#### EDA 결론
- 데이터는 주로 문서 출처와 과학 상식 정보를 기반으로 하며, RAG 시스템 구축 시 적절한 문서 검색 및 답변 생성이 필수적입니다.
- 테스트 데이터가 공개되지 않으므로, 평가 데이터를 최대한 활용한 성능 검증과 함께 다양한 프롬프트 최적화 방법을 시도해야 합니다.
```


### 4.3. Data Processing

#### 1. **질문 생성 및 Cosine Embedding Loss 계산**

1. **질문 생성**: `documents.jsonl` 파일의 각 레코드에서 문서 내용을 바탕으로 ChatGPT를 사용하여 각 문서당 5개의 질문을 생성했습니다. 생성된 질문은 문서 내용과 얼마나 연관이 있는지 측정하기 위한 시작점입니다.
   
2. **Cosine Embedding Loss 계산**: 생성된 질문과 문서 간의 연관성을 Cosine Embedding Loss로 측정하였습니다. 이 Loss 값이 낮을수록 질문과 문서가 더 관련성이 높은 것으로 간주하며, 이를 Positive pair로 정의하고, Loss 값이 높을수록 Negative pair로 정의됩니다. 각 문서에 대해 5개의 Positive 질문과 5개의 Negative 질문을 생성하였고, 이를 JSON 형식으로 저장하였습니다.

```python
import json
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up your OpenAI client
client = OpenAI(api_key="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")

# Function to generate positive and negative questions using OpenAI API
def generate_questions_openai(content):
    # Prompt to generate positive questions
    positive_prompt = f"Create 5 questions in Korean that are related to the following content:\n\n{content}\n"
    negative_prompt = "Create 5 questions in Korean that are not related to the content above (completely unrelated)."

    # Generate positive questions
    positive_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": positive_prompt}],
        max_tokens=150,
        n=1,
        temperature=0.7
    )
    positive_questions = positive_response.choices[0].message.content.strip().split('\n')

    # Generate negative questions
    negative_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": negative_prompt}],
        max_tokens=150,
        n=1,
        temperature=0.7
    )
    negative_questions = negative_response.choices[0].message.content.strip().split('\n')

    return positive_questions, negative_questions

# Load the documents.jsonl file
input_file = '/home/data/documents.jsonl'
output_file = '/home/data/test.jsonl'

data = []
with open(input_file, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Prepare to write results into test.jsonl
with open(output_file, 'w') as outfile:
    test_id = 1

    for record in data:
        content = record['content']
        
        # Generate positive and negative questions using OpenAI API
        positive_questions, negative_questions = generate_questions_openai(content)
        all_questions = positive_questions + negative_questions

        # Vectorize content and questions using TF-IDF
        vectorizer = TfidfVectorizer()
        texts = [content] + all_questions
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Calculate Cosine Similarity and Cosine Embedding Loss
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        cosine_embedding_losses = 1 - cosine_similarities[0]

        # Create test.jsonl entries for each question
        for i, question in enumerate(all_questions):
            CELoss = cosine_embedding_losses[i]
            entry = {
                "test_id": f"{test_id:02d}",
                "msg": [{"role": "user", "content": question}],
                "docid": record['docid'],
                "CELoss": float(CELoss)
            }
            # Write each entry to the test.jsonl file
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
            test_id += 1

print("test.jsonl file has been created.")
```

3. **테스트 데이터 생성**: 각 문서에 대해 Positive 및 Negative 질문을 생성한 후, 해당 질문과 관련된 Cosine Embedding Loss를 포함한 `test.jsonl` 파일을 생성하였습니다. 이 파일에는 각 질문과 관련된 Cosine Embedding Loss 값이 포함되어 있습니다. 

```python
{
    "test_id": 01,
    "msg": [{"role": "user", "content": "나무의 분류에 대해 조사해 보기 위한 방법은?"}],
    "CELoss": 0.876
}
```

#### 2. **Ground Truth 생성**

1. **Ground Truth 데이터 생성**: `test.jsonl` 파일에 포함된 각 질문에 대해 가장 유사한 3개의 문서를 `SentenceTransformer` 모델을 사용하여 찾았습니다. 이 과정에서 Cosine Similarity를 사용하여 각 질문과 문서 간의 유사도를 계산하였습니다.

```python
import json
import os
import pandas as pd
from tqdm import tqdm  # 추가된 라이브러리
from sentence_transformers import SentenceTransformer, util

# Load pre-trained sentence transformer model
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# Define file paths
documents_file_path = '/home/data/documents.jsonl'
test_file_path = '/home/data/test.jsonl'
output_file_path = 'gt.csv'

# Load documents data
documents = []
with open(documents_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        documents.append(json.loads(line))

# Load test data
tests = []
with open(test_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        tests.append(json.loads(line))

# Extract content from documents
document_contents = {doc['docid']: doc['content'] for doc in documents}

# Function to compute cosine similarity
def get_top3_similar_documents(query, document_contents):
    # Encode the query and document contents
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(list(document_contents.values()), convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    
    # Get top 3 document indices
    top3_indices = cosine_scores.topk(3).indices.cpu().numpy()
    
    # Map indices back to docids
    docids = list(document_contents.keys())
    top3_docids = [docids[idx] for idx in top3_indices]
    
    return top3_docids

# Prepare gt.csv data (5개 레코드 단위로만 GT 대상)
gt_data = []

# tqdm을 사용해 진행 상황 표시
for idx, test in tqdm(enumerate(tests), total=len(tests)):
    if (idx + 1) % 5 == 1:  # 1, 6, 11, 16, ... 등 선택
        test_id = test['test_id']
        query = test['msg'][0]['content']
        
        # Get top 3 similar documents
        top3_docids = get_top3_similar_documents(query, document_contents)
        
        # Store the result
        gt_data.append({'test_id': test_id, 'doc_ids': ','.join(top3_docids)})

# Convert to DataFrame and save as CSV
gt_df = pd.DataFrame(gt_data)
gt_df.to_csv(output_file_path, index=False)

print(f"gt.csv has been created with top 3 similar documents for every 5th query.")

```

2. **Ground Truth 저장**: 각 질문에 대해 상위 3개의 문서를 Ground Truth로 저장하였고, 이를 CSV 파일로 변환하여 `gt.csv` 파일로 저장하였습니다. 이 파일은 이후 모델 평가에 사용됩니다.

```csv
test_id,doc_ids
01,42508ee0-c543-4338-878e-d98c6babee66,7a3e9dc2-2572-4954-82b4-1786e9e48f1f
...
```

#### 3. **MAP 계산**

1. **MAP (Mean Average Precision) 계산**: 생성된 Ground Truth (`gt.csv`)와 모델의 예측 결과를 바탕으로 MAP 점수를 계산하기 위한 Python 코드를 작성하였습니다. 각 질문에 대해 예측된 상위 3개의 문서가 Ground Truth와 얼마나 일치하는지를 평가합니다.
   
2. **MAP 계산 과정**: 각 질문에 대해 평균 정밀도(Precision)를 계산하고, 이를 기반으로 최종 MAP 점수를 계산하였습니다. 

```python
def calc_map(gt, pred):
    sum_average_precision = 0

    for j in pred:
        eval_id = j["eval_id"]

        if eval_id in gt and gt[eval_id]:
            hit_count = 0
            sum_precision = 0

            for i, docid in enumerate(j["topk"][:3]):
                if docid in gt[eval_id]:
                    hit_count += 1
                    sum_precision += hit_count / (i + 1)

            average_precision = sum_precision / hit_count if hit_count > 0 else 0
        else:
            average_precision = 0 if j["topk"] else 1

        sum_average_precision += average_precision

    return sum_average_precision / len(pred)
```

3. **최종 결과**: `gt.csv`와 모델의 Prediction 파일을 비교하여 MAP 점수를 출력하였습니다. 이를 통해 모델이 질문에 대한 적절한 문서를 얼마나 정확하게 예측했는지 평가할 수 있었습니다.

```bash
MAP Score: 0.7654
```

### 4. **데이터 처리 결론**

이번 데이터 처리 과정에서는 `documents.jsonl`에서 생성된 질문과 문서 간의 연관성을 Cosine Embedding Loss로 측정하였고, 이를 기반으로 Positive와 Negative 질문 쌍을 생성하여 `test.jsonl` 파일을 만들었습니다. 이 데이터를 활용해 Ground Truth 파일인 `gt.csv`를 생성하고, 이를 바탕으로 모델의 성능을 평가하는 MAP 점수를 계산하였습니다.

이 과정에서 중요한 점은 Cosine Embedding Loss를 통해 질문과 문서 간의 관련성을 정량적으로 평가할 수 있었으며, 이를 바탕으로 보다 정확한 질의응답 시스템을 구축하는 데 필요한 평가 방법론을 수립할 수 있었습니다.
```

## 5. Modeling

## 5.1 베이스라인 모델 코드 분석 (팀 전체)

이 코드는 **RAG (Retrieval-Augmented Generation)** 시스템을 구현하는 데 필요한 다양한 요소들을 처리하는 베이스라인 소스입니다. 주요 단계는 문서 임베딩 생성, Elasticsearch 기반의 검색 기능 구현, 그리고 OpenAI의 GPT 모델을 사용한 질의응답 시스템을 결합하여 과학 상식 관련 질문에 답변을 생성하는 것입니다. 아래에서 주요 구성 요소를 분석해보겠습니다.

### 1. **Sentence Embedding 생성**
- **모델 초기화**: `SentenceTransformer`를 사용하여 한국어 임베딩을 생성할 수 있는 `snunlp/KR-SBERT-V40K-klueNLI-augSTS` 모델을 사용합니다.
- **임베딩 생성**: `get_embedding` 함수는 주어진 텍스트(문서나 질문)를 임베딩으로 변환합니다. 이를 통해 문장 간의 유사도를 계산할 수 있습니다.
- **배치 단위 임베딩 생성**: `get_embeddings_in_batches` 함수는 문서 리스트를 입력으로 받아 배치 단위로 임베딩을 생성합니다. 대규모 데이터 처리 시 효율성을 높이기 위해 배치로 나눠서 처리합니다.

### 2. **Elasticsearch 설정 및 색인 작업**
- **Elasticsearch 클라이언트**: `Elasticsearch` 객체를 사용하여 ES 클러스터에 연결합니다. 기본 인증 정보와 SSL 인증서를 사용하여 보안 연결을 설정합니다.
- **색인 설정 및 생성**: `create_es_index` 함수는 `nori` 분석기를 포함한 텍스트 필드와 `dense_vector` 필드를 사용하여 벡터 유사도를 계산할 수 있는 인덱스를 생성합니다.
- **문서 임베딩 색인**: 문서와 그 임베딩을 `bulk_add` 함수를 사용하여 Elasticsearch에 대량으로 색인합니다. 이 작업을 통해 문서 검색 및 벡터 유사도 검색에 사용할 수 있는 준비가 완료됩니다.

### 3. **검색 기능 구현**
- **역색인 검색**: `sparse_retrieve` 함수는 주어진 쿼리 문자열을 역색인 방식으로 검색합니다. 이 방식은 주로 TF-IDF와 같은 전통적인 정보 검색 방법을 사용하여 문서의 내용을 기반으로 검색합니다.
- **벡터 유사도 검색**: `dense_retrieve` 함수는 문서 임베딩과 쿼리 임베딩 간의 유사도를 계산하여 KNN (최근접 이웃) 방식으로 검색합니다. 이는 의미적 유사도를 기반으로 문서를 검색합니다.

### 4. **RAG 구현 (Retrieval-Augmented Generation)**
- **OpenAI API 설정**: GPT-3.5를 활용하여 질문에 대한 답변을 생성합니다. API 호출 시 질의 분석 및 검색 호출을 위해 별도의 프롬프트(`persona_qa`, `persona_function_calling`)를 설정합니다.
- **Function Calling**: 과학 상식에 관련된 질문을 처리하는 `function_calling` 기능을 통해, 필요할 때 검색 엔진을 호출하여 적합한 문서를 찾고, 이를 기반으로 LLM이 답변을 생성할 수 있게 합니다.
- **Answer 생성**: `answer_question` 함수는 검색된 문서들(`sparse_retrieve`나 `dense_retrieve` 결과)을 바탕으로 LLM이 답변을 생성하게 합니다. 검색된 문서가 없을 경우에도 LLM이 답변을 생성하도록 설정되어 있습니다.

### 5. **평가 및 결과 저장**
- **RAG 시스템 평가**: `eval_rag` 함수는 평가 데이터(`eval.jsonl`)를 읽어서 각 질문에 대해 검색 및 답변을 생성한 후 결과를 파일로 저장합니다. 대회 포맷에 맞게 `sample_submission.csv` 파일을 생성합니다.

### 6. **코드 흐름 요약**
1. **문서 임베딩 생성 및 Elasticsearch 색인**: 
   - `documents.jsonl` 파일에서 문서의 내용을 읽어 임베딩을 생성하고, 이를 Elasticsearch에 색인합니다.
   
2. **질문에 대한 답변 생성 (RAG)**: 
   - 사용자의 질문에 대해 역색인 검색 또는 벡터 유사도 검색을 통해 관련 문서를 찾고, 이를 바탕으로 GPT 모델이 답변을 생성합니다.
   
3. **평가 및 제출**: 
   - 평가 데이터를 사용해 RAG 시스템의 성능을 평가하고, 그 결과를 CSV 파일로 저장하여 대회 제출 포맷에 맞춥니다.

### 7. **Elasticsearch 설정의 세부 내용**
- **Nori 분석기**: 한국어 처리를 위해 `nori` 분석기가 사용되었습니다. 이는 형태소 분석을 통해 한국어 문장을 효과적으로 색인할 수 있도록 합니다.
- **Dense Vector 필드**: 문서의 벡터 임베딩을 `dense_vector` 필드로 저장하여 벡터 유사도 검색을 가능하게 합니다. L2 노름 기반 유사도를 계산하여 KNN 방식으로 검색을 수행합니다.

이 코드는 한국어로 된 과학 상식 질문에 대해 관련 문서를 검색하고, 이를 바탕으로 GPT 모델이 답변을 생성하는 RAG 시스템의 기본적인 구조를 구현한 것입니다. Elasticsearch를 통해 효율적인 문서 검색을 수행하고, LLM을 활용하여 검색된 문서를 바탕으로 답변을 생성하는 데 중점을 두고 있습니다.

## 5.2. 개인 모델 실험 (박석)

두 가지 실험을 설명합니다. 두 실험 모두 **Elasticsearch**와 **SentenceTransformer**를 결합하여 문서 임베딩 생성 및 검색을 수행한 후, **OpenAI**의 GPT 모델을 사용하여 최종 답변을 생성하는 **RAG** (Retrieval-Augmented Generation) 시스템을 기반으로 진행되었습니다.

---

### 5.2.1 첫 번째 실험

#### 5.2.1.1 목표
첫 번째 실험의 목표는 문서의 임베딩을 생성하여 Elasticsearch에 색인하고, 이를 바탕으로 벡터 유사도 검색과 역색인 검색을 수행하는 것입니다. 검색된 문서를 바탕으로 GPT-4 모델을 사용하여 답변을 생성하고 평가합니다. 

#### 5.2.1.2 주요 구현

1. **임베딩 생성 및 색인**  
   - **SentenceTransformer** 모델을 사용하여 문서의 임베딩을 생성합니다. 
   - 생성된 임베딩을 Elasticsearch의 `dense_vector` 필드에 저장하고, 문서의 내용을 `text` 필드에 색인합니다.

2. **검색 기능 구현**  
   - **역색인 검색** (`sparse_retrieve`): 텍스트 기반으로 Elasticsearch의 역색인을 사용하여 관련 문서를 검색합니다.
   - **벡터 유사도 검색** (`dense_retrieve`): 생성된 임베딩을 사용하여 KNN 방식으로 벡터 유사도를 계산한 후 관련 문서를 검색합니다.

3. **RAG 시스템 구현**  
   - GPT-4를 활용하여 사용자의 질문에 대한 답변을 생성합니다.
   - 검색된 문서를 바탕으로 LLM이 답변을 생성하며, 만약 적합한 검색 결과가 없는 경우 "답변 불가" 메시지를 생성합니다.

#### 5.2.1.3 실험 과정 및 결과

- **문서 임베딩 색인**: `documents.jsonl` 파일에서 문서 임베딩을 생성하여 Elasticsearch에 색인한 후, 질문을 바탕으로 벡터 유사도 및 역색인 검색을 수행합니다.
  
- **검색 예제**:  
  ```bash
  test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
  ```

- **검색 결과**:  
  검색 결과를 통해 문서를 추출하고, GPT-4 모델을 사용하여 최종 답변을 생성합니다.

#### 5.2.1.4 결론
첫 번째 실험에서는 GPT-4 모델을 사용하여 RAG 시스템을 구현하였으며, 역색인과 벡터 유사도를 결합한 검색을 통해 정확한 문서 검색과 답변 생성을 목표로 하였습니다.
이 실험에서는 ChatGPT 버전을 3.5에서 4.0으로 변경했으며, 한글로 된 프롬프트를 영문으로 바꾸어 진행했습니다. 결과로 MAP 수치가 0.7030 에서 0.7394 로 향상되었습니다. 

---

### 5.2.2 두 번째 실험

#### 5.2.2.1 목표
두 번째 실험은 첫 번째 실험을 개선한 것으로, **동의어 필터**를 추가하고, **하이브리드 검색**(역색인 + 벡터 검색)을 결합하여 검색 성능을 향상시키고자 하였습니다.

#### 5.2.2.2 주요 구현

1. **동의어 필터 추가**  
   - 과학 및 기술 용어의 동의어를 포함한 파일을 생성하여, Elasticsearch에 동의어 필터로 설정합니다. 이를 통해 동일한 개념을 가진 여러 표현을 효과적으로 검색할 수 있습니다.

2. **하이브리드 검색 구현**  
   - **하이브리드 검색** (`hybrid_retrieve`): 역색인과 벡터 유사도 검색 결과를 결합하여 문서 검색을 수행합니다. 두 검색 방법의 점수를 결합하여 최종 결과를 생성합니다.
   
3. **RAG 시스템**  
   - GPT-4를 사용하여 사용자의 질문에 대해 검색된 문서를 바탕으로 답변을 생성합니다.

#### 5.2.2.3 실험 과정 및 결과

1. **동의어 필터 적용**: 
   - **동의어 사전**을 통해 과학 기술 용어의 다양한 표현을 처리합니다. 예를 들어, "금성"과 "비너스", "인공지능"과 "AI" 등의 동의어를 동일하게 처리하여 검색 성능을 높입니다.

2. **하이브리드 검색 결과**:  
   하이브리드 검색을 통해 벡터 유사도와 역색인 검색 결과를 결합하고, 점수를 평균화하여 최종 검색 결과를 제공합니다.

#### 5.2.2.4 결론
두 번째 실험에서는 동의어 필터와 하이브리드 검색 방식을 도입하여 검색 성능을 개선하였습니다. 이를 통해 다양한 표현을 보다 효과적으로 검색할 수 있었고, RAG 시스템에서의 문서 검색 정확도를 높였습니다. 결과로 MAP 수치가 0.7030 에서 0.7212 로 향상되었습니다. 

---

### 5.2.3 실험 정리

두 실험 모두 문서 임베딩 생성 및 Elasticsearch를 활용한 검색 기능을 바탕으로 한 RAG 시스템의 성능을 개선하려는 시도였습니다. 첫 번째 실험에서는 기본적인 벡터 유사도 검색과 역색인 검색을 구현하였고, 두 번째 실험에서는 동의어 필터와 하이브리드 검색을 추가하여 검색 정확도를 향상시켰습니다.

### 5.3. 개인 모델 실험 (백경탁)

### 5.3. 개인 모델 실험 (백경탁)

백경탁님의 실험은 **과학 상식 문서 검색과 질의응답 시스템(RAG)** 성능을 개선하기 위한 여러 가지 시도를 통해 이루어졌습니다. 주요 작업은 임베딩 모델 변경, 검색 방식 최적화, 프롬프트 엔지니어링, 동의어 필터 추가 등을 통해 성능을 개선하는 것이 목표였습니다.

---

### 5.3.1 데이터

#### 5.3.1.1 학습 데이터 - `documents.jsonl`
학습 데이터는 4,200여 개의 과학 상식 문서로 구성되었습니다. 각 문서는 고유 ID(`docid`), 출처(`src`), 그리고 문서 내용(`content`)으로 이루어져 있습니다. 이 문서들은 과학 상식 기반으로 RAG 시스템에서 검색될 수 있도록 색인됩니다.

- **예시 문서**:
  ```json
  {
    "docid": "42508ee0-c543-4338-878e-d98c6babee66",
    "src": "ko_mmlu__nutrition__test",
    "content": "건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다..."
  }
  ```

#### 5.3.1.2 평가 데이터 - `eval.jsonl`
평가 데이터는 총 220개의 메시지로 이루어져 있으며, 그중 20개는 멀티턴 대화를 포함하고 20개는 일상 대화가 포함됩니다. 이는 검색뿐만 아니라 질의의 의도를 얼마나 잘 분석하는지를 평가하는 데 사용됩니다.

- **예시 일상 대화**:
  ```json
  {"eval_id": 276, "msg": [{"role": "user", "content": "요새 너무 힘들다."}]}
  ```

- **예시 멀티턴 대화**:
  ```json
  {"eval_id": 107, "msg": [{"role": "user", "content": "기억 상실증 걸리면 너무 무섭겠다."}, {"role": "assistant", "content": "네 맞습니다."}, {"role": "user", "content": "어떤 원인 때문에 발생하는지 궁금해."}]}
  ```

---

### 5.3.2 실험 단계 및 성과

#### 5.3.2.1 BaseLine Code 실행 및 점수 확인
- 초기 **MAP 점수**: 0.7152
- 과학 지식 질문임에도 검색이 이루어지지 않는 경우 확인됨.
  - 예: `Dmitri Ivanovsky`, `merge sort algorithm`, `benefits of good sleep`

#### 5.3.2.2 모델 변경에 따른 성능 테스트
- **LLM 모델 변경**: `gpt-4o-mini` 사용
- **임베딩 모델 변경**:
  - `jhgan/ko-sbert-multitask`
  - `jhgan/ko-sroberta-multitask`

  그러나 **성능 개선 없음**.

#### 5.3.2.3 데이터 확인 및 테스트 데이터 생성
- **ChatGPT**를 활용하여 문서에 대한 질문을 생성하고, 평가 데이터의 답변을 수기 작성하여 74개의 테스트 샘플을 생성.

#### 5.3.2.4 Prompt Engineering
- 특정 과학 지식 질문에서 검색 함수가 호출되지 않는 문제를 해결하기 위해 **프롬프트 수정**:
  ```python
  persona_function_calling = """
  ## Role: 과학 상식 전문가

  ## Instruction
  - 사용자가 지식에 대해 질문하면 반드시 search 함수를 호출해야 한다.
  - 나머지 일상 대화 메시지에는 반드시 함수를 호출하지 않고 적절한 대답을 생성한다.
  """
  ```

- 프롬프트 수정 후 **MAP 성능 향상**:
  - 0.7152 → **0.7515**

#### 5.3.2.5 Hybrid 검색 (Sparse Vector + Dense Vector)
- **임베딩 모델**:
  - `snunlp/KR-SBERT-V40K-klueNLI-augSTS` 사용 시, 역색인 검색: 0.7515 / 벡터 유사도 검색: 0.4409
  - `jhgan/ko-sroberta-multitask` 사용 시, 역색인 검색: 0.7545 / 벡터 유사도 검색: 0.7

- **결론**: `jhgan/ko-sroberta-multitask` 모델을 사용한 하이브리드 검색 방식이 최적의 성능을 보였으며, 이를 기준으로 실험을 계속 진행.

#### 5.3.2.6 Embedding Model 변경 - 고성능 모델 탐색
다양한 임베딩 모델을 테스트하여 성능을 비교:
- `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- `jhgan/ko-sroberta-multitask`
- `ddobokki/klue-roberta-base-nli-sts`
- `dragonkue/bge-m3-ko` **(최고 성능 모델)**

- **모델 변경 후 성능 개선**:
  - `jhgan/ko-sroberta-multitask`: **0.8371**
  - `dragonkue/bge-m3-ko`: **0.8621** (0.025 향상)

#### 5.3.2.7 검색 알고리즘 및 Synonym 추가
- **BM25** → **LMJelinekMercer**로 검색 알고리즘 변경:
  - **0.8621** → **0.8682** (0.0061 향상)

- **동의어 필터** 추가:
  - `복숭아, 복숭아나무`, `리보솜, 리보오솜`, `잠, 수면` 등 과학 관련 동의어를 추가하여 검색 성능을 향상시킴.
  - **0.8682** → **0.8773** (0.0091 향상)

---

### 5.3.3 실험 결론

백경탁님의 실험은 **임베딩 모델 변경**, **프롬프트 엔지니어링**, **동의어 필터 추가** 등의 시도를 통해 RAG 시스템의 성능을 지속적으로 개선하는 과정이었습니다. 

- **최종 MAP 점수**: **0.8773**
- 성능 향상을 위해 **hybrid 검색**과 **고성능 임베딩 모델** 사용이 주요 기여 요소로 작용하였습니다.

### 5.4. 개인 모델 실험 (위효연)

위효연님은 이번 스터디 동안 그동안 뒤쳐졌던 수업을 따라가는데 중점을 맞추고 프롬프트 엔지니어링과 한국어 임베딩 언어모델들의 평가지표를 찾는데 집중하였습니다.

프롬프트 엔지니어링을 통한 질문 생성
### Positive Pair (CosineEmbeddingLoss = 1)
1. 원자핵이 무엇인가요? / 원자의 중심에 위치한 원자핵이란 무엇인가요?
2. 지구의 자전이란 무엇인가요? / 지구가 자전하는 과정은 무엇인가요?
3. 세포가 하는 역할은 무엇인가요? / 세포가 생물체에서 어떤 기능을 하는지 설명해주세요.
4. 중력의 정의는 무엇인가요? / 중력은 어떤 힘인가요?
5. 물질의 상태 변화는 무엇인가요? / 고체, 액체, 기체로 물질이 변화하는 것을 상태 변화라고 하나요?
6. 태양 에너지는 어떻게 사용되나요? / 태양 에너지를 활용하는 방법은 무엇인가요?
7. 식물이 광합성을 하는 이유는 무엇인가요? / 식물이 광합성을 통해 에너지를 얻는 이유는 무엇인가요?
8. 열 에너지는 어떻게 발생하나요? / 열 에너지가 발생하는 원리는 무엇인가요?
9. 바이러스는 어떻게 전염되나요? / 바이러스가 다른 개체로 전염되는 방식은 무엇인가요?
10. 달은 왜 지구 주위를 도나요? / 달이 지구 궤도를 도는 이유는 무엇인가요?

### Negative Pair (CosineEmbeddingLoss = -1)
1. 원자핵이 무엇인가요? / 바다의 염도는 왜 중요한가요?
2. 지구의 자전이란 무엇인가요? / 유전자는 생물체에 어떤 영향을 주나요?
3. 세포가 하는 역할은 무엇인가요? / 블랙홀은 어떻게 형성되나요?
4. 중력의 정의는 무엇인가요? / 산소는 호흡 과정에서 어떤 역할을 하나요?
5. 물질의 상태 변화는 무엇인가요? / 광합성은 어떻게 이루어지나요?
6. 태양 에너지는 어떻게 사용되나요? / 전자기파는 어떻게 발생하나요?
7. 식물이 광합성을 하는 이유는 무엇인가요? / 유전자가 어떻게 전달되나요?
8. 열 에너지는 어떻게 발생하나요? / 원자는 어떤 구조로 이루어져 있나요?
9. 바이러스는 어떻게 전염되나요? / 에너지는 어떻게 보존되나요?
10. 달은 왜 지구 주위를 도나요? / 태풍은 어떻게 발생하나요?

- 작업을 진행하면서 결과물에대해 팀원들끼리 합의가 안됬을시(인간끼리 합의가 안됬을때)→ 결국 테스트케이스가 부족해서 이다. 테스트케이스로 증명해야된다.
  
**참조 인용: https://github.com/su-park/mteb_ko_leaderboard?tab=readme-ov-file#mteb-평가

MTEB 평가

텍스트 임베딩의 성능을 다면적으로 평가하기 위한 벤치마크 셋
평가셋이 대부분 영어 위주로 구성되어 아래에서는 한글 데이터셋만 대상으로 성능 측정
Task	Definition	Metric
BitextMining	task of finding parallel sentences in two languages	F1
Classification	task of assigning a label to a text	Accuracy
Clustering	task of grouping similar documents together	Validity Measure (V-measure)
PairClassification	task of determining whether two texts are similar	Average Precision
Reranking	task of reordering a list of documents to improve relevance	Mean Average Precision
Retrieval	task of finding relevant documents for a query	nDCG@10
STS	task of determining how similar two texts are	Spearman Correlation
성능 평가

no.	Model	BitextMining kor-eng	BitextMining eng-kor	Classification	MultiLabel Classification	Clustering	PairClassification	Reranking	Retrieval	STS
1	SFR-Embedding-2_R	94.49	95.79	68.74	9.80	53.78	60.42	48.17	73.18	81.16
2	gte-Qwen2-7B-instruct	94.32	95.11	72.33	9.22	53.27	74.15	50.05	74.96	85.70
3	bge-multilingual-gemma2	94.78	95.56	69.06	13.14	40.10	68.98	37.06	58.87	83.83
4	e5-mistral-7b-instruct	94.53	95.94	67.47	9.08	58.02	63.08	54.92	63.08	84.54
5	jina-embeddings-v3	91.70	91.95	63.17	12.03	39.83	56.93	42.95	74.98	83.50
6	LLM2Vec-Meta-Llama-3-supervised	94.37	95.33	68.27	9.37	47.01	65.45	52.63	68.43	83.44
7	multilingual-e5-large-instruct	94.91	95.83	67.03	10.20	55.76	61.18	52.86	74.52	85.15
8	multilingual-e5-large	93.64	94.33	62.96	9.18	39.02	57.55	54.87	73.47	80.62
9	bge-m3	94.53	95.81	63.48	10.92	38.04	61.20	59.98	72.29	83.13
10	ko-sroberta-multitask	69.66	57.56	61.62	8.93	36.41	65.64	48.33	60.98	85.39
bge-m3-korean	93.32	94.82	58.89	9.21	23.967	64.13	52.44	67.94	85.43
KoE5	93.00	93.92	64.56	11.57	43.24	57.37	54.14	73.69	82.15
no.	Model	Tatoeba BitextMining kor-eng	Flores BitextMining eng-kor	Flores BitextMining kor-eng	NTREX BitextMining eng-kor	NTREX BitextMining kor-eng	IWSLT2017 BitextMining eng-kor	IWSLT2017 BitextMining kor-eng	MassiveIntent Classification	MassiveScenario Classification	Klue-TC Classification	SIB200 Classification	MultilingualSentiment Classification	KorHate Classification	KorSarcasm Classification	KorHateSpeechML MultiLabel Classification	SIB200ClusteringS2S Clustering	Klue-NLI PairClassification	PawsX PairClassification	MIRACL Reranking	Ko-StrategyQA Retrieval	XPQA Retrieval	PublicHealthQA Retrieval	Belebele Retrieval	MIRACL Retrieval	STS17 STS	KorSTS STS	Klue-STS STS
1	SFR-Embedding-2_R	91.04	100.00	100.00	98.81	99.33	88.57	87.59	75.40	85.67	58.50	80.19	79.68	45.57	56.15	9.80	53.78	67.49	53.34	48.17	77.27	37.24	86.35	91.85	47.01	78.86	77.39	87.24
2	gte-Qwen2-7B-instruct	91.40	100.00	100.00	98.83	99.09	86.51	86.77	79.25	86.42	69.20	86.12	78.22	46.94	60.13	9.22	53.27	79.57	68.72	50.05	81.07	37.81	86.15	94.79		83.86	83.47	89.77
3	bge-multilingual-gemma2	92.35	100.00	99.86	98.16	98.99	88.51	87.90	73.56	81.10	63.61	77.15	81.69	48.84	57.46	13.14	40.10	80.12	57.84	37.06	51.33	37.67	65.89	80.59	19.57	81.38	81.61	88.49
4	e5-mistral-7b-instruct	91.28	100.00	100.00	98.68	99.13	89.15	87.71	69.87	74.39	60.15	84.16	80.74	45.97	57.01	9.08	58.02	73.02	53.14	54.92	79.30	39.22	88.71	92.61		83.69	82.27	87.65
5	jina-embeddings-v3	86.93	97.89	99.07	93.05	95.63	84.92	85.16	62.32	70.16	58.72	74.75	74.12	42.23	59.87	12.03	39.83	62.11	51.74	42.95	78.35	41.67	87.92	91.98		84.00	82.53	83.98
6	LLM2Vec-Meta-Llama-3-supervised	90.94	100.00	99.86	98.07	98.82	87.92	87.86	75.15	79.87	60.83	79.85	79.30	44.58	58.33	9.37	47.01	77.57	53.33	52.63	70.44	36.58	79.76	86.95	34.90	81.22	81.06	88.03
7	multilingual-e5-large-instruct	91.76	100.00	100.00	99.08	99.39	88.42	88.49	64.15	70.50	62.24	81.71	78.29	45.30	58..17	10.20	55.76	70.54	51.81	52.86	79.86	39.74	84.87	93.60		84.30	82.71	88.44
8	multilingual-e5-large	89.70	99.42	99.86	96.46	98.93	87.11	86.08	63.74	70.66	59.68	74.60	72.57	43.18	56.26	9.18	39.02	63.42	51.68	54.87	79.82	36.99	82.88	94.18	65.56	81.04	79.24	81.59
9	bge-m3	90.44	99.86	100.00	98.60	99.08	88.96	88.59	66.53	72.90	54.67	71.91	78.16	43.38	56.79	10.92	38.04	70.05	52.34	59.98	79.40	36.15	80.41	93.18	70.14	81.42	80.26	87.70
10	ko-sroberta-multitask	61.05	72.31	86.39	52.16	71.95	48.20	59.26	64.80	70.12	52.10	69.75	73.82	43.67	57.11	8.93	36.41	78.38	52.89	48.33	65.10	27.96	69.21	81.63	36.69	86.46	85.58	84.13
bge-m3-korean	89.36	99.73	100.00	98.21	98.20	86.54	85.75	66.28	72.80	41.23	59.95	74.98	41.34	55.69	9.21	23.97	75.95	52.32	52.44	75.27	31.66	77.55	87.31		85.15	83.34	87.81
KoE5	89.41	99.47	99.73	96.78	97.39	85.51	85.47	66.35	74.13	63.74	74.85	73.05	43.92	55.90	11.57	43.24	61.71	53.03	54.14	79.76	36.13	84.49	94.41		81.35	79.45	85.65



참고자료. 

- https://arxiv.org/search/   (논문을 간결하게 보기위해 metric을 찾을것)
- https://www.elastic.co/search-labs/blog/evaluating-rag-metrics
- https://cookbook.openai.com/


### 5.5. 개인 모델 실험 (남상이)
TBD

## 6. Result

#### Final(Public) - Rank 7

![UpStage-IR4-LeaderBoard-Final-Public](./images/UpStage-IR4-LeaderBoard-Final-Public.png)

#### Final(Private) - Rank 7

![UpStage-IR4-LeaderBoard-Final-Private](./images/UpStage-IR4-LeaderBoard-Final-Private.png)


#### Submit history

**bc44_jina_hy...k_syn**, **bc43_bge-m3-...k_syn** was submitted!!! 

| Model Name                 | Submitter | MAP   | MAP (Final) | MRR   | MRR (Final) | Created at          | Phase    |
|----------------------------|-----------|-------|-------------|-------|-------------|---------------------|----------|
| ir4-parksurk-exp02          | 박석      | 0.7212| 0.6636      | 0.7242| 0.6652      | 2024.10.24 17:41    | Complete |
| sample_submission           | 위효연    | -     | -           | -     | -           | 2024.10.24 17:39    | Failed   |
| ir4-parksurk...mpt01        | 박석      | 0.7394| 0.6818      | 0.7439| 0.6848      | 2024.10.24 15:12    | Complete |
| bc45_bge-m3-...n_sz5        | 백경탁    | 0.8636| 0.8545      | 0.8667| 0.8561      | 2024.10.24 12:59    | Complete |
| bc44_jina_hy...k_syn        | 백경탁    | 0.8545| 0.8568      | 0.8576| 0.8561      | 2024.10.23 20:39    | Complete |
| bc43_bge-m3-...k_syn        | 백경탁    | 0.8773| 0.8576      | 0.8803| 0.8576      | 2024.10.23 19:02    | Complete |
| bc42_bge-m3-...k_syn        | 백경탁    | 0.8727| 0.8636      | 0.8758| 0.8636      | 2024.10.23 15:37    | Complete |
| bc41_bge-m3-...chlet        | 백경탁    | 0.8538| 0.8606      | 0.8545| 0.8606      | 2024.10.23 15:12    | Complete |
| bc40_bge-m3-...linek        | 백경탁    | 0.8682| 0.8682      | 0.8697| 0.8682      | 2024.10.23 14:17    | Complete |
| bc39_bge-m3-...linek        | 백경탁    | 0.8485| 0.8636      | 0.8530| 0.8636      | 2024.10.23 13:05    | Complete |
| bc38_bge-m3-...5_d10        | 백경탁    | 0.8621| 0.8652      | 0.8636| 0.8652      | 2024.10.23 12:38    | Complete |
| bc37_klue-sr...2_d10        | 백경탁    | 0.8295| 0.7955      | 0.8348| 0.7985      | 2024.10.23 10:41    | Complete |
| bc36_distilb...1_d10        | 백경탁    | 0.7697| 0.7121      | 0.7758| 0.7152      | 2024.10.22 19:39    | Complete |
| bc35_ko-srob...5_d10        | 백경탁    | 0.8371| 0.7788      | 0.8394| 0.7818      | 2024.10.22 13:21    | Complete |
| bc34_ko-srob...8_sz5        | 백경탁    | 0.7606| 0.7167      | 0.7667| 0.7182      | 2024.10.22 09:47    | Complete |
| bc33_ko-srob...0_sz5        | 백경탁    | 0.8379| 0.7864      | 0.8439| 0.7879      | 2024.10.22 09:29    | Complete |
| bc32_ko-srob...10_k5        | 백경탁    | 0.8333| 0.7864      | 0.8394| 0.7879      | 2024.10.21 21:19    | Complete |
| bc31_ko-srob...2_d10        | 백경탁    | 0.8371| 0.8045      | 0.8424| 0.8091      | 2024.10.21 19:52    | Complete |
| bc30_ko-srob..._nori        | 백경탁    | 0.8280| 0.8182      | 0.8303| 0.8212      | 2024.10.21 05:57    | Complete |
| bc29_klue-sr...linek        | 백경탁    | 0.8136| 0.8295      | 0.8227| 0.8303      | 2024.10.21 05:23    | Complete |
| bc28_ko-srob...linek        | 백경탁    | 0.8439| 0.8409      | 0.8500| 0.8394      | 2024.10.21 05:03    | Complete |
| bc27_ko-srob...10_k5        | 백경탁    | 0.8288| 0.8273      | 0.8333| 0.8288      | 2024.10.21 04:10    | Complete |
| bc26_ko-srob...10_k5        | 백경탁    | 0.8455| 0.8348      | 0.8500| 0.8379      | 2024.10.21 03:40    | Complete |
| bc25_ko-srob...10_k5        | 백경탁    | 0.8303| 0.8394      | 0.8379| 0.8455      | 2024.10.21 03:21    | Complete |
| bc24_ko-sber...2_d10        | 백경탁    | 0.7788| 0.8053      | 0.7833| 0.8061      | 2024.10.20 22:04    | Complete |
| bc23_ko-srob...linek        | 백경탁    | 0.7591| 0.7591      | 0.7636| 0.7576      | 2024.10.20 21:41    | Complete |
| bc22_ko-srob...linek        | 백경탁    | 0.8318| 0.8182      | 0.8379| 0.8182      | 2024.10.20 12:17    | Complete |
| bc21_ko-srob...linek        | 백경탁    | 0.8356| 0.8371      | 0.8364| 0.8379      | 2024.10.20 11:52    | Complete |
| bc20_ko-srob...chlet        | 백경탁    | 0.7758| 0.8045      | 0.7758| 0.8061      | 2024.10.20 11:20    | Complete |
| bc19_ko-srob...5_d10        | 백경탁    | 0.8341| 0.8402      | 0.8379| 0.8455      | 2024.10.19 11:42    | Complete |
| bc18_ko-srob...5_d20        | 백경탁    | 0.7561| 0.7894      | 0.7621| 0.7909      | 2024.10.19 11:24    | Complete |
| bc17_ko-srob...2_d10        | 백경탁    | 0.8462| 0.8447      | 0.8485| 0.8500      | 2024.10.19 10:55    | Complete |
| bc16_ko-srob...5_d20        | 백경탁    | 0.7606| 0.7121      | 0.7667| 0.7136      | 2024.10.19 10:19    | Complete |
| bc15_ko-srob...0_d10        | 백경탁    | 0.7545| 0.6985      | 0.7606| 0.7000      | 2024.10.18 19:49    | Complete |
| bc11_ko-srob...0_d20        | 백경탁    | 0.7606| 0.7121      | 0.7667| 0.7136      | 2024.10.17 15:03    | Complete |
| bc11_ko-srob...05_d1        | 백경탁    | 0.7470| 0.7977      | 0.7470| 0.7970      | 2024.10.17 13:58    | Complete |
| bc12_KR-SBERT_dense         | 백경탁    | 0.4409| 0.5038      | 0.4394| 0.5045      | 2024.10.17 13:34    | Complete |
| bc11_ko-srob...parse        | 백경탁    | 0.7545| 0.7076      | 0.7606| 0.7091      | 2024.10.17 12:59    | Complete |
| basecode10_k...itask        | 백경탁    | 0.7000| 0.7348      | 0.7000| 0.7394      | 2024.10.17 11:47    | Complete |
| BaseCode9_pr...parse        | 백경탁    | 0.7515| 0.7076      | 0.7576| 0.7091      | 2024.10.16 20:24    | Complete |
| ir4-parksurk...eline        | 박석      | 0.7030| 0.6970      | 0.7076| 0.7015      | 2024.10.16 04:16    | Complete |
| gpt-3.5-turb..._must        | 백경탁    | 0.6864| 0.7091      | 0.6894| 0.7121      | 2024.10.08 17:58    | Complete |
| gpt-3.5-turb...ompt2        | 백경탁    | 0.6515| 0.6030      | 0.6561| 0.6061      | 2024.10.08 14:06    | Complete |
| gpt-3.5-turb...mpt01        | 백경탁    | 0.6394| 0.5788      | 0.6439| 0.5803      | 2024.10.08 09:04    | Complete |
| B_ko-srobert...itask        | 백경탁    | 0.7015| 0.7061      | 0.7076| 0.7106      | 2024.10.08 01:07    | Complete |
| Baseline_ko-...itask        | 백경탁    | 0.7015| 0.7061      | 0.7076| 0.7106      | 2024.10.08 01:01    | Complete |
| gpt-3.5-turb...itask        | 백경탁    | 0.7015| 0.6970      | 0.7076| 0.7015      | 2024.10.07 13:55    | Complete |
| baseline_gpt-4o-mini        | 백경탁    | 0.2879| 0.2879      | 0.2894| 0.2879      | 2024.10.07 11:36    | Complete |
| Baseline_gpt...-1106        | 백경탁    | 0.7152| 0.7136      | 0.7197| 0.7167      | 2024.10.05 20:39    | Complete |


### Mentoring history

- 멘토 : 김강민 멘토님

### 첫 번째 멘토링 (10월 8일)

**Q1. 테스트 데이터 관련 문제**  
- 문제: 테스트 데이터가 없어 어떤 부분이 잘못되었는지 알기 어려움.  
- 해결 방법: 데이터 자체의 문제인지 시스템의 문제인지를 명확히 분석해야 함.

**Q2. LLM 모델과 프롬프트 변경**  
- 실험: HuggingsFace LLM 모델과 프롬프트 변경 시도 (과학상식 → 세부 카테고리로 변경)  
- 결과: LLM 모델만 변경하는 것은 큰 효과가 없으며, 아키텍처 또는 컴포넌트 변경이 더 효과적일 것. 프롬프트 변경은 긍정적인 효과가 있을 수 있으며, 히스토리를 남기는 것이 좋음.  
- 팁: 데이터 중심(Data-Centric) 접근이 성능 개선에 효과적일 수 있음. 무료로 사용 가능한 API는 HuggingsFace에서 제공하는 모델을 Self-hosted 방식으로 사용할 수 있음.

**Q3. 테스트 세트 생성**  
- ChatGPT를 활용해 테스트 세트를 자동화하여 생성하는 것이 좋음. 손으로 만드는 것보다 효율적.

**Q5. RAG의 핵심 요소**  
- 가장 중요한 컴포넌트는 **프롬프트 엔지니어링**이며, RAG는 다양한 NLP 기술을 모두 활용해야 함.  
- 문서가 많을 때는 문서를 의미적으로 잘라서 처리하는 **천킹(chunking)** 방식을 시도해야 함.

**Q6. 비용 및 시간 문제 해결**  
- 검색 결과 필터링과 같은 전략을 통해 비용 및 시간을 최적화해야 함.

### 두 번째 멘토링 (10월 16일)

**검색엔진 관련**  
- **정확성**: 하이브리드 서치 및 동의어 사전 처리 기능이 필요함.  
- **속도**: 클라우드 인프라를 활용한 샤딩 및 분산처리 필요.

**최신 기법 리서치 방법**  
- arXiv와 HuggingFace에서 최신 기법을 리서치할 것을 권장.

**평가 방법**  
- **정성적 수치**: 휴먼 평가  
- **정량적 수치**: 수학적 평가를 통해 성능 평가.

### 세 번째 멘토링 (10월 22일)

**Q1. 최신 논문 적용**  
- 성능이 좋은 데이터셋과 임베딩 모델을 확인하고, 원인 분석을 통해 문제를 해결할 것을 추천.

**Q2. 다양한 실험 시도**  
- Sparse vs Dense 실험은 GPT를 사용하여 자동으로 처리하게 할 수 있으며, 질문에 대한 여러 답변 생성 시 **Random seed**와 옵션을 변경하여 해결할 수 있음.

**Q3. Question Generation 관련**  
- Pororo 라이브러리를 활용할 수 있으나 최신 GPT 모델을 사용하는 것이 더 효과적일 수 있음.

**백경탁 실험 결과**  
- 베이스라인: 0.7251 → 백경탁: 0.8461  
- 프롬프트 변경 후 성능이 소폭 상승함.  
- 하이브리드 검색에서 **Sparse와 Dense**의 밸런스가 중요한데, 가중치 설정이 어려운 점이 있음.  
- 최신 로베르타 모델로 교체 및 MinMax Scale을 사용해 스코어 조정.

멘토링 세션에서 받은 주요 내용들을 기반으로, 문제 해결과 성능 개선에 중점을 두었습니다.

### Presentation

- - [UpStageAILAB-4th-Competition-IR-Presentation-IR4-20241025.pptx]([TBD](https://docs.google.com/presentation/d/18v89iJzpVrBSn4RgcgUDYV5_aZA37hrc/edit?usp=sharing&ouid=102302788798357252978&rtpof=true&sd=true))

### Reference

- 위 개인별 실험 일지 참조
