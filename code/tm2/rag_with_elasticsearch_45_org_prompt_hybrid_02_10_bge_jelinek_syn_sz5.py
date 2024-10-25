import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#2024.10.17
# model = SentenceTransformer("jhgan/ko-sroberta-multitask")
#2024.10.23
model = SentenceTransformer("dragonkue/bge-m3-ko")

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)

# 2024.10.17
# 역색인 + Vector 유사도 혼합
def hybrid_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    body = {
        "query": {
            "match": {
                "content": {
                    "query": query_str,
                    # "boost": 0.0005
                    "boost": 0.02
                }
            }
        },
        "knn": {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": size,
            "num_candidates": 100,
            "boost": 1.0
        },
        "size": size
    }
    
    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", body=body)


es_username = "elastic"
es_password = "vB3rUCgR03bYsH0Ab0-="

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter", "synonyms_filter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            },
            #2024.10.23
            "synonyms_filter": {
                "type": "synonym",
                "lenient": True,
                "synonyms": [
                    "복숭아, 복숭아나무",
                    "밀물, 만조",
                    "썰물, 간조",
                    "관계, 서로 연결, 긴밀",
                    "순기능, 도움",
                    "리보오솜, 리보솜",
                    "네트웍, 네트워크",
                    "역할, 기능, 기여, 담당",
                    "전력, 전류",
                    "이루어져, 구성, 형성",
                    "땅, 토양",
                    "가장 많이, 대부분",
                    "단점, 부정적인",
                    "원자력 발전, 에너지",
                    "형태, 유형",
                    "원자, 가장 작은 입자",
                    "내부 구조, 구성"
                    "자연, 생태계",
                    "뭉쳐, 모여",
                    "날씨, 기후",
                    "측정, 추정",
                    "비만도, 체중 상태",
                    "관계, 영향",
                    "뭐야, 의미",
                    "해안, 바다",
                    "전구, 램프",
                    "번식, 생식",
                    "작은 기체 하나, 기체의 분자",
                    "다음 세대, 후손",
                    "무게, 질량",
                    "잠, 수면",
                    "약, 약물",
                    "자석, 자성체",
                    "동물, 생물체",
                    "누가, 비해",
                    "결혼 전, 혼전",
                    "숨, 호흡",
                ]
            }
        }
    },
    # 2024.10.23
    "index": {
        "similarity": {
            "lm_jelinek_mercer": { 
                "type": "LMJelinekMercer", 
                "lambda": 0.7
            } 
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori", "similarity": "lm_jelinek_mercer"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 1024, #768
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("../data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

# test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# # 역색인을 사용하는 검색 예제
# search_result_retrieve = sparse_retrieve(test_query, 3)

# # 결과 출력 테스트
# for rst in search_result_retrieve['hits']['hits']:
#     print('score:', rst['_score'], 'source:', rst['_source']["content"])

# # Vector 유사도 사용한 검색 예제
# search_result_retrieve = dense_retrieve(test_query, 3)

# # 결과 출력 테스트
# for rst in search_result_retrieve['hits']['hits']:
#     print('score:', rst['_score'], 'source:', rst['_source']["content"])


# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = "sk-JS_ZVqHYqxrFdIABig_Bnh3z-fL0q_ulYjltMwt8PST3BlbkFJeOxUpgYs6y6egDu9X0_9WmMhwoEpicu67XMK3KZkUA"

# 조직ID??
client = OpenAI(
  organization='org-P2RnpUC5a0tpU9HZXmFMHhAa'
)

# client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
llm_model = "gpt-3.5-turbo-1106"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 지식에 대해 질문하면 반드시 search 함수를 호출해야 한다.
- 나머지 일상대화 메시지에는 반드시 함수를 호출하지 않고 적절한 대답을 생성한다.
"""

# Function calling에 사용할 함수 정의
####### prompt Test #############
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User's question in Korean. Full message if the user message is single-turn."
                    }
                },
                "required": ["question"],
                "type": "object"
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        # standalone_query = function_args.get("standalone_query")
        standalone_query = function_args.get("question")

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        # search_result = sparse_retrieve(standalone_query, 3)
        
        # 2024.10.17
        # Vector 유사도 사용 Test
        # search_result = dense_retrieve(standalone_query, 3)

        # 2024.10.17
        # hybrid_retrieve
        search_result = hybrid_retrieve(standalone_query, 5)
        
        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("../data/eval.jsonl", "submission_45.csv")
