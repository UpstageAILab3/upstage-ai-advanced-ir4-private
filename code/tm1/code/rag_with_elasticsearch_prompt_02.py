import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# Elasticsearch client 생성
es_username = "elastic"
es_password = "6dKfSe4v-hJtAQIXhdGr"
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

# 동의어 사전 파일 생성
synonym_file_path = "./science_technology_synonyms.txt"
def create_synonym_file():
    synonyms = [
        "금성, 비너스",
        "밝다, 빛나다",
        "행성, 천체",
        "과학, 사이언스, 자연과학",
        "기술, 테크놀로지, 공학",
        "컴퓨터, 전산, 연산",
        "데이터, 정보, 자료",
        "인공지능, AI, 기계학습",
        "로봇, 자동화, 기계",
        "인터넷, 네트워크, 웹",
        "우주, 천체, 우주 공간",
        "생물학, 바이오, 생명과학",
        "화학, 물질, 화합물",
        "물리학, 물리, 자연법칙",
        "전기, 전자, 에너지",
        "수학, 계산, 통계",
        "환경, 에코, 생태계",
        "에너지, 동력, 자원",
        "공학, 엔지니어링, 기술 설계",
        "항공, 비행, 항공우주",
        "유전자, DNA, 유전학",
        "의학, 치료, 건강",
        "기상학, 날씨, 기후"
    ]
    with open(synonym_file_path, 'w') as f:
        for synonym in synonyms:
            f.write(f"{synonym}\n")

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
    # 동의어 사전 파일이 없으면 생성
    if not os.path.exists(synonym_file_path):
        create_synonym_file()
    
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

# 하이브리드 검색 함수 (역색인 + 벡터 검색)
def hybrid_retrieve(query_str, size):
    # 역색인 검색
    sparse_results = sparse_retrieve(query_str, size)
    
    # 벡터 유사도 검색
    dense_results = dense_retrieve(query_str, size)

    # 하이브리드 결과 결합
    results = {}
    
    # 역색인 검색 결과 반영
    for hit in sparse_results['hits']['hits']:
        docid = hit['_source']['docid']
        results[docid] = {"score": hit['_score'], "source": hit['_source']['content'], "method": "sparse"}
    
    # 벡터 검색 결과 반영 (점수를 평균화하거나 특정 방식으로 결합)
    for hit in dense_results['hits']['hits']:
        docid = hit['_source']['docid']
        if docid in results:
            results[docid]['score'] = (results[docid]['score'] + hit['_score']) / 2  # 평균 점수로 결합
        else:
            results[docid] = {"score": hit['_score'], "source": hit['_source']['content'], "method": "dense"}

    # 상위 결과를 점수에 따라 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return sorted_results[:size]

# Elasticsearch 인덱스 생성 (동의어 필터 적용)
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter", "synonym_filter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            },
            "synonym_filter": {
                "type": "synonym",
                "synonyms_path": "synonym.txt"
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
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

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 하이브리드 검색 예제
search_results = hybrid_retrieve(test_query, 3)
for docid, result in search_results:
    print(f"DocID: {docid}, Score: {result['score']}, Method: {result['method']}, Content: {result['source']}")

# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = "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"


client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
# llm_model = "gpt-3.5-turbo-1106"
# 사용할 모델을 설정(여기서는 gpt-4 모델 사용)
llm_model = "gpt-4"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: Knowledge Expert

## Instructions
- Generate concise answers using the user's previous message information and the given reference information.
- If the provided search result information is insufficient to answer, respond by stating that there is not enough information to answer.
- Respond in Korean.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: Knowledge Expert

## Instructions
- When the user asks a question related to scientific knowledge during the conversation, you should be able to call the search API.
- For other conversation messages not related to scientific knowledge, generate an appropriate response.
- Respond in Korean.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
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
        standalone_query = function_args.get("standalone_query")

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        search_result = sparse_retrieve(standalone_query, 3)

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
def test_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["test_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
test_rag("../data/test.jsonl", "ir4-parksurk-exp02-test.csv")

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
eval_rag("../data/eval.jsonl", "ir4-parksurk-exp02.csv")
