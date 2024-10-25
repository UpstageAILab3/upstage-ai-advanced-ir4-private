import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#2024.10.23
model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")

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
                    "boost": 0.00205
                    # "boost": 0.004
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
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
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
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search",
#             "description": "search relevant documents",
#             "parameters": {
#                 "properties": {
#                     "standalone_query": {
#                         "type": "string",
#                         "description": "Final query suitable for use in search from the user messages history."
#                     }
#                 },
#                 "required": ["standalone_query"],
#                 "type": "object"
#             }
#         }
#     },
# ]

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

# Max score 비교하자
    sparse_mx_score = 0
    dense_mx_score = 0
    
    sparse_mn_score = 1000
    dense_mn_score = 1000

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        # standalone_query = function_args.get("standalone_query")
        standalone_query = function_args.get("question")

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        sparse_result = sparse_retrieve(standalone_query, 3)
        
        # Max score 구하기
        
        for i,rst in enumerate(sparse_result['hits']['hits']):
            if sparse_mx_score < rst["_score"]:
                sparse_mx_score = rst["_score"]
            if sparse_mn_score > rst["_score"]:
                sparse_mn_score = rst["_score"]
        
        # 2024.10.17
        # Vector 유사도 사용 Test
        dense_result = dense_retrieve(standalone_query, 3)
        
        # Max score 구하기
        
        for i,rst in enumerate(dense_result['hits']['hits']):
            if dense_mx_score < rst["_score"]:
                dense_mx_score = rst["_score"]
            if dense_mn_score > rst["_score"]:
                dense_mn_score = rst["_score"]

        # 2024.10.17
        # hybrid_retrieve
        search_result = hybrid_retrieve(standalone_query, 3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        ## OpenAI API 호출 건너뛰기 - 비용문제 - 실제 결과값 만들때 풀것.
        # content = json.dumps(retrieved_context)
        # messages.append({"role": "assistant", "content": content})
        # msg = [{"role": "system", "content": persona_qa}] + messages
        # try:
        #     qaresult = client.chat.completions.create(
        #             model=llm_model,
        #             messages=msg,
        #             temperature=0,
        #             seed=1,
        #             timeout=30
        #         )
        # except Exception as e:
        #     traceback.print_exc()
        #     return response
        # response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content
        


    return response, sparse_mx_score, sparse_mn_score, dense_mx_score, dense_mn_score


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        t_sparse_mx_score = 0
        t_sparse_mn_score = 1000
        t_dense_mx_score = 0 
        t_dense_mn_score = 1000

        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response, sparse_mx_score, sparse_mn_score, dense_mx_score, dense_mn_score = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1
            
            if t_sparse_mx_score < sparse_mx_score:
                t_sparse_mx_score = sparse_mx_score
                
            if t_sparse_mn_score > sparse_mn_score:
                t_sparse_mn_score = sparse_mn_score
                
            if t_dense_mx_score < dense_mx_score:
                t_dense_mx_score = dense_mx_score
                
            if t_dense_mn_score > dense_mn_score:
                t_dense_mn_score = dense_mn_score
            
        print("\nScore : ", t_sparse_mx_score, t_sparse_mn_score, t_dense_mx_score, t_dense_mn_score)

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
# eval_rag("../data/eval.jsonl", "submission_7.csv")
eval_rag("../data/test2.jsonl", "test_42.csv")







###############################################################
# 결과를 저장할 리스트 초기화
pred = []

with open("test_42.csv", "r") as f:
    for line in f:
        data = json.loads(line)
        pred.append(data)

# 결과 출력
# print(pred)

##################################################################

def calc_map(gt, pred):    
    sum_average_precision = 0    
    for j in pred:        
        if gt[j["eval_id"]]:            
            hit_count = 0            
            sum_precision = 0            
            for i,docid in enumerate(j["topk"][:3]):                
                if docid in gt[j["eval_id"]]:                    
                    hit_count += 1                    
                    sum_precision += hit_count/(i+1)            
            average_precision = sum_precision / hit_count if hit_count > 0 else 0        
        else:            
            average_precision = 0 if j["topk"] else 1        
        sum_average_precision += average_precision    
    return sum_average_precision/len(pred)
#################################################################
gt = {
	0:["42508ee0-c543-4338-878e-d98c6babee66"],
	1:["4a437e7f-16c1-4c62-96b9-f173d44f4339"],
	2:["d3c68be5-9cb1-4d6e-ba18-5f81cf89affb"],
	3:[],
	444:["910107a6-2a42-41a2-b337-fbf22d6440fe"],
	5:["74f22819-1a8e-4646-8a9d-13323de8cdb8"],
	6:["80feb7f2-1b24-4a9d-87a3-4976e9304e74"],
	7:[],
	8:[],
	9:["93c91ff9-641a-4117-bf83-6327d3e382eb"],
	10:["418e1e6c-c2de-4ffb-aded-0992fe833776"],
	11:["69de4a81-3716-4195-973c-39eee3771186"],
	12:["11cca31a-3997-43fd-a04b-01a2e318d7a2"],
	13:["6b6971ff-885f-48cf-adca-46bbd01041e6"],
	14:["77d420f0-2eb3-4898-8bbc-12348c7f9003"],
    78: ["c63b9e3a-716f-423a-9c9b-0bcaa1b9f35d"],
    213: ["79c93deb-fe60-4c81-8d51-cb7400a0a156"],
    107: ["25de4ffd-cee4-4f27-907e-fd6b802c6ede"],
    81: ["bd91bda8-351e-4683-bb1a-8254f93e2376"],
    280: ["38686456-b993-4cbb-af0d-1c53df2f3e12"],
    10: ["99a07643-8479-4d34-9de8-68627854f458"],
    100: ["d9ce8432-f72e-4253-9735-98318a6f9f7f"],
    279: ["0f0dd1ae-a36c-4c97-9785-4698400c67b1", "de1ab247-9d48-48f7-8499-31606f53c108"],
    42: ["4b49f3a2-32c9-4b2e-89c4-4719f98e7a74"],
    308: ["72c780ec-57bb-4fe1-976a-d7ee3d3dbb52"],
    205: ["ae28101b-a42e-45b7-b24b-4ea0f1fb2d50"],
    289: ["1f442344-084b-44f8-838b-332be289083c", "421aac6b-49ce-4697-a68f-850152f323d7"],
    268: ["7c14c33c-f15c-41f6-ab5c-4d37c2eb3513"],
    18: ["63846d07-8443-4bf8-8cd9-bc6cc7826555"],
    9: ["bfbba89d-fdaa-400d-8848-ca1ff8d51cd7"],
    101: ["144f5e5e-8069-425f-80b3-6388195ba4ee"],
    236: ["2077ea5b-53ac-4242-bbfc-20005ad63db8"],
    59: ["c1edd0df-b9f4-4dc2-90f4-38708cddddba", "4344f76e-1747-4bc9-8d02-c26db29151f4"],
    25: ["35c5dcc7-4720-4318-901e-770105ae63fd"],
    5: ["84f3f0e3-7ff2-4090-9961-aa7bbe8ca412", "59d5d7bb-6700-40ad-8884-ff43b1a9a1a0", "abf99ff1-d6bf-4020-b752-da7cb8611915"],
    104: ["73089763-06d2-4395-b235-aa3e6a399531"],
    276: [],
    14: ["2077ea5b-53ac-4242-bbfc-20005ad63db8"],
    270: ["a729b4f2-c734-4c60-9205-1518ba762593", "191c4b9f-6feb-49dd-90ad-9f2eebb6113e"],
    238: ["f48600d6-e492-43eb-b564-1860aa81da5f"],
    269: ["05b5a4f4-b115-4b76-9fe1-b80c4498289b"],
    43: ["cefe7caf-6cd1-422a-b41e-e82b543556e9", "8a78364e-63bf-4915-b718-fdc461bc62c9"],
    65: ["0bda5010-9ac6-447b-b484-60e380f4921d", "f5f54058-8c3c-4f6b-9549-db99b17685ed", "d9492876-df4d-4570-a58d-5a0438315fc9"],
    97: ["1655c90b-29c7-47ef-a092-01f2550db3aa", "85d28a10-9380-4afe-afef-b34449ef86bf"],
    206: ["70d104b2-8d74-4799-a09d-5a4c8dd577c0"],
    21: ["7150c749-dff2-4bd5-90ff-ff1e1cda468b"],
    221: ["8ae1234f-2a28-4069-a017-e99de5d67cc6"],
    71: ["5043c033-841c-46dd-94a3-1b5bef034c62"],
    254: ["af966ff7-109a-4c28-a644-393f5333ce69"],
    226: ["e21aceaf-be57-426c-b999-7ee8a309db36"],
    241: ["468d098e-2322-4950-ac11-9756f3112944"],
    261: [],
    45: ["41ca41ac-66e3-4a6b-a604-87bf8b3a8d4d"],
    19: ["d8ad7175-469b-45b5-8eb9-69504cd04f0f"],
    210: ["4764014a-4240-4c65-aa92-20eb1369a2f7"],
    231: ["5392d86a-bc7a-46c3-8272-94d982a65eed"],
    233: ["029064ed-d9f6-42cd-9b86-88cc9f611414"],
    263: ["1a277fb7-4cd7-409b-9f28-d83cef78ca10", "c8fd4323-9af9-4a0d-ab53-e563c71f9795"],
    201: ["469e37c0-a241-4675-8b1a-aa31d11a438c"],
    293: ["36788458-5fb5-4bcd-be02-3a47e5c8c19d"],
    208: ["b22a35e9-244e-44d6-b7bf-97f3ff664866"],
    282: ["2fb58e26-5ea0-4b50-b80d-4b03640042b4"],
    62: ["e7fef7f2-2549-499b-8b36-e4628119d352"],
    55: ["21383ddc-b6bb-4cf7-8815-139a3c4d9fae"],
    257: ["c8bd9b15-8ce0-4307-9f49-0f205217178f"],
    58: ["25bf6c36-116f-42c9-9d1c-c179e6292a34"],
    283: [],
    32: [],
    94: [],
    15: ["26cb5bba-0b80-41d4-9e42-aada06c879ca"],
    4: ["aa674ad5-ae70-4223-8685-e717a27dc1b3"],
    300: ["41ce1303-0091-4414-b26d-18f66101a99f"],
    243: ["5ff8f00a-a4e6-43fd-8616-3104a4c4d637"],
    34: ["55726582-8401-4a6f-889b-e9bd3953be7c", "cc6c9dc5-4d30-4653-bee7-9f3ba90fbf48", "6335780f-292d-49da-a79c-3eee1d51a903"],
    246: ["cc56ca24-fde0-458d-95f7-d3d31b79acb5", "ec539caa-4b62-4b5f-8428-489809f80611"],
    212: ["b303e4ec-87fb-40f9-8704-9037bad5af8a"],
    214: ["2213e6c8-ebb4-4cb5-b4c4-4a1773c63bcc"],
    259: ["dbfa9bbf-e2da-4d01-8aea-d6f25d43ffcf"],
    267: ["c3829f80-57bf-4db6-9e06-dd81b8bb6148"],
    90: [],
    66: ["ec87a926-171d-4f62-9acc-1b870c010a16"],
    20: ["ad5d883f-4352-4d25-ba7b-cbb605c73662"],
    24: ["bf4977ff-8fa5-4e82-b957-b6955c5bcbf0", "77a60236-e5b8-4c86-a422-1f3fa9726492"],
    281: ["de5d9a1f-67f6-45ea-a4a4-0540f5b7583e"],
    264: ["31e7fe84-bd3a-4b49-8adc-cc2e1f7a5b42"],
    79: ["c59f4bbd-71f6-4073-807f-09e8f0d3efcc"],
    304: ["ff7e8f8e-8d03-419f-bd98-93e3651fd01a"],
    292: ["8eac310f-a32f-462b-8ca8-da7c8fcd41e7", "655f64b4-74e8-4f5e-9b49-13a976ad3ee4"],
    215: ["ae30b754-a275-43dc-a2c9-95ab33a7c557"],
    225: ["6fec9716-f49f-40e9-913e-db3f4df46379"],
    108: ["d23082b4-ab12-4bcb-8658-d54ba791f263"],
    98: ["5db9db05-7e10-41d4-8f1f-81259fdc8ea7"],
    38: ["d525c38e-a296-4dec-8a60-d1fb98a355e3"],
    287: ["79c61856-f978-48c5-a7a8-0863bc661106"],
    295: ["6a8852ee-b847-4bc8-9b22-a9b47b56181e"],
    255: ["62cb0474-809d-4554-82b8-861ba23d8cb4"],
    76: ["6c9bca81-11e0-41a4-b894-3bcd66dc2bf6"],
    248: ["b39fe7fd-1fa8-46ef-ba31-f4464727d56e"],
    85: ["15f317a8-1d4d-41c4-93ec-bc9be43c3984"],
    290: ["93d1aa65-a113-4dd2-8959-e8204bfea616"],
    73: ["0d316cb2-c466-4b2b-b28b-0a726d513ef2"],
    309: ["395d6c0b-a199-451e-81ed-b49bfd853927"],
    239: ["2ead4363-f521-4377-b71c-f380fd9f5094", "deb0df64-4649-4785-a87a-2ad90a819c25"],
    284: ["552fc915-5ee7-4d6f-a45f-86e8e6a5d02c", "6b11c698-eedd-4731-ae6f-c91a327c4725"],
    220: [],
    96: ["810e57ae-aab1-470d-b078-435beb1b5ce8"],
    217: ["0a29b75f-af58-4b6a-a124-0d71ffbcbc89"],
    6: ["1a707124-fe53-45aa-baaa-e3e8d6697742"],
    237: ["59d9a47e-3f46-4caa-b4e9-9d3056b3e453"],
    245: [],
    35: ["02759425-149b-467e-af41-11f924577549", "4392ecba-ea79-496b-8931-e1a79caad179", "c6f86dc0-0ecc-4232-87ad-d12dea2e4c5b"],
    242: ["209121c3-69df-414a-a1b9-c74a1b14384f"],
    296: ["2570699b-b8c3-409e-b797-a1ee57c1fd86"],
    307: ["5a8089d2-36d8-4734-a16c-85db8226bfd3"],
    229: [],
    30: ["f532c956-a335-4688-9078-b1a13a0cfe58", "cc9425b7-beaa-4bd9-9ed4-96cd82bda481"],
    16: ["21737d6c-8ec9-40fd-80ed-50548228f0e9"],
    247: [],
    285: ["b38604f4-94bd-4482-97df-7a768331558a"],
    77: ["5b8fa65f-9a69-4990-8618-50aa1911e3ec"], 
    250: ["1f70c6f1-f1a7-44be-90d6-d397bace344b", "a06ea0ff-67d9-4967-bbe2-0d9022551740"],
    68: ["59ce17a4-82be-4e0c-adbd-3ef4fd4e8a33"],
    86: ["d2762c95-6397-44b5-b20e-81d4b333dd69"],
    232: ["af4a89a5-4fe3-4655-88d7-fd2fcd118441"],
    109: ["00774f87-0573-4dd8-b4dc-c43ce406a51f"],
    1: ["4a8549c8-8e81-4804-9df9-852952dd5747"],
    28: ["03390467-c6dc-4242-a039-c7e9a8bb242f"],
    203: ["f81c0a09-4082-4b6a-8546-12371ff89fba"],
    91: ["ec326ad8-286b-4f58-9a03-31b94e969f33"],
    105: ["b7a0428e-c402-43cb-8e89-3d5f9a7f644b"],
    67: [],
    13: ["30ee5c53-1559-42f5-94c3-2ffb0b9682aa"],
    57: []
}

map_score = calc_map(gt, pred)
print("\ntest Mapping score", map_score)

# Score :  45.68352 7.731579 0.025545895 0.006014769

# test Mapping score 0.7044444444444444