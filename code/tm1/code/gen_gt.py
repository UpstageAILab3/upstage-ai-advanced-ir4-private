import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Prepare gt.csv data
gt_data = []

for test in tests:
    test_id = test['test_id']
    query = test['msg'][0]['content']
    
    # Get top 3 similar documents
    top3_docids = get_top3_similar_documents(query, document_contents)
    
    # Store the result
    gt_data.append({'test_id': test_id, 'doc_ids': ','.join(top3_docids)})

# Convert to DataFrame and save as CSV
gt_df = pd.DataFrame(gt_data)
gt_df.to_csv(output_file_path, index=False)

print(f"gt.csv has been created with top 3 similar documents for each query.")
