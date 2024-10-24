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