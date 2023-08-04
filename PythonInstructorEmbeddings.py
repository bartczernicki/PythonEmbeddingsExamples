import numpy as np
import json
import torch
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.device_count())
print("device = ", device)

# Load the Instructor model
model = INSTRUCTOR('hkunlp/instructor-large')


sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction,sentence]])
print(embeddings.shape)

# FAISS index setup
dimension = 768  # Instructor-XL output dimension
# index_ins = faiss.IndexFlatL2(dimension)

# Extract and vectorize data
db_filename = 'arxiv-metadata-10000.json'
num_lines = 10000
batch_size = 4

# Load all papers from JSON
with open(db_filename, 'r') as f:
    papers = [json.loads(line) for line in f]


# Extract the papers' titles and abstracts
texts = [f"{paper['title']}: {paper['abstract']}" for paper in papers]

# Preparation for encoding
instructions = ["Represent the science titles and abstracts: "] * len(texts)

# Prepare the inputs
inputs = [[instr, txt] for instr, txt in zip(instructions, texts)]

# Create vectors using Instructor
vectors = model.encode(
    sentences=inputs[:num_lines],
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=str(device)
)


# prepare texts with instructions
# text_instruction_pairs = [
#     {"instruction": "Represent the Science title:", "text": "3D ActionSLAM: wearable person tracking in multi-floor environments"},
#     {"instruction": "Represent the Medicine sentence for retrieving a duplicate sentence:", "text": "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear."}
# ]

# # postprocess
# texts_with_instructions = []
# for pair in text_instruction_pairs:
#     texts_with_instructions.append([pair["instruction"], pair["text"]])

# # calculate embeddings
# customized_embeddings = model.encode(texts_with_instructions)

# for pair, embedding in zip(text_instruction_pairs, customized_embeddings):
#     print("Instruction: ", pair["instruction"])
#     print("text: ", pair["text"])
#     print("Embedding: ", embedding)
#     print("")


# from sklearn.metrics.pairwise import cosine_similarity
# sentences_a = [['Represent the Science sentence: ','Parton energy loss in QCD matter'], 
#                ['Represent the Financial statement: ','The Federal Reserve on Wednesday raised its benchmark interest rate.']
#                ]
# sentences_b = [['Represent the Science sentence: ','The Chiral Phase Transition in Dissipative Dynamics'],
#                ['Represent the Financial statement: ','The funds rose less than 0.5 per cent on Friday']]
# embeddings_a = model.encode(sentences_a)
# embeddings_b = model.encode(sentences_b)
# similarities = cosine_similarity(embeddings_a,embeddings_b)    
# print(similarities)


query  = [['Represent the Wikipedia question for retrieving supporting documents: ','where is the food stored in a yam plant']]
corpus = [['Represent the Wikipedia document for retrieval: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
          ['Represent the Wikipedia document for retrieval: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loans—and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
          ['Represent the Wikipedia document for retrieval: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.']]
query_embeddings = model.encode(query)
corpus_embeddings = model.encode(corpus)
similarities = cosine_similarity(query_embeddings,corpus_embeddings)
print(similarities)
retrieved_doc_id = np.argmax(similarities)
print(retrieved_doc_id)

query  = [['Represent the question for retrieving supporting documents and paragraphs: ','What contributed to sub-prime lending?']]
corpus = [['Represent the paragraph for retrieval: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
          ['Represent the paragraph for retrieval: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loans—and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
          ['Represent the paragraph for retrieval: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.']]
query_embeddings = model.encode(query)
corpus_embeddings = model.encode(corpus)
similarities = cosine_similarity(query_embeddings,corpus_embeddings)
print(similarities)
retrieved_doc_id = np.argmax(similarities)
print(retrieved_doc_id)

