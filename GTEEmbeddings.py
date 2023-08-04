from sentence_transformers import SentenceTransformer, util
import torch

print("Loading Models...")
modelGTESmall = SentenceTransformer('thenlper/gte-small')
modelGTELarge = SentenceTransformer('thenlper/gte-large')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]

print("Processing Embeddings...")
corpus_embeddings_small = modelGTESmall.encode(corpus, convert_to_tensor=False)
corpus_embeddings_large = modelGTELarge.encode(corpus, convert_to_tensor=False)

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 
           'A cheetah chases prey on across a field.']

print("Semantic Search over Embeddings...")
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    # Encode the query to embeddings     
    query_embedding_small = modelGTESmall.encode(query)
    query_embedding_large = modelGTELarge.encode(query)    

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores_small = util.cos_sim(query_embedding_small, corpus_embeddings_small)[0]
    top_results_small = torch.topk(cos_scores_small, k=top_k)

    cos_scores_large = util.cos_sim(query_embedding_large, corpus_embeddings_large)[0]
    top_results_large = torch.topk(cos_scores_large, k=top_k)
    
    print("\n\n============SMALL=========\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    for score, idx in zip(top_results_small[0], top_results_small[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))    
    print("\n\n============LARGE=========\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    for score, idx in zip(top_results_large[0], top_results_large[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))


