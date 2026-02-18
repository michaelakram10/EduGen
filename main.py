from rag_pipeline import RAGPipeline

rag = RAGPipeline("pdfs")

while True:
    q = input("\nAsk: ")
    print(rag.query(q))
