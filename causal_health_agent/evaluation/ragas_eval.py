"""Stage 4/5 evaluation with RAGAS (paper Sec. IV-D, Table VI): Faithfulness, Answer Relevance, Context Precision.
Optional dependency:  pip install ragas langchain-openai   (written against ragas>=0.2; UNTESTED here)
Rows: {"query": str, "contexts": [str], "answer": str}  (answer = the Stage-5 evidence-grounded summary)."""
import os

from .. import config as C


def ragas_scores(rows, model=C.QUERY_MODEL):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
    llm = LangchainLLMWrapper(ChatOpenAI(model=model, base_url=C.OPENROUTER_BASE_URL,
                                         api_key=os.environ["OPENROUTER_API_KEY"]))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small",
                                                      api_key=os.environ["OPENAI_API_KEY"]))
    ds = EvaluationDataset.from_list([{"user_input": r["query"], "retrieved_contexts": r["contexts"],
                                       "response": r["answer"]} for r in rows])
    res = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
                   llm=llm, embeddings=emb)
    return res.to_pandas().mean(numeric_only=True).round(3).to_dict()
