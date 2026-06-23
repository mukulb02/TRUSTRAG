import pandas as pd

from agents.retrieval_agent import RetrievalAgent
from agents.evidence_agent import EvidenceAgent

df = pd.read_csv(
    "train_split.csv"
)

corpus = df["text"].tolist()

retriever = RetrievalAgent(
    corpus
)

evidence_agent = EvidenceAgent()

query = df.iloc[0]["query"]

retrieval = retriever.retrieve(
    query
)

evidence = evidence_agent.verify(
    query,
    retrieval["docs"]
)

print(evidence)