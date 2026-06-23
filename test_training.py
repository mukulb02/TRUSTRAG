# import pandas as pd

# df = pd.read_csv("trust_training.csv")

# print(df.shape)

# print(df["label"].value_counts())

# print(
#     df["label"].value_counts(
#         normalize=True
#     )
# )


import pandas as pd
from agents.retrieval_agent import RetrievalAgent

df = pd.read_csv("train_split.csv")

corpus = df["text"].tolist()

agent = RetrievalAgent(corpus)

query = df.iloc[0]["query"]

result = agent.retrieve(query)

print(result.keys())
print()

print("Margin:", result["margin"])

for i,d in enumerate(result["docs"][:3]):
    print("\nDOC",i)
    print(d["text"][:500])