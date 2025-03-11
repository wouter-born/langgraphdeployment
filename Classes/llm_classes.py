from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAi
##################################
# LLM Setup
##################################
# modelVers = ChatGroq(
#     temperature=0,
#     model_name="llama-3.3-70b-versatile",
#     api_key="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",
#     disable_streaming=True
# )

modelDeepSpec = ChatGroq(
    temperature=1,
    model_name="deepseek-r1-distill-llama-70b-specdec",
    api_key="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",
    disable_streaming=True
)


# modelSpec = ChatGroq(
#     temperature=0,
#     model_name="llama-3.3-70b-versatile",
#     api_key="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",
#     disable_streaming=True
#)

modelVers = ChatOpenAi(
    temperature=0,
    model_name="gpt-4o-mini")

modelSpec = ChatOpenAi(
    temperature=0,
    model_name="gpt-4o-mini")