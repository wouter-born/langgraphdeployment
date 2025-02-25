from langchain_groq import ChatGroq
##################################
# LLM Setup
##################################
modelVers = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key="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",
    disable_streaming=True
)

modelSpec = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key="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",
    disable_streaming=True
)

# modelVers = ChatOpenAi(
#     temperature=0,
#     model_name="ChatGPT-4o")

# modelSpec = ChatOpenAi(
#     temperature=0,
#     model_name="ChatGPT-4o")