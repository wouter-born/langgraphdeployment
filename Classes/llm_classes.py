from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

##################################
# LLM Setup
##################################
modelVers = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key="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",
    disable_streaming=True
)

modelDeepSpec = ChatGroq(
    temperature=1,
    model_name="deepseek-r1-distill-llama-70b-specdec",
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




llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    temperature=0,
    timeout=None,
    max_retries=2,
    disable_streaming=True
)


# Enabling these two lines will change models.
# modelSpec = llm
# modelVers = llm