import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

# LangGraph + LLM imports (adapt paths to your environment)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import START, END, MessagesState, StateGraph

################################################################################
# LLM Configuration
################################################################################

# Replace with your own LLM integration
llm = ChatOpenAI(model="gpt-4o", temperature=0) 


################################################################################
# State Definitions
################################################################################

class LPLetterGraphState(TypedDict):
    """ Top-level state for generating an LP letter. """
    # Input data
    previous_letters: List[str]           # Past quarterly letters
    portfolio_updates: List[str]          # Raw updates from portfolio companies
    theme: str                            # High-level theme or focus for the letter
    
    # For style / duplication
    style_insights: str                   # Summary or key style points extracted
    
    # Proposed subset of updates
    proposed_updates: List[str]           # The LLM's proposed subset of updates
    user_selected_updates: List[str]      # The user-confirmed subset of updates
    
    # Macro context (for introduction)
    search_query: str                     # Query for macro environment
    macro_context: List[str]              # Documents or text from web search (macro outlook)
    
    # Final output
    draft_letter: str                     # Draft version of the letter
    final_letter: str                     # Finalized version

################################################################################
# Node 1: Extract Style
################################################################################

style_extraction_instructions = """
You are given a series of previous LP letters. 
Your job: 
1. Identify common style elements (tone, structure, phrasing) 
2. Identify any repeated paragraphs that we must NOT repeat verbatim. 
Output a concise but thorough summary, focusing on the style and repeated text to avoid.
"""

def extract_style(state: LPLetterGraphState):
    """Analyze previous letters for style guidelines and duplication warnings."""
    previous_letters = state["previous_letters"]
    # Join the letters for analysis
    combined_letters = "\n\n---\n\n".join(previous_letters)
    
    system_msg = SystemMessage(content=style_extraction_instructions)
    style_summary = llm.invoke(
        [system_msg, 
         HumanMessage(content=f"Here are previous letters:\n{combined_letters}")]
    )
    return {
        "style_insights": style_summary.content
    }

################################################################################
# Node 2: Propose Which Updates to Include
################################################################################

proposal_instructions = """
You are given the style insights (how we typically write) plus a list of 10 portfolio updates. 
We can only choose 5-6 updates to feature in the final letter. 
1. Review the style insights so you maintain the correct tone. 
2. Propose which updates are the most relevant or interesting to our LPs, and explain why. 
3. Keep your proposal concise.
"""

def propose_updates(state: LPLetterGraphState):
    """Have the LLM propose which updates are most relevant."""
    style_insights = state["style_insights"]
    portfolio_updates = state["portfolio_updates"]
    
    # Format the updates
    updates_str = "\n\n".join(
        [f"- {idx+1}. {update}" for idx, update in enumerate(portfolio_updates)]
    )

    system_msg = SystemMessage(content=proposal_instructions)
    proposal = llm.invoke([
        system_msg,
        HumanMessage(
            content=(
                f"STYLE INSIGHTS:\n{style_insights}\n\n"
                f"PORTFOLIO UPDATES:\n{updates_str}\n\n"
                "Please propose which 5-6 updates to include."
            )
        )
    ])
    return {"proposed_updates": [proposal.content]}

################################################################################
# Node 3: User Confirms or Modifies the Proposed Updates
################################################################################

def user_feedback_proposed_updates(state: LPLetterGraphState):
    """
    This node allows the user to confirm or modify the proposed updates.
    The typical flow: 
    1. LLM suggests updates. 
    2. We show them to the user. 
    3. The user picks which updates to keep or modifies them.
    """

    # This is a "placeholder" node to be interrupted in LangGraph Studio,
    # so the user can edit `user_selected_updates`.
    pass

################################################################################
# Node 4: Web Search (Macro Environment)
################################################################################

macro_instructions = """
You will be given a short query about the macro environment for VC or public markets. 
Generate a structured search query to gather relevant details for an LP letter introduction.
"""

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for macro environment")

def search_web_macro(state: LPLetterGraphState):
    """Perform a web search to gather relevant macro context."""
    # Step 1: LLM to refine or create a search query
    structured_llm = llm.with_structured_output(SearchQuery)
    
    user_query = state["search_query"]
    # We pass in instructions + user query
    search_query_obj = structured_llm.invoke([
        SystemMessage(content=macro_instructions),
        HumanMessage(content=f"User wants macro outlook: {user_query}")
    ])
    
    # Step 2: Perform the actual search
    tavily_search = TavilySearchResults(max_results=3)
    search_results = tavily_search.invoke(search_query_obj.search_query)

    # Format for readability
    formatted_results = []
    for doc in search_results:
        formatted_results.append(
            f"<Document href='{doc['url']}'>\n{doc['content']}\n</Document>"
        )

    return {"macro_context": formatted_results}

################################################################################
# Node 5: Draft LP Letter
################################################################################

draft_instructions = """
You are drafting a quarterly LP letter for Born Capital. 
Use the following guidelines:

1. Maintain the style extracted from previous letters:
{style_insights}

2. The letter should have:
   - A short personal greeting
   - A brief macro / market overview (from macro context)
   - Updates on selected portfolio companies
   - A closing remark

3. Do NOT reuse paragraphs verbatim from prior letters. 
4. Use the selected updates (only). 
5. Keep a professional, concise, and forward-looking tone.

Write the letter now.
"""

def draft_letter(state: LPLetterGraphState):
    style_insights = state["style_insights"]
    macro_context = "\n\n".join(state["macro_context"])
    chosen_updates = state["user_selected_updates"]
    theme = state["theme"]
    
    # Build a bullet list of chosen updates
    chosen_updates_str = "\n".join(
        [f"- {update}" for update in chosen_updates]
    )

    system_msg = SystemMessage(
        content=draft_instructions.format(style_insights=style_insights)
    )
    letter_draft = llm.invoke([
        system_msg,
        HumanMessage(
            content=(
                f"The theme of this letter: {theme}\n\n"
                f"MACRO CONTEXT:\n{macro_context}\n\n"
                f"SELECTED PORTFOLIO UPDATES:\n{chosen_updates_str}"
            )
        )
    ])
    return {"draft_letter": letter_draft.content}

################################################################################
# Node 6: User Edits Final Draft
################################################################################

def user_edits_draft(state: LPLetterGraphState):
    """
    This node is a placeholder for user to provide final edits to the letter.
    In LangGraph Studio, you can interrupt and manually edit `draft_letter` -> `final_letter`.
    """
    pass

################################################################################
# Node 7: Finalize Letter
################################################################################

def finalize_letter(state: LPLetterGraphState):
    """
    Copy the user-edited `draft_letter` into `final_letter` 
    or do minimal post-processing. 
    """
    return {"final_letter": state["draft_letter"]}

################################################################################
# Graph Construction
################################################################################

lp_builder = StateGraph(LPLetterGraphState)

# 1) Style extraction
lp_builder.add_node("extract_style", extract_style)
# 2) LLM proposes which updates to include
lp_builder.add_node("propose_updates", propose_updates)
# 3) User feedback to confirm or modify proposed updates
lp_builder.add_node("user_feedback_proposed_updates", user_feedback_proposed_updates)
# 4) Web search for macro environment
lp_builder.add_node("search_web_macro", search_web_macro)
# 5) Draft the letter
lp_builder.add_node("draft_letter_node", draft_letter)
# 6) User edits final draft
lp_builder.add_node("user_edits_draft", user_edits_draft)
# 7) Finalize the letter
lp_builder.add_node("finalize_letter", finalize_letter)

# Edges
lp_builder.add_edge(START, "extract_style")
lp_builder.add_edge("extract_style", "propose_updates")
lp_builder.add_edge("propose_updates", "user_feedback_proposed_updates")
lp_builder.add_edge("user_feedback_proposed_updates", "search_web_macro")
lp_builder.add_edge("search_web_macro", "draft_letter_node")
lp_builder.add_edge("draft_letter_node", "user_edits_draft")
lp_builder.add_edge("user_edits_draft", "finalize_letter")
lp_builder.add_edge("finalize_letter", END)

# Compile the graph
graph = lp_builder.compile(
    # We can choose to interrupt before user feedback steps
    interrupt_before=["user_feedback_proposed_updates", "user_edits_draft"]
)

# The resulting lp_letter_graph can now be run in LangGraph Studio. 
# You can provide initial state, step through nodes interactively, 
# and produce your final LP letter.
