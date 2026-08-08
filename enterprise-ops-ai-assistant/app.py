import streamlit as st
import os
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Operations AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# INITIALIZE BACKEND
# ============================================================

@st.cache_resource
def initialize_backend():

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-oss-20b:free",
        temperature=0
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Load company documents
    # --------------------------------------------------------

    documents_path = "documents"

    all_documents = []

    pdf_files = [
        "leave_policy.pdf",
        "attendance_policy.pdf",
        "wfh_policy.pdf",
        "employee_handbook.pdf"
    ]

    for filename in pdf_files:

        loader = PyPDFLoader(
            os.path.join(documents_path, filename)
        )

        all_documents.extend(loader.load())

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_documents)

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return llm, retriever


# ============================================================
# BACKEND INITIALIZATION
# ============================================================

try:

    llm, retriever = initialize_backend()

    llm_status = True
    kb_status = True

except Exception as e:

    st.error(f"Backend initialization failed: {e}")
    st.stop()


# ============================================================
# HR AGENT
# ============================================================

hr_prompt = ChatPromptTemplate.from_template(
    """
You are the HR Assistant for NovaTech Solutions Pvt. Ltd.

Your responsibility is to answer employee questions related to:

- Leave
- Attendance
- Work From Home
- Employee policies
- Employee handbook

Use the provided company knowledge base as your primary source.

IMPORTANT RULES:

1. Answer using the retrieved company documents.
2. Do not invent company policies.
3. If the information is not available in the retrieved documents,
   clearly state that the information is not available in the company
   knowledge base.
4. Keep answers clear, professional, and concise.

Retrieved Company Information:

{context}

Employee Question:

{question}
"""
)

hr_chain = hr_prompt | llm


def hr_agent(question):

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    response = hr_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    sources = []

    for document in documents:

        source = document.metadata.get("source")

        if source and source not in sources:
            sources.append(
                os.path.basename(source)
            )

    return {
        "answer": response.content,
        "sources": sources
    }


# ============================================================
# RESEARCH AGENT
# ============================================================

research_prompt = ChatPromptTemplate.from_template(
    """
You are the Research Agent for NovaTech Solutions Pvt. Ltd.

Your responsibility is to research topics using reliable external
information.

Analyze the provided research information and produce a clear,
factual summary.

Rules:

1. Base the answer on the provided research information.
2. Do not invent facts.
3. Prefer recent information when appropriate.
4. Keep the final answer concise but useful.
5. Mention the sources used when possible.

Research Question:

{question}

Research Information:

{research_results}
"""
)

research_chain = research_prompt | llm


def wikipedia_search(question):

    """
    Simple Wikipedia lookup using Wikipedia's REST API.
    """

    search_url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent":
        "NovaTech-Research-Agent/1.0 academic-project"
    }

    params = {
        "action": "query",
        "list": "search",
        "srsearch": question,
        "format": "json",
        "utf8": 1,
        "srlimit": 3
    }

    response = requests.get(
        search_url,
        params=params,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for item in data["query"]["search"]:

        results.append(
            {
                "title": item["title"],
                "snippet": item["snippet"],
                "url":
                    "https://en.wikipedia.org/wiki/"
                    + item["title"].replace(" ", "_")
            }
        )

    return results


def research_agent(question):

    try:

        wiki_results = wikipedia_search(question)

        if not wiki_results:

            return {
                "answer":
                    "No external research results were found.",
                "sources": []
            }

        research_context = ""

        for result in wiki_results:

            research_context += (
                f"Title: {result['title']}\n"
                f"Information: {result['snippet']}\n"
                f"URL: {result['url']}\n\n"
            )

        response = research_chain.invoke(
            {
                "question": question,
                "research_results": research_context
            }
        )

        sources = [
            result["url"]
            for result in wiki_results
        ]

        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception:

        return {
            "answer":
                "Sorry, I couldn't complete the research request.",
            "sources": []
        }


# ============================================================
# COORDINATOR
# ============================================================

classification_prompt = ChatPromptTemplate.from_template(
    """
You are the Coordinator Agent for NovaTech Solutions Pvt. Ltd.

Determine which specialized agent should handle the employee request.

Available agents:

HR:
Handles company policies, leave, attendance, work from home,
employee handbook, and HR-related questions.

RESEARCH:
Handles external research, current information, technology trends,
companies, products, and general internet research.

Return ONLY one of:

HR
RESEARCH

Employee Request:

{question}
"""
)

classification_chain = classification_prompt | llm


def classify_request(question):

    response = classification_chain.invoke(
        {
            "question": question
        }
    )

    result = response.content.strip().upper()

    if "RESEARCH" in result:
        return "RESEARCH"

    if "HR" in result:
        return "HR"

    return "UNKNOWN"


# ------------------------------------------------------------
# RunnableBranch
# ------------------------------------------------------------

def run_hr(data):

    return hr_agent(data["question"])


def run_research(data):

    return research_agent(data["question"])


def run_default(data):

    return {
        "answer":
            "I can currently help with company HR questions "
            "or external research requests. Please rephrase "
            "your request.",
        "sources": []
    }


routing_chain = RunnableBranch(
    (
        lambda x: x["route"] == "HR",
        run_hr
    ),
    (
        lambda x: x["route"] == "RESEARCH",
        run_research
    ),
    run_default
)


def coordinator(question):

    route = classify_request(question)

    result = routing_chain.invoke(
        {
            "question": question,
            "route": route
        }
    )

    if route == "HR":
        agent_name = "HR Agent"

    elif route == "RESEARCH":
        agent_name = "Research Agent"

    else:
        agent_name = "Default Response"

    return {
        "agent": agent_name,
        "response": result["answer"],
        "sources": result.get("sources", [])
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏢 Enterprise AI Assistant")

    st.markdown("### Available Agents")

    st.markdown(
        """
🧑‍💼 **HR Agent**

Handles:

- Leave policies
- Attendance
- Work From Home
- Employee Handbook
"""
    )

    st.markdown(
        """
🔎 **Research Agent**

Handles:

- External research
- AI trends
- Companies
- Technology
- Current information
"""
    )

    st.divider()

    st.markdown("### System Status")

    if llm_status:
        st.success("🟢 LLM Connected")
    else:
        st.error("🔴 LLM unavailable")

    if kb_status:
        st.success("🟢 Knowledge Base Available")
    else:
        st.error("🔴 Knowledge Base unavailable")

    st.success("🟢 Coordinator Available")

    st.divider()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title("Enterprise Operations AI Assistant")

st.subheader("NovaTech Solutions Pvt. Ltd.")

st.write(
    """
Your AI-powered employee assistant for HR questions,
company knowledge, and external research.
"""
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

with st.expander("💡 Try asking"):

    st.markdown(
        """
- What is the annual leave entitlement?
- How many days can employees work from home?
- What are the standard working hours?
- Research Microsoft's latest AI products.
- What are the latest Generative AI trends?
"""
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("agent"):

                st.caption(
                    f"Agent: {message['agent']}"
                )

            if message.get("sources"):

                with st.expander("📚 Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"- {source}"
                        )

            if message.get("routing"):

                with st.expander(
                    "🔧 Agent Execution Details"
                ):

                    st.write(
                        f"Selected Agent: "
                        f"{message['agent']}"
                    )

                    st.write(
                        "Routing Method: RunnableBranch"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask an HR or research question..."
)


if user_input:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    # --------------------------------------------------------
    # Process request
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = coordinator(user_input)

                answer = result["response"]
                agent = result["agent"]
                sources = result.get("sources", [])

                st.markdown(answer)

                st.caption(
                    f"Agent: {agent}"
                )

                if sources:

                    with st.expander("📚 Sources"):

                        for source in sources:

                            st.write(
                                f"- {source}"
                            )

                with st.expander(
                    "🔧 Agent Execution Details"
                ):

                    st.write(
                        f"Selected Agent: {agent}"
                    )

                    st.write(
                        "Routing Method: RunnableBranch"
                    )

                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "agent": agent,
                        "sources": sources,
                        "routing": True
                    }
                )

            except Exception:

                error_message = (
                    "Sorry, I couldn't process that request. "
                    "Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "agent": "Error",
                        "sources": []
                    }
                )