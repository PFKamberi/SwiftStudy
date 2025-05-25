from openai import OpenAI
import streamlit as st
from PIL import Image 
import base64
import streamlit as st
import onnxruntime as ort
from transformers import AutoTokenizer
from rag import index_data, retrieve_text, create_client_and_collection
import os
from huggingface_hub import snapshot_download
import onnxruntime as ort
from openai import OpenAI
import tempfile
from streamlit_chat import message  

OPENAI_API_KEY = ""

# ➤ Vazei titlo kai morfopoiei ti selida
st.set_page_config(page_title="SwiftStudy", layout="wide")

@st.cache_resource
def openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_models():
    # Download the full repo containing all ONNX model shards
    model_dir = snapshot_download(repo_id="aapot/bge-m3-onnx")

    # Find the full ONNX model file inside the downloaded folder
    onnx_model_path = os.path.join(model_dir, "model.onnx")
    ort_session = ort.InferenceSession(onnx_model_path)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return tokenizer, ort_session

# Initialize client and collection
@st.cache_resource
def init_qdrant():
    return create_client_and_collection("makeathon")

tokenizer, ort_session = load_models()
qdrant_client = init_qdrant()
openai_client = openai_client()



# ➤ Emfanizei titlo kai ypotiitlo stin koryfi
st.markdown(
    "<div style='text-align: center; margin-top: 20px; font-size: 42px; font-weight: bold;'>SwiftStudy</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center; font-size: 16px;'>Your AI-powered study assistant for any subject — upload notes or type your own, and start chatting or exploring tools.</div>",
    unsafe_allow_html=True
)

# ➤ Sidebar: Emfanizei tools gia epilogi
with st.sidebar:

    st.markdown("<h4>🌐 Select Language</h4>", unsafe_allow_html=True)

    language = st.radio(
        "Choose a language:",
        options=["🇬🇧 English", "🇬🇷 Greek"],
        key="language_choice"
    )

    st.markdown("<h4>Explanation Mode</h4>", unsafe_allow_html=True)
    explanation_mode = st.radio(
    "Choose how you'd like things explained:",
    ["Beginner", "Advanced"],
    key="explanation_mode",
    index=0
    )
    
    st.markdown("<h2 style='margin-bottom: 10px;'>📚 Optional Tools</h2>", unsafe_allow_html=True)

    # ➤ Arxikopoiisi katastaseon an den yparxoun idi
    if "active_tool" not in st.session_state:
        st.session_state.active_tool = ""
    if "show_description" not in st.session_state:
        st.session_state.show_description = False

    # ➤ Lista me ta diaforetika tools pou mporei na epilexei o xristis
    tools = [
        {
            "key": "chat",
            "label": "Chat with you Data",
            "icon": "🤖",
            "description": "🤖 **Chat with you Data**: Start a conversation with the assistant to ask questions, explore ideas, clarify topics, or get customized explanations based on your uploaded content."
        },
        {
            "key": "explain",
            "label": "Simplify & Explain",
            "icon": "🧠",
            "description": "🧠 **Simplify & Explain**: Breaks down complex concepts into simpler, easier-to-understand explanations."
        },
        #{
        #    "key": "guide",
        #    "label": "Study Guide (Easy → Hard)",
        #    "icon": "📘",
        #    "description": "📘 **Study Guide (Easy → Hard)**: Creates a structured learning path, starting from the basics and moving to more advanced topics."
        #},
        {
            "key": "flashcards",
            "label": "Flashcards (Q&A Format)",
            "icon": "🃏",
            "description": "🃏 **Flashcards (Q&A Format)**: Turns your notes into flashcards with questions and answers to boost active recall and memory."
        },
        {
            "key": "summary",
            "label": "Extract & Summarize",
            "icon": "📄",
            "description": "📄 **Extract & Summarize**: Pulls out key points from your content and presents them in a short, clear summary."
        },
        {
        "key": "quiz",
        "label": "Quizzes (Multiple Choice)",
        "icon": "❓",
        "description": "❓ **Quizzes (Multiple Choice)**: Creates interactive quizzes based on your content to test your understanding and retention."
        }
    ]

    # ➤ Emfanizei koumpia gia kathe tool me xromatismo
    st.markdown("<div style='margin-bottom: 15px;'>", unsafe_allow_html=True)
    for tool in tools:
        is_active = st.session_state.active_tool == tool["key"]
        button_label = f"{tool['icon']} {tool['label']}"
        button_style = """
            background-color: #333;
            border: 2px solid red;
            color: white;
        """ if is_active else """
            background-color: #1e1e1e;
            border: 1px solid #444;
            color: white;
        """

        clicked = st.button(
            button_label,
            key=f"{tool['key']}_button",
            use_container_width=True
        )

        # ➤ An o xristis ksanapatisei to idi epilegmeno tool, to kanei unselect
        if clicked:
            st.session_state.active_tool = "" if is_active else tool["key"]
            st.session_state.show_description = False  # ➤ Krivei perigrafes otan allazei tool
            st.rerun()

        st.markdown(
            f"""<style>
                div[data-testid="stButton"] > button[title="{button_label}"] {{
                    {button_style}
                    width: 100%;
                    padding: 0.6em 1em;
                    border-radius: 8px;
                    text-align: left;
                    font-size: 16px;
                    margin-bottom: 6px;
                }}
            </style>""",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ➤ Checkbox gia emfanisi perigrrafon tools
    st.session_state.show_description = st.checkbox("📝 Show All Tool Descriptions", value=st.session_state.show_description)

    # ➤ An einai checked, emfanizei tis perigrafes gia kathe tool
    if st.session_state.show_description:
        st.markdown("<hr>", unsafe_allow_html=True)
        for tool in tools:
            st.markdown(
                f"<div style='font-size: 13px; color: #bbb; margin-bottom: 10px;'>{tool['description']}</div>",
                unsafe_allow_html=True
            )

    # ➤ Emfanizei to epilegmeno tool
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.session_state.active_tool:
        current_tool_label = next((t["label"] for t in tools if t["key"] == st.session_state.active_tool), "")
        st.success(f"✅ Selected Tool: {current_tool_label}")
    else:
        st.info("❌ No Tool Selected — using default chatbot.")

# ➤ Section gia fortosi arxeion i xeirografi eisagogi keimenou
st.markdown("---")
#st.subheader("📥 Upload or Enter Your Study Content")


if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# File uploader
uploaded_files = st.file_uploader("Drag and drop a file here (PDF, image, audio, video) or click to browse:",
    type=["pdf", "mp3", "wav", "mp4", "png", "jpg", "jpeg"], accept_multiple_files=True)

#manual_text = st.text_area("✏️ Or type/paste your content below:", height=200)

if uploaded_files is not []:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        if uploaded_file.name in st.session_state.indexed_files:
            continue 
    
        file_extension = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Indexing file..."):
            index_data(
                client=qdrant_client,
                filepath=tmp_path,
                filename=filename,
                ort_session=ort_session,
                tokenizer=tokenizer
            )

        st.session_state.indexed_files.add(filename)


# ➤ Energo tool pou exei epilexthei
active_tool = st.session_state.active_tool

# ➤ Emfanisi tou antistoixou tool interface
if active_tool == "summary":
    st.subheader("📄 Extracted Content")

    indexed_files = list(st.session_state.indexed_files)

    if not indexed_files:
        st.info("📂 Please upload a file to begin.")
        st.stop()
    else:
        filename = st.selectbox("Select a file to summarize:", options=indexed_files)


    if st.button("Select"):
        with st.spinner("Creating summary..."):
            retrieved_text = retrieve_text(None, qdrant_client, ort_session, tokenizer, "makeathon", 20, filename)
            difficulty = "begginer friendly" if st.session_state.explanation_mode == "Beginner (ELI5)" else "advanced"
            lang = "Greek"if language == "🇬🇷 Greek" else "English"
            if retrieve_text != []:
                
                prompt = f"""You are an expert summarizer. Your task is to read the following content and extract the **key points** into a short, clear, and well-organized summary.

                Instructions:
                - Focus on the most important ideas, facts, or concepts.
                - Remove any unnecessary details or repetition.
                - Present the summary in **bullet points** or **short paragraphs**.
                - Use clear, concise language that captures the essence of the content.
                - Make the summary suitable for someone who wants a **quick understanding** of the material without reading everything.
                - The style of the summary should be {difficulty} for a college student.
                - Answer in the {lang} language.

                Content to summarize: {retrieved_text}
                Return Markdown. """

                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                st.markdown(response.choices[0].message.content)
        
            else:
                st.markdown("No relevant information found.")



if active_tool == "explain":
    st.subheader("🧠 Simplified Explanation")
    st.info("AI-generated simplified explanation of the provided content.")
    
    indexed_files = list(st.session_state.indexed_files)

    if not indexed_files:
        st.info("📂 Please upload a file to begin.")
        st.stop()
    else:
        filename = st.selectbox("Select a file to summarize:", options=indexed_files)
        if st.button("Select"):
            with st.spinner("Creating explanation..."):
                retrieved_text = retrieve_text(None, qdrant_client, ort_session, tokenizer, "makeathon", 20, filename)
                difficulty = "begginer friendly" if st.session_state.explanation_mode == "Beginner (ELI5)" else "advanced"
                lang = "Greek"if language == "🇬🇷 Greek" else "English"

                if retrieve_text != []:
                    
                    prompt = f"""You are a skilled educator who specializes in breaking down complex information into simple, easy-to-understand explanations.

                    Your task is to read the following content and rewrite it in a way that:
                    - Uses plain, everyday language.
                    - Avoids technical jargon (or explains it clearly if needed).
                    - Includes helpful analogies or real-world examples where appropriate.
                    - Is clear enough for a beginner or someone with no background in the subject.
                    - Keeps the tone friendly and conversational.
                    - Organizes the explanation with bullet points or short paragraphs for easy reading.
                    - The style of the explanation should be {difficulty} for a college student.
                    - Answer in the {lang} language.

                    Content to simplify: {retrieved_text}
                    Return Markdown. """

                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.markdown(response.choices[0].message.content)
            
                else:
                    st.markdown("No relevant information found.")

elif active_tool == "flashcards":
    st.subheader("🃏 Flashcards")
    st.info("AI-generated flashcards based on your content. Please enter a topic and/or filename to generate flashcards.")

    indexed_files = list(st.session_state.indexed_files)

    if not indexed_files:
        st.info("📂 Please upload a file to begin.")
        st.stop()
    else:
        query = st.text_input("Enter your topic:")
        filename = st.selectbox("Select a file to summarize:", options=indexed_files)

    if st.button("Submit"):
        if not query and not filename:
            st.warning("Please enter a topic and/or filename.")
            st.stop()
        elif filename and filename not in set(st.session_state.indexed_files):
            st.warning("The filename you entered is not indexed. Please upload the file first.")
            st.stop()
        elif (query and (not filename or filename in set(st.session_state.indexed_files))) or (filename and not query and filename in set(st.session_state.indexed_files)):
            with st.spinner("Creating flashcards..."):
                retrieved_text = retrieve_text(query, qdrant_client, ort_session, tokenizer, "makeathon", 20, filename)
                difficulty = "easy"if st.session_state.explanation_mode == "Beginner (ELI5)" else "hard"
                lang = "Greek"if language == "🇬🇷 Greek" else "English"
                if retrieve_text != []:
                    
                    prompt = f"""You are a helpful educational assistant.

                    Based on the following content, generate a list of **flashcards** for study and review purposes. Each flashcard should be in the form of a **Question and Answer (Q&A)**.

                    Instructions:
                    - Each flashcard should contain a **concise question** and a **clear, specific answer**.
                    - Focus on **key facts, definitions, concepts, or processes**.
                    - Avoid overly broad or vague questions.
                    - Include around **10 flashcards** total, unless the content is very limited.
                    - The difficulty level of the questions should be {difficulty} for a college student.  
                    - Answer in the {lang} language.
                    - Format each flashcard clearly like this:

                        Q: What is [question]?
                        A: [answer]

                    Content: {retrieved_text}
                    Return Markdown. """

                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.markdown(response.choices[0].message.content)
            
                else:
                    st.markdown("No relevant information found.")

elif active_tool == "quiz":
    st.subheader("❓ Quizzes")
    st.info("AI-generated multiple choice questions based on your study material.")
    
    indexed_files = list(st.session_state.indexed_files)

    if not indexed_files:
        st.info("📂 Please upload a file to begin.")
        st.stop()
    else:
        query = st.text_input("Enter your topic:")
        filename = st.selectbox("Select a file to summarize:", options=indexed_files)

    
    # Retrieve text
    if st.button("Submit"):
        if not query and not filename:
            st.warning("Please enter a topic and/or filename.")
            st.stop()
        elif filename and filename not in set(st.session_state.indexed_files):
            st.warning("The filename you entered is not indexed. Please upload the file first.")
            st.stop()
        elif (query and (not filename or filename in set(st.session_state.indexed_files))) or (filename and not query and filename in set(st.session_state.indexed_files)):
            
            with st.spinner("Creating Quiz"):
                retrieved_text = retrieve_text(query, qdrant_client, ort_session, tokenizer, "makeathon", 20, filename)

                difficulty = "easy"if st.session_state.explanation_mode == "Beginner (ELI5)" else "hard"
                lang = "Greek"if language == "🇬🇷 Greek" else "English"

                if retrieve_text != []:
                    
                    prompt = f"""You are an expert educator. Based on the following extracted content from lecture slides and video transcription,
                    create a **short quiz** with **5 multiple-choice questions** that test understanding of the key concepts. For each question,
                    provide 4 options (A-D) and clearly mark the correct answer. 
                    Content: {retrieved_text}
                    Make the questions clear, concise, and relevant to the material. The difficulty level of the questions should be {difficulty}
                    for a college student. Answer in the {lang} language.
                    Return Markdown. """

                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.markdown(response.choices[0].message.content)
            
                else:
                    st.markdown("No relevant information found.")
            
elif active_tool == "chat":
    st.subheader("🤖 Chat with your data")
    st.info("Start a conversation with the assistant to ask questions, explore ideas, clarify topics, or get customized explanations based on your uploaded content.")

    indexed_files = list(st.session_state.indexed_files)

    if not indexed_files:
        st.info("📂 Please upload a file to begin.")
        st.stop()
    else:
        filename = st.selectbox("Select a file to chat with:", options=indexed_files)

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "system", "content": "You are a helpful study assistant."}
            ]

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("🔄 Clear Chat / Reset Subject"):
                st.session_state.chat_messages = [{"role": "system", "content": "You are a helpful study assistant."}]
        with col2:
            show_reset_info = st.checkbox("ℹ️", key="show_reset_info")

        if st.session_state.get("show_reset_info", False):
            st.markdown("""
                ✅ **What this button does:**
                - Clears the conversation history
                - Useful for switching topics
                - Resets the chat context
            """, unsafe_allow_html=True)

        # Display past messages
        for msg in st.session_state.chat_messages[1:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Get new user input
        user_question = st.chat_input("Ask a question about your content...")
        if user_question:
            with st.spinner("🤖 Thinking..."):
                retrieved_text = retrieve_text(None, qdrant_client, ort_session, tokenizer, "makeathon", 20, filename)

                st.session_state.chat_messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)


                messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_messages]
                lang = "Greek"if language == "🇬🇷 Greek" else "English"

                explanation_prompt = (
                    f"""
                    You are a knowledgeable tutor. Your job is to explain things in a simple, clear way that a college student can understand — avoid jargon unless necessary, and use analogies or examples when helpful.
                    Answer in the {lang} language.
                    Use the following context as your knowledge base:
                    \"\"\"{retrieved_text}\"\"\"
                    """
                    if st.session_state.explanation_mode == "Beginner"
                    else f"""
                    You are a subject matter expert. Provide thorough, technically accurate explanations suitable for an advanced college-level student.
                    Answer in the {lang} language.
                    Ground your responses in the following source material:
                    \"\"\"{retrieved_text}\"\"\"
                    """
                )

                messages.insert(0, {
                    "role": "system",
                    "content": f"You are a knowledgeable and patient tutor. {explanation_prompt}"
                })

                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )

                assistant_reply = response.choices[0].message.content

                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)

                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})
