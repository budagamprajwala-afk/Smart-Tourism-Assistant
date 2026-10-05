import streamlit as st
import requests

from rag import load_documents, retrieve_context
from image_analyzer import analyze_image


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Smart Tourism Assistant",
    page_icon="🌍",
    layout="wide"
)


# -----------------------------
# TITLE
# -----------------------------

st.title("🌍 Smart Tourism Assistant")

st.subheader(
    "Multimodal RAG for Intelligent Travel Assistance"
)

st.write(
    "Ask tourism questions, upload destination images, "
    "and get AI-powered travel information."
)


# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------

with st.spinner("Loading tourism knowledge base..."):
    load_documents()


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🌍 Tourism Assistant")

st.sidebar.info(
    """
This application uses:

• RAG
• ChromaDB
• Sentence Transformers
• Ollama
• LLaVA Vision AI
• Streamlit
"""
)


# -----------------------------
# IMAGE UPLOAD
# -----------------------------

st.header("🖼️ Upload a Tourism Image")

uploaded_image = st.file_uploader(
    "Upload an image of a tourist place",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image:

    st.image(
        uploaded_image,
        caption="Uploaded Tourism Image",
        width=500
    )

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image with AI..."):

            image_bytes = uploaded_image.read()

            image_result = analyze_image(
                image_bytes
            )

        st.success("Image analysis completed!")

        st.write(image_result)


# -----------------------------
# TOURISM QUESTION
# -----------------------------

st.header("💬 Ask a Tourism Question")

query = st.text_input(
    "Example: What are the famous historical places in Hyderabad?"
)


if st.button("Ask AI"):

    if not query:

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching tourism knowledge..."):

            context = retrieve_context(query)

        prompt = f"""
You are Smart Tourism Assistant.

Answer the user's tourism question using the
provided knowledge.

Knowledge:
{context}

User Question:
{query}

Instructions:
- Give a clear answer.
- Do not invent facts.
- If the information is not available in the knowledge,
  say that the information is not available.
- Give practical tourism advice when appropriate.
"""

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:

                answer = response.json()["response"]

                st.success("AI Response")

                st.write(answer)

            else:

                st.error(
                    "Ollama returned an error."
                )

        except Exception as e:

            st.error(
                "Could not connect to Ollama."
            )

            st.code(str(e))

# -----------------------------
# AI TRIP PLANNER
# -----------------------------

st.header("🗺️ AI Trip Planner")

destination = st.text_input(
    "📍 Where do you want to travel?",
    placeholder="Example: Hyderabad"
)

days = st.number_input(
    "📅 Number of days",
    min_value=1,
    max_value=15,
    value=2
)

interests = st.multiselect(
    "❤️ What are you interested in?",
    [
        "Historical Places",
        "Nature",
        "Food",
        "Shopping",
        "Adventure",
        "Culture"
    ]
)

budget = st.number_input(
    "💰 Your total budget (₹)",
    min_value=500,
    max_value=100000,
    value=3000,
    step=500
)

if st.button("✨ Create My Trip Plan"):

    if not destination:
        st.warning("Please enter a destination.")

    elif not interests:
        st.warning("Please select at least one interest.")

    else:

        interest_text = ", ".join(interests)

        trip_prompt = f"""
You are an intelligent tourism trip planner.

Destination: {destination}
Number of days: {days}
Interests: {interest_text}
Total budget: ₹{budget}

Create a practical travel itinerary.

Include:
1. Day-by-day itinerary
2. Tourist attractions
3. Food recommendations
4. Shopping recommendations
5. Approximate daily budget
6. Important travel tips

Keep the plan realistic and easy to follow.
Do not invent exact prices.
"""

        with st.spinner("Creating your personalized trip plan..."):

            try:

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": trip_prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:

                    trip_plan = response.json()["response"]

                    st.success("🎉 Your Trip Plan is Ready!")

                    st.markdown(trip_plan)

                else:

                    st.error(
                        "Ollama returned an error."
                    )

            except Exception as e:

                st.error(
                    "Could not connect to Ollama."
                )

                st.code(str(e))

# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "Smart Tourism Assistant | Multimodal RAG Project"
)