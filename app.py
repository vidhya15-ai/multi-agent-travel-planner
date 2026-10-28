import os
import streamlit as st
from langchain_core.messages import HumanMessage
from main import app  # Import the compiled LangGraph app from your main.py

# Page Configuration
st.set_page_config(
    page_title="AI Travel Planner Agent", page_icon="✈️", layout="centered"
)

st.title("✈️ AI Travel Planning Assistant")
st.write(
    "Powered by LangGraph multi-agent architecture and PostgreSQL"
    " checkpointer."
)

# Initialize a session state thread_id if it doesn't exist
if "thread_id" not in st.session_state:
  st.session_state.thread_id = "travel_session_001"

# Sidebar configuration for thread management
with st.sidebar:
  st.header("Session Settings")
  thread_input = st.text_input("Thread ID", value=st.session_state.thread_id)
  if thread_input:
    st.session_state.thread_id = thread_input

  st.info(
      "Your graph state is safely checkpointed to PostgreSQL using this"
      " thread ID."
  )

# Main User Input Form
user_query = st.text_area(
    "Enter your travel request:",
    placeholder=(
        "e.g., Plan a 4-day trip to Tokyo with flights and boutique hotels."
    ),
)

if st.button("🚀 Generate Travel Plan", type="primary"):
  if not user_query.strip():
    st.warning("Please enter a valid travel request.")
  else:
    # Config for LangGraph checkpointer
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Initial state payload matching your TravelState TypedDict
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "flight_result": "",
        "hotel_result": "",
        "itinerary": "",
        "llm_calls": 0,
    }

    # Run the LangGraph app with status indicator
    with st.status(
        "🤖 Multi-agent crew at work (Flights → Hotels → Itinerary → Final)...",
        expanded=True,
    ) as status:
      try:
        st.write("🔍 `flight_agent` fetching flight data...")
        # Invoke your compiled langgraph app
        result = app.invoke(initial_state, config=config)

        status.update(
            label="✅ Travel plan generated successfully!",
            state="complete",
            expanded=False,
        )

        # Display Final Travel Plan output
        st.divider()
        st.subheader("🗺️ Final Travel Plan")
        final_output = result["messages"][-1].content
        st.markdown(final_output)

        # Optional metrics
        st.caption(
            f"Total LLM Invocations used in graph: {result.get('llm_calls', 'N/A')}"
        )

        # Download button
        st.download_button(
            label="Download Travel Plan (.md)",
            data=final_output,
            file_name="travel_plan.md",
            mime="text/markdown",
        )

      except Exception as e:
        status.update(label="❌ Error during execution", state="error")
        st.error(f"An error occurred: {e}")