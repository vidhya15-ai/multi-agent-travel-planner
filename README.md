# ✈️ Multi-Agent AI Travel Planner

A production-grade, multi-agent AI Travel Planning Assistant powered by **LangGraph**, **ChatGroq**, **Streamlit**, and **PostgreSQL**. The system features state persistence across user sessions, guardrails for intent classification, and multi-tool integration (Flight Search & Tavily Hotel Search).

---

## 🌟 Key Features

* **Multi-Agent Architecture**: Modular pipeline that breaks down travel planning into specialized, collaborative agents.
* **Intent Guardrails & Triage**: An entry point `intent_classifier` routes requests. Travel planning queries enter the tool pipeline, while general greetings and off-topic prompts route to a polite fallback agent without hallucinating or executing unnecessary tool calls.
* **Persistent Session State**: Full workflow checkpointer powered by `PostgresSaver` and PostgreSQL, ensuring thread-level history and recovery.
* **Streamlit UI**: Clean interactive frontend featuring session thread tracking, real-time agent execution status, and markdown itinerary exports.

---

## 🤖 Agent Roles & Responsibilities

1. **`intent_classifier` (Router Node)**
   * **Role:** Guardrail & Triage Router
   * **Function:** Analyzes the incoming query and classifies it as either `travel_plan` or `general_chat`. It acts as a gatekeeper to ensure search tools are only invoked when required.

2. **`general_chat_agent` (Fallback Node)**
   * **Role:** Conversational & Boundary Manager
   * **Function:** Safely handles greetings and off-topic queries. Prompts users for travel details or politely explains system boundaries for non-travel requests.

3. **`flight_agent` (Travel Step 1)**
   * **Role:** Flight Retrieval Specialist
   * **Function:** Queries flight search tools/APIs using the user request and stores the raw flight choices into `flight_result`.

4. **`hotel_agent` (Travel Step 2)**
   * **Role:** Accommodation & Search Specialist
   * **Function:** Calls Tavily Search to gather real-time hotel options, accommodation details, and local highlights into `hotel_result`.

5. **`itinerary_agent` (Travel Step 3)**
   * **Role:** Master Schedule Architect
   * **Function:** Integrates raw flight and hotel state outputs to construct a comprehensive, day-by-day itinerary tailored to user preferences.

6. **`final_agent` (Travel Step 4)**
   * **Role:** Quality Control & Final Editor
   * **Function:** Formats and polishes the complete travel package with clean markdown headers, transit details, and practical tips while preventing pricing or availability hallucinations.

---

## 🏗️ Architecture Flow

```
                  ┌──────────────────────┐
                  │      User Input      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  intent_classifier   │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
     (TRAVEL_PLAN)                     (GENERAL_CHAT)
            │                                 │
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │  flight_agent   │               │  general_chat   │
   └────────┬────────┘               └────────┬────────┘
            │                                 │
            ▼                                 │
   ┌─────────────────┐                        │
   │   hotel_agent   │                        │
   └────────┬────────┘                        │
            │                                 │
            ▼                                 │
   ┌─────────────────┐                        │
   │ itinerary_agent │                        │
   └────────┬────────┘                        │
            │                                 │
            ▼                                 │
   ┌─────────────────┐                        │
   │   final_agent   │                        │
   └────────┬────────┘                        │
            │                                 │
            ▼                                 ▼
         [ END ]                           [ END ]
```

---

## 📁 Repository Structure

```text
.
├── app.py                 # Streamlit Frontend Web Application
├── main.py                # LangGraph StateGraph Definition & Logic
├── tools/
│   ├── flight_tool.py     # Flight search API wrapper
│   └── tavily_tool.py     # Tavily search integration for hotels
├── .env                   # Environment variables (API keys & DB URL)
├── requirements.txt       # Project dependencies
└── README.md              # Project Documentation
```

---

## 🛠️ Prerequisites & Setup

### 1. Requirements
* Python 3.10+
* PostgreSQL Database (Local instance or cloud hosted via [Render](https://render.com))

### 2. Environment Variables
Create a `.env` file in the root directory and configure the following keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key

# Database URL (Render External or Local Postgres URL)
DATABASE_URL=postgresql://username:password@host:5432/database_name
```

---

## 🚀 Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Terminal Interface (CLI)
To run the terminal version of the application:
```bash
python main.py
```

### 3. Run Streamlit UI
To launch the interactive web app:
```bash
streamlit run app.py
```

---

## 🗄️ Database Management & Verification

The graph automatically initializes its checkpoint tables (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`) inside PostgreSQL on first startup via `checkpointer.setup()`.

### Querying Conversation History (pgAdmin / SQL)
To view raw writes logged by thread ID:
```sql
SELECT thread_id, channel, type, task_id, idx 
FROM public.checkpoint_writes 
WHERE thread_id = 'travel_session_001' 
ORDER BY idx ASC;
```