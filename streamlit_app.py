"""NFL Matchup Brief Agent: web page (Streamlit Community Cloud).
Version 1: a plain chat box. Each message is answered on its own, final answer only."""
import os
import streamlit as st
import sys
sys.stdout.reconfigure(line_buffering=True)   # write log lines right away

# Keys come from the app's Secrets settings, never from the code or the repo.
for k in ("GROQ_API_KEY", "ODDS_API_KEY", "NEWS_API_KEY"):
    try:
        os.environ[k] = st.secrets[k]
    except Exception:
        pass   # missing key: the tool that needs it says so

@st.cache_resource
def load_agent():
    import agent          # loads the schedule and stats once per server start
    return agent

agent = load_agent()

st.title("NFL Matchup Brief Agent")

if "chat" not in st.session_state:
    st.session_state.chat = []
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)

if msg := st.chat_input("Ask anything"):
    st.session_state.chat.append(("user", msg))
    st.chat_message("user").write(msg)
    try:
        final, _, _ = agent.run_agent(msg, verbose=True)
    except Exception as e:
        final = f"Something went wrong: {type(e).__name__}"
    st.session_state.chat.append(("assistant", final))
    st.chat_message("assistant").write(final)
