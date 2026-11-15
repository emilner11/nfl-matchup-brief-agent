"""NFL Matchup Brief Agent: web page (Streamlit Community Cloud).
Version 2 (Assignment 9 fixes):
  - Recover: the chat keeps context, so a reply to a clarifying question continues the same question.
  - Evidence: each answer passes a code check (agent.check_evidence) and shows a list of the sources
    actually checked, with times and article links."""
import os
import sys
import streamlit as st
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
    st.session_state.chat = []   # dicts: role, text, sources

def show(m):
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("sources"):
            with st.expander("Sources checked for this answer"):
                st.markdown(m["sources"])

for m in st.session_state.chat:
    show(m)

if msg := st.chat_input("Ask anything"):
    history = [(m["role"], m["text"]) for m in st.session_state.chat]
    user = {"role": "user", "text": msg}
    st.session_state.chat.append(user)
    show(user)
    try:
        final, trace, _ = agent.run_agent(msg, verbose=True, history=history)
        sources = agent.sources_list(trace) or "No tools were called for this answer."
    except Exception as e:
        final, sources = f"Something went wrong: {type(e).__name__}", ""
    reply = {"role": "assistant", "text": final, "sources": sources}
    st.session_state.chat.append(reply)
    show(reply)
