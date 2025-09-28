"""
FAQ Chatbot System - TASK 2 Complete Implementation

Author: Mohamed Mostafa
A self-contained FAQ chatbot with advanced NLP processing and production-ready Streamlit UI.
Features: spaCy preprocessing, TF-IDF similarity matching, chat-bubble interface, 
responsive design, and accessibility features.

Usage: python faq_chatbot.py (Streamlit UI) or python faq_chatbot.py --cli (console)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import streamlit as st
except ImportError as e:
    print(
        f"Missing dependencies. Install with: pip install pandas scikit-learn spacy streamlit requests"
    )
    sys.exit(1)


class FAQChatbot:

    def __init__(self, csv_file='faqs.csv'):
        self.csv_file = csv_file
        self.df = None
        self.nlp = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._ensure_spacy()
        self._ensure_faq_data()
        self._load_and_preprocess()

    def _ensure_spacy(self):
        """Load spaCy model, download if needed"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except IOError:
            print("Downloading spaCy English model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def _ensure_faq_data(self):
        """Download sample FAQ data if faqs.csv doesn't exist"""
        if not os.path.exists(self.csv_file):
            print(
                f"File {self.csv_file} not found. Creating sample Zoom FAQ data..."
            )
            sample_faqs = [
                ("How do I start a Zoom meeting?",
                 "Click 'New Meeting' in the Zoom app or go to zoom.us and click 'Host a Meeting'."
                 ),
                ("How do I join a Zoom meeting?",
                 "Click the meeting link, use the meeting ID, or dial in by phone."
                 ),
                ("How do I share my screen in Zoom?",
                 "Click 'Share Screen' at the bottom, then select what to share."
                 ),
                ("How do I mute and unmute in Zoom?",
                 "Click the microphone icon or use spacebar to temporarily unmute."
                 ),
                ("How do I turn my video on and off?",
                 "Click the video camera icon at the bottom left."),
                ("What is the maximum number of participants?",
                 "Basic accounts support up to 100 participants."),
                ("How do I record a Zoom meeting?",
                 "Click 'Record' at the bottom. Choose local or cloud recording."
                 ),
                ("How do I use breakout rooms?",
                 "Host clicks 'Breakout Rooms' and assigns participants to rooms."
                 ),
                ("How do I change my background?",
                 "Go to Settings > Background & Filters, choose virtual background."
                 ),
                ("How do I schedule a meeting?",
                 "In Zoom app, click 'Schedule' and set meeting details."),
                ("What internet speed do I need?",
                 "Zoom recommends at least 1.5 Mbps upload and download."),
                ("How do I use the whiteboard?",
                 "Click 'Share Screen' and select 'Whiteboard'."),
                ("How do I manage participants?",
                 "Click 'Participants' to see attendees and manage permissions."
                 ),
                ("Can I use Zoom without the app?",
                 "Yes, join through web browser, though app has more features."
                 ),
                ("How do I change my display name?",
                 "Click your name in participants list and select 'Rename'."),
                ("How do I enable captions?",
                 "Click 'Live Transcript' for automatic captions."),
                ("What is Zoom waiting room?",
                 "Feature that lets hosts control when participants join for security."
                 ),
                ("How do I use chat?",
                 "Click 'Chat' to send messages to all or privately."),
                ("How do I raise my hand?",
                 "Click 'Participants' then 'Raise Hand' at bottom."),
                ("Can I use Zoom on mobile?",
                 "Yes, iOS and Android apps available with most features."),
                ("How do I test audio/video?",
                 "Go to Settings > Audio/Video to test before meeting."),
                ("What is a meeting ID?",
                 "Unique 9-11 digit number identifying each meeting."),
                ("How do I enable HD video?",
                 "Settings > Video and check 'Enable HD'."),
                ("How do I use reactions?",
                 "Click 'Reactions' for emoji responses like thumbs up."),
                ("Can I join by phone?",
                 "Yes, every meeting includes dial-in numbers for audio."),
                ("What is a personal meeting room?",
                 "Permanent meeting room with fixed ID you can use anytime."),
                ("What is screen annotation?",
                 "Drawing tools available when someone shares screen."),
                ("How do I manage account settings?",
                 "Sign in at zoom.us and go to Settings."),
                ("Can I have multiple meetings?",
                 "Basic accounts can only host one meeting at a time."),
                ("How do I end a meeting?",
                 "Click 'End Meeting' and select 'End Meeting for All'."),
                ("What security features exist?",
                 "Waiting rooms, passwords, host controls, meeting locks."),
                ("How do I update Zoom?",
                 "App prompts for updates or check in settings.")
            ]
            df = pd.DataFrame(sample_faqs, columns=['question', 'answer'])
            df.to_csv(self.csv_file, index=False)

    def _clean_text(self, text):
        """Clean and preprocess text using spaCy"""
        if pd.isna(text) or not text: return ""
        text = str(text).lower()
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        doc = self.nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
            and token.is_alpha and len(token.text) >= 2
        ]
        return ' '.join(tokens)

    def _load_and_preprocess(self):
        """Load CSV and preprocess questions"""
        self.df = pd.read_csv(self.csv_file)
        if 'question' not in self.df.columns or 'answer' not in self.df.columns:
            raise ValueError(
                "CSV must contain 'question' and 'answer' columns")
        self.df['clean_question'] = self.df['question'].apply(self._clean_text)
        self.df = self.df[self.df['clean_question'].str.len() > 0].reset_index(
            drop=True)
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                                min_df=1,
                                                stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df['clean_question'])

    def reply(self, user_question: str):
        """Generate reply with similarity scores"""
        if not user_question.strip(): return "Please ask a question.", []
        clean_q = self._clean_text(user_question)
        if not clean_q: return "I couldn't understand your question.", []

        user_vector = self.tfidf_vectorizer.transform([clean_q])
        similarities = cosine_similarity(user_vector,
                                         self.tfidf_matrix).flatten()

        # Get top 3 matches
        top_indices = similarities.argsort()[-3:][::-1]
        top_matches = [(self.df.iloc[i]['question'], self.df.iloc[i]['answer'],
                        similarities[i]) for i in top_indices]

        best_similarity = similarities.max()
        if best_similarity >= 0.6:
            return self.df.iloc[similarities.argmax()]['answer'], top_matches
        return "I couldn't find a close match for your question. Try rephrasing or contact our support team for assistance! 💬", top_matches

    def run_console(self):
        """Console interface"""
        print("FAQ Bot ready. Type 'quit' to exit:")
        while True:
            try:
                q = input("\n> ").strip()
                if q.lower() in ['quit', 'exit', 'q']: break
                if q:
                    response, _ = self.reply(q)
                    print(f"\nBot: {response}")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


def init_streamlit_ui():
    """Initialize Streamlit UI with custom styling"""
    st.set_page_config(page_title="FAQ Chatbot",
                       page_icon="🤖",
                       layout="wide",
                       initial_sidebar_state="collapsed")

    # Simplified CSS without animations
    st.markdown("""
    <style>
        .stApp > header {visibility: hidden;}
        .stApp {padding-top: 0rem;}
        .chat-container {max-width: 800px; margin: 0 auto; padding: 1rem;}
        .chat-message {margin: 1rem 0; display: flex; align-items: flex-start;}
        .chat-message.user {flex-direction: row-reverse;}
        .chat-bubble {
            max-width: 70%; padding: 0.8rem 1.2rem; border-radius: 20px; 
            margin: 0 0.5rem; font-size: 16px; line-height: 1.4;
        }
        .chat-bubble.user {
            background: #007bff; color: white; border-bottom-right-radius: 5px;
        }
        .chat-bubble.bot {
            background: #f1f3f4; color: #333; border-bottom-left-radius: 5px;
        }
        .avatar {
            width: 40px; height: 40px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; font-size: 20px;
            flex-shrink: 0;
        }
        .avatar.user {background: #007bff; color: white;}
        .avatar.bot {background: #28a745; color: white;}
        .input-container {
            background: white; padding: 1rem; border-top: 1px solid #ddd;
        }
        @media (max-width: 768px) {
            .chat-bubble {max-width: 85%; font-size: 14px;}
            .chat-container {padding: 0.5rem;}
        }
    </style>
    """,
                unsafe_allow_html=True)


def run_streamlit_ui():
    """Main Streamlit UI"""
    init_streamlit_ui()

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FAQChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.title("FAQ Chatbot")

    # New chat and export buttons
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🗑️ New Chat"):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("📄 Export Chat") and st.session_state.messages:
            chat_md = "# FAQ Chat Export\n\n"
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Bot"
                chat_md += f"**{role}:** {msg['content']}\n\n"
            chat_md += f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
            st.download_button("💾 Download", chat_md, "faq_chat.md",
                               "text/markdown")

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            role_class = "user" if message["role"] == "user" else "bot"
            avatar = "👤" if message["role"] == "user" else "🤖"

            st.markdown(f"""
            <div class="chat-message {role_class}">
                <div class="avatar {role_class}">{avatar}</div>
                <div class="chat-bubble {role_class}">{message["content"]}</div>
            </div>
            """,
                        unsafe_allow_html=True)

            # Show debug info for bot messages
            if message["role"] == "assistant" and "matches" in message:
                st.markdown("**🔍 Debug: Top 3 Matches**")
                for i, (q, a, score) in enumerate(message["matches"], 1):
                    st.write(f"**{i}. Similarity: {score:.1%}**")
                    st.write(f"Q: {q}")
                    st.write(f"A: {a[:100]}...")
                    if i < len(message["matches"]):
                        st.markdown("---")

    # Input area
    st.markdown("<br><br>",
                unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask your question:",
                placeholder="How do I start a Zoom meeting?",
                label_visibility="collapsed",
                key="user_input")
        with col2:
            send_clicked = st.button("Send",
                                     type="primary",
                                     use_container_width=True)

    # Process input - simplified without typing animation
    if send_clicked and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip()
        })

        # Get bot response immediately without delay
        response, matches = st.session_state.chatbot.reply(user_input.strip())

        # Add bot message immediately
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "matches": matches
        })

        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 14px;'>"
        f"Created by Mohamed Mostafa • {datetime.now().strftime('%Y-%m-%d')}"
        f"</div>",
        unsafe_allow_html=True)


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="FAQ Chatbot System")
    parser.add_argument('--cli',
                        action='store_true',
                        help='Launch console interface')
    args = parser.parse_args()

    if args.cli:
        chatbot = FAQChatbot()
        chatbot.run_console()
    else:
        run_streamlit_ui()


if __name__ == "__main__":
    main()