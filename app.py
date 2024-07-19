import streamlit as st
import openai
from openai import OpenAI
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from sqlite3 import Connection
import sqlite3
import sklearn
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from time import sleep

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

def openaiage(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

def create_connection(db_name: str) -> Connection:
    conn = sqlite3.connect(db_name)
    return conn

def run_query(conn: Connection, query: str) -> pd.DataFrame:
    df = pd.read_sql_query(query, conn)
    return df

def create_table(conn: Connection, df: pd.DataFrame, table_name: str):
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def generate_gpt_reponse(gpt_input, max_tokens):

    # load api key from secrets
    # openai.api_key = st.secrets["openai_api_key"]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {"role": "user", "content": gpt_input},
        ]
    )

    gpt_response = completion.choices[0].message['content'].strip()
    return gpt_response

def find_assistant(id_or_name=None):
    if not id_or_name:
        return
    assistants =  client.beta.assistants.list()
    if not assistants:
        return

    for asst in assistants:
        if asst.id == id_or_name:
            return client.beta.assistants.retrieve(assistant_id=asst.id)
        if asst.name == id_or_name:
            return client.beta.assistants.retrieve(assistant_id=asst.id)
    return

def create_assistant(name):
    message_file = client.files.create(
        file=open("data/batdongsan.csv", "rb"), purpose="assistants"
    )
    file_id = message_file.id
    assistant = client.beta.assistants.create(
        name=name,
        instructions="You are an expert real estate data analyst. Use you knowledge base to answer questions about real estate and data analyst. Do not show the steps, I just need to give me the answer for the question by reading the data in the attached CSV. The answer must be in vietnamese.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o-mini",
        tool_resources={
            "code_interpreter": {
                "file_ids": [file_id]
            }
        }
    )

def generate_assistant_response(thread, user_input):
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    # Run thread with the assistant and poll for the results
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    print(run.status)
    while (run.status != "completed"):
        sleep(1)
        print("Waiting for the Assistant to respond...")
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(run.status)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    response = ''
    for m in messages:
        print(m)
        print(m.content[0].text)
        print(m.role + ": " + str(m.content[0].text.value))
        if m.role == 'assistant':
            response = response + '\n' + str(m.content[0].text.value)

    return response.strip()

lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-4o", use_cache=True)

st.sidebar.title("Ứng dụng Biểu đồ Truy vấn")
menu = st.sidebar.selectbox("Chọn một tùy chọn", ["Tóm tắt", "Hỏi đáp và tạo biểu đồ", "Hỏi đáp nhu cầu", "Hỏi đáp nhu cầu không cần data"])

# Init AI Assistant
bot_name = "Real Estate Data Analyst Assistant"
assistant = find_assistant(bot_name)
if assistant is None:
    assistant = create_assistant(bot_name)

if menu == "Tóm tắt":
    st.subheader("Tóm tắt dữ liệu của bạn")
    file_uploader = st.file_uploader("Tải lên tệp CSV của bạn", type="csv")
    if file_uploader is not None:
        path_to_save = "input.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        try:
            summary = lida.summarize("input.csv", summary_method="default", textgen_config=textgen_config)
            st.write(summary)
            goals = lida.goals(summary, n=2, textgen_config=textgen_config)
            for goal in goals:
                st.write(goal)
            library = "seaborn"
            charts = lida.visualize(summary=summary, goal=goals[0], textgen_config=textgen_config, library=library)
            img_base64_string = charts[0].raster
            img = openaiage(img_base64_string)  # Đảm bảo hàm base64_to_image được thay thế bằng hàm openaiage
            st.image(img)
        except Exception:
            st.write("Bạn có thể làm rõ yêu cầu của bạn hơn không")

if menu == "Hỏi đáp và tạo biểu đồ":
    st.subheader("Truy vấn dữ liệu của bạn để tạo biểu đồ")
    file_uploader = st.file_uploader("Tải lên tệp CSV của bạn", type="csv")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())

        if 'questions' not in st.session_state:
            st.session_state.questions = []
            st.session_state.responses = []

        question = st.text_area("Đặt câu hỏi của bạn", height=100)
        if st.button("Tạo biểu đồ"):
            if question:
                st.session_state.questions.append(question)
                combined_context = " ".join(st.session_state.questions)
                try:
                    summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                    charts = lida.visualize(summary=summary, goal=question, textgen_config=textgen_config)
                    img_base64_string = charts[0].raster
                    img = openaiage(img_base64_string)  # Đảm bảo hàm base64_to_image được thay thế bằng hàm openaiage
                    st.session_state.responses.append(img)
                except Exception:
                    st.write("Bạn có thể làm rõ yêu cầu của bạn hơn không")
        
        if st.session_state.questions:
            for i, (q, img) in enumerate(zip(st.session_state.questions, st.session_state.responses)):
                st.write(f"Câu hỏi {i+1}: {q}")
                st.image(img, caption=f"Biểu đồ cho câu hỏi {i+1}")


elif menu == "Hỏi đáp nhu cầu":
    st.subheader("Chat trực tiếp với GPT")

    file_uploader = st.file_uploader("Tải lên tệp CSV của bạn", type="csv")
    if file_uploader is not None:
        path_to_save = "chat_data.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        data = pd.read_csv(path_to_save)
        st.write("Dữ liệu CSV đã tải lên:")
        st.dataframe(data)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_area("Nhập câu hỏi của bạn", height=100)
        if st.button("Gửi"):
            if user_input:
                # try:
                
                gpt_input = f"Đọc data sau {data} và trả lời câu hỏi {user_input}"
                
                response = generate_gpt_reponse(gpt_input, max_tokens=1000)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                # except Exception:
                    
                #     chat_response = "Bạn có ohể làm rõ yêu cầu của bạn hơn không"
                #     st.session_state.chat_history.append({"role": "assistant", "content": chat_response})

        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.write(f"**Bạn**: {chat['content']}")
                elif chat["role"] == "assistant":
                    st.write(f"**Assistant**: {chat['content']}")

elif menu == "Hỏi đáp nhu cầu không cần data":
    st.subheader("Chat trực tiếp với GPT trên tệp data VN")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    thread = client.beta.threads.create()
    user_input = st.text_area("Nhập câu hỏi của bạn", height=100)
    if st.button("Gửi"):
        if user_input:
            try:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                chat_response = generate_assistant_response(thread, user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": chat_response})
            except Exception as e:
                print(e)
                chat_response = "Bạn có thể làm rõ yêu cầu của bạn hơn không"
                st.session_state.chat_history.append({"role": "assistant", "content": chat_response})

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.write(f"**Bạn**: {chat['content']}")
            elif chat["role"] == "assistant":
                st.write(f"**Assistant**: {chat['content']}")