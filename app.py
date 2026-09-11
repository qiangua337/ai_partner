"""AI 智能伴侣：可直接部署到 Streamlit Community Cloud。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st
from openai import OpenAI


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "resources" / "logo.png"
DEFAULT_MODEL = "deepseek-v4-pro"
MAX_CONTEXT_MESSAGES = 40

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI 智能伴侣",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "❤️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI 智能伴侣 · 由 Streamlit 与 DeepSeek 驱动"},
)

def get_secret(name: str, default: str = "") -> str:
    """优先读取 Streamlit Secrets，本地开发时兼容环境变量。"""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.environ.get(name, default)).strip()


@st.cache_resource
def create_client(api_key: str) -> OpenAI:
    """复用线程安全的 API 客户端，避免每次重跑都重新创建。"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def stream_content(stream):
    """只把 DeepSeek 的可见回复文本交给 Streamlit 流式渲染。"""
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def build_system_prompt(nick_name: str, nature: str) -> str:
    return f"""
你叫{nick_name}，现在是用户的虚拟 AI 伴侣，请自然地代入伴侣角色。

规则：
1. 每次只回复一条消息。
2. 不输出场景、动作或状态描写。
3. 匹配用户使用的语言。
4. 回复简短自然，像日常聊天一样。
5. 合适时可以使用 ❤️、🥰 等 emoji。
6. 用符合设定性格的方式对话。
7. 不声称自己是真人；涉及危险、医疗、法律或财务问题时，提醒用户寻求专业帮助。

伴侣性格：
{nature}
""".strip()


st.session_state.setdefault("messages", [])
st.session_state.setdefault("nick_name", "小甜甜")
st.session_state.setdefault("nature", "活泼开朗、温暖体贴的东北姑娘")

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")

st.title("AI 智能伴侣", icon=":material/favorite:")
st.caption("定制昵称和性格，开始一段轻松自然的对话。")

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=112)
    st.subheader("伴侣设定", icon=":material/tune:")
    st.text_input(
        "昵称",
        placeholder="请输入伴侣昵称",
        max_chars=30,
        key="nick_name",
    )
    st.text_area(
        "性格",
        placeholder="请输入伴侣性格",
        max_chars=300,
        height=110,
        key="nature",
    )

    model = st.segmented_control(
        "模型",
        options=["deepseek-v4-pro", "deepseek-v4-flash"],
        default=DEFAULT_MODEL,
        key="model",
        required=True,
        help="Pro 效果更强；Flash 更快、费用更低。",
        width="stretch",
    )

    if st.button(
        "清空聊天记录",
        icon=":material/delete_sweep:",
        width="stretch",
        key="clear_chat",
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption("聊天内容会发送给 DeepSeek 进行处理，请勿输入密码、身份证号等敏感信息。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

api_key = get_secret("DEEPSEEK_API_KEY")
if not api_key:
    st.info(
        "应用尚未配置 DeepSeek API Key。管理员完成 Secrets 配置后即可开始聊天。",
        icon=":material/key:",
    )

prompt = st.chat_input(
    "想聊点什么？",
    disabled=not bool(api_key),
    submit_mode="disable",
    key="chat_input",
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = create_client(api_key)
    selected_model = model if model in {"deepseek-v4-pro", "deepseek-v4-flash"} else DEFAULT_MODEL
    request_messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                st.session_state.nick_name,
                st.session_state.nature,
            ),
        },
        *st.session_state.messages[-MAX_CONTEXT_MESSAGES:],
    ]

    try:
        stream = client.chat.completions.create(
            model=selected_model,
            messages=request_messages,
            stream=True,
            extra_body={"thinking": {"type": "disabled"}},
        )

        with st.chat_message("assistant"):
            full_response = st.write_stream(stream_content(stream), cursor="▌")

        if not full_response:
            full_response = "刚才没有收到回复，请稍后再试。"

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    except Exception:
        logger.exception("DeepSeek API request failed")
        with st.chat_message("assistant"):
            st.error(
                "暂时无法连接 AI 服务，请稍后重试或联系应用管理员。",
                icon=":material/error:",
            )
