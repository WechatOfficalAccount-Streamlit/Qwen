import streamlit as st

from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []



DEFAULT_CKPT_PATH = './models/Qwen-7B-Chat'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict(args, model, tokenizer, config, _query, _task_history):
    print(f"User: {_parse_text(_query)}")
    #_chatbot.append((_parse_text(_query), ""))
    full_response = ""

    for response in model.chat_stream(tokenizer, _query, history=_task_history, generation_config=config):
        # _chatbot[-1] = (_parse_text(_query), _parse_text(response))
        #
        # yield _chatbot
        full_response = _parse_text(response)

    print(f"History: {_task_history}")
    _task_history.append((_query, full_response))
    print(f"Qwen-Chat: {_parse_text(full_response)}")

    return [_parse_text(full_response), _task_history]



def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)

    st.success("欢迎与通义千问进行交流")
    user_input = st.chat_input("请输入你计划咨询的问题，按回车键提交！")
    if user_input is not None:
        progress_bar = st.empty()
        with st.spinner("内容已提交，通义千问模型正在作答中！"):
            question = user_input
            _task_history = st.session_state['chat_history']
            return_result = predict(args, model, tokenizer, config, question, _task_history)
            feedback = return_result[0]
            if feedback:
                progress_bar.progress(100)
                st.session_state['chat_history'].append((user_input, feedback))
                for i in range(len(st.session_state["chat_history"])):
                    user_info = st.chat_message("user")
                    user_content = st.session_state["chat_history"][i][0]
                    user_info.write(user_content)

                    assistant_info = st.chat_message("assistant")
                    assistant_content = st.session_state["chat_history"][i][1]
                    assistant_info.write(assistant_content)

                with st.sidebar:
                    if st.sidebar.button("清除对话历史"):
                        st.session_state["chat_history"] = []

            else:
                st.info("对不起，我回答不了这个问题，请你更换一个问题，谢谢！")





if __name__ == '__main__':
    main()