# import module
import requests
import streamlit as st
import time

# Title
#st.title("Hello GeeksForGeeks !!!")

url = "http://localhost:8000/translation?"
#url = "http://localhost:8000/summerization?"
#uvm.


def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}


def main():
    st.set_page_config(page_title="AIServiceHub", page_icon="🤖")
    st.title("AIServiceHub - Translation EN - GER")
    session = requests.Session()
    with st.form("my_prompt"):
        prompt = st.text_input(label="prompt", key="sentence")

        submitted = st.form_submit_button("Submit")
        x = time.time()
        if submitted:
            st.write("Result")
            data = session.post(url,data ='{"sentence": "'+prompt+'"}')
            if data:
                y = time.time()
                st.write("response time in s:", float(y - x))
                st.write(data.status_code)
                st.write(data.text)
            else:
                st.error("Error")


if __name__ == '__main__':
    main()