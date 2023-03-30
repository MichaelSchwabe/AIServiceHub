# import module
import requests
import streamlit as st

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
    st.title("My AIServiceHub - Translation EN - GER Service")
    session = requests.Session()
    with st.form("my_prompt"):
        prompt = st.text_input(label="prompt", key="sentence")

        submitted = st.form_submit_button("Submit")

        if submitted:
            st.write("Result")
            data = session.post(url,data ='{"sentence": "'+prompt+'"}')
            if data:
                st.write(data.status_code)
                st.write(data.text)
            else:
                st.error("Error")


if __name__ == '__main__':
    main()