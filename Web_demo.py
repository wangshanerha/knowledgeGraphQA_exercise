import streamlit as st

def main():
    # 设置初始页面为Home
    session_state = st.session_state
    if 'page' not in session_state:
        session_state['page'] = 'introduce'

    # 页面
    page1 = st.Page("Web_introduce.py", title="Introduce")
    page2 = st.Page("Web_KGshow.py", title="KGshow")
    page3 = st.Page("Web_QA.py", title="QA")

    pg = st.navigation([page1, page2,page3])
    pg.run()

if __name__ == '__main__':
    main()
