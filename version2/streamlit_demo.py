import streamlit as st

def main():
    # 设置初始页面为Home
    session_state = st.session_state
    if 'page' not in session_state:
        session_state['page'] = 'introduce'

    # # 导航栏
    # page = st.sidebar.radio('Navigate', ['introduce', 'QA'])
    # st.sidebar.success("在上方选择一个演示。")
    # if page == 'introduce':
    #     page_introduce()
    # elif page == 'QA':
    #     page_QA()

    # 页面
    page1 = st.Page("introduce.py", title="introduce")
    page2 = st.Page("web_QA.py", title="QA")

    pg = st.navigation([page1, page2])
    pg.run()

if __name__ == '__main__':
    main()
