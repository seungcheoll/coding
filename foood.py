import streamlit as st
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import random
import chromedriver_autoinstaller  # ✅ 추가됨

# 크롬 드라이버 설정 (Selenium 사용)
def setup_selenium():
    chromedriver_autoinstaller.install()  # ✅ 드라이버 자동 설치
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver

# 카카오톡 채널에서 이미지와 텍스트 추출
def crawl_kakao_channel(url):
    try:
        driver = setup_selenium()
        driver.get(url)
        time.sleep(1)
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")

        # 텍스트 추출
        title_tag = soup.find("strong", class_="tit_card")
        if title_tag:
            st.subheader(title_tag.text.strip())

        # 이미지들 추출
        image_tags = soup.find_all("div", class_="item_archive_image")
        if not image_tags:
            st.warning("❗ 이미지를 찾을 수 없습니다.")
        for div in image_tags:
            img = div.find("img")
            if img and img.get("src"):
                st.image(img["src"], width=400)

    except Exception as e:
        st.error(f"⚠ 크롤링 중 오류 발생: {e}")

# 최근 게시물만 추출
def extract_recent_posts(page_source):
    soup = BeautifulSoup(page_source, "html.parser")
    posts = soup.find_all("div", class_="area_card")
    found = 0

    for post in posts:
        date_tag = post.find("span", class_="txt_date")
        post_date = date_tag.get_text(strip=True) if date_tag else ""

        if "시간" not in post_date and "분" not in post_date:
            continue

        found += 1

        title_tag = post.find("strong", class_="tit_card")
        title = title_tag.get_text(strip=True) if title_tag else "(제목 없음)"

        desc_tag = post.find("div", class_="desc_card")
        desc = desc_tag.get_text(strip=True) if desc_tag else ""

        img_div = post.find("div", class_="wrap_fit_thumb")
        image_url = None
        if img_div and "style" in img_div.attrs:
            style = img_div["style"]
            match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
            if match:
                image_url = match.group(1)

        if desc:
            st.subheader(desc)
        if image_url:
            st.image(image_url, width=400)

    if found == 0:
        st.info("아직 오늘 메뉴가 공지되지 않았습니다.")

# 정담식당용 크롤링
def crawl_kakao_channel2(url):
    try:
        driver = setup_selenium()
        driver.get(url)
        time.sleep(1)
        page_source = driver.page_source
        driver.quit()

        extract_recent_posts(page_source)

    except Exception as e:
        st.error(f"❗ 크롤링 오류: {e}")

# 메인 앱 실행
def main():
    st.title("오늘의 메뉴's 🍽️")
    st.subheader("<<카카오톡 채널 메뉴 🍜>>")

    # ✅ URL 연결 제대로 수정
    shumhaus_url = "https://pf.kakao.com/_CiVis/108791568"
    jeongdam_url = "https://pf.kakao.com/_vKxgdn/posts"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 슈마우스만찬")
        st.text(f"▶ URL: {shumhaus_url}")
        crawl_kakao_channel(shumhaus_url)

    with col2:
        st.subheader("📌 정담식당")
        st.text(f"▶ URL: {jeongdam_url}")
        crawl_kakao_channel2(jeongdam_url)

    st.title("오늘의 식당 추천 메뉴 🍽️")
    restaurants = [
        {"name": "정담식당", "url": jeongdam_url, "func": crawl_kakao_channel2},
        {"name": "슈마우스만찬", "url": shumhaus_url, "func": crawl_kakao_channel},
    ]
    recommended = random.choice(restaurants)
    st.success(f"오늘의 추천 식당은~~~~ 🍴 {recommended['name']} 입니다!")

if __name__ == "__main__":
    main()
