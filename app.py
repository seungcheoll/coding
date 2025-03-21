from flask import Flask, jsonify
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import time
import re

app = Flask(__name__)

def setup_selenium():
    # 크롬 드라이버 자동 설치
    chromedriver_autoinstaller.install()
    options = Options()
    options.add_argument("--headless")  # 창 없이 실행
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

@app.route("/crawl", methods=["GET"])
def crawl_kakao():
    url = "https://pf.kakao.com/_vKxgdn/posts"  # 정담식당 채널 주소
    driver = setup_selenium()
    driver.get(url)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    posts = soup.find_all("div", class_="area_card")
    result = []

    for post in posts:
        # "몇 시간 전" 또는 "몇 분 전"인 글만 추출
        date_tag = post.find("span", class_="txt_date")
        if not date_tag or ("시간" not in date_tag.text and "분" not in date_tag.text):
            continue

        # 설명 추출
        desc = post.find("div", class_="desc_card")
        desc_text = desc.text.strip() if desc else ""

        # 이미지 추출
        img_div = post.find("div", class_="wrap_fit_thumb")
        image_url = None
        if img_div and "style" in img_div.attrs:
            style = img_div["style"]
            match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
            if match:
                image_url = match.group(1)

        result.append({
            "description": desc_text,
            "image": image_url
        })

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
