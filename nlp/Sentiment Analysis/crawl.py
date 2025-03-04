import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

print('Python script_Crawling is starting')

# 네이버에서 영화 검색
def search_movie_on_naver(movie_title,review_num):
    # Chrome 드라이버 설정
    chrome_driver_path = "c:/py_temp/chromedriver.exe"
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service)

    # 네이버 홈페이지 열기
    driver.get("https://www.naver.com")

    # 검색창 찾기
    search_box = driver.find_element(By.NAME, "query")
    search_box.send_keys(movie_title)  # 영화 제목 입력
    search_box.submit()  # 검색 실행

    time.sleep(2)  # 검색 결과 로딩 대기

    # "관람평" 메뉴 클릭
    try:
        review_menu = driver.find_element(By.XPATH, "//span[text()='관람평']")
        review_menu.click()  # 관람평 클릭
        time.sleep(2)  # 관람평 페이지 로딩 대기
    except Exception as e:
        print(f"관람평 메뉴 클릭 오류: {e}")

    # 댓글 수집
    comments = []
    driver.execute_script("window.scrollTo(0, 0);")  # 스크롤을 맨 위로
    # 페이지 크기 조정 (80%)
    driver.execute_script("document.body.style.transform = 'scale(0.8)';")
    driver.execute_script("document.body.style.transformOrigin = '0 0';")

#---------------------------------------------------실관람객 리뷰댓글수집-------------------------------------------#

    try:     
        # 댓글 섹션 찾기
        comment_section = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[4]/div[4]')
        driver.execute_script("arguments[0].scrollIntoView();", comment_section)
        time.sleep(1)

        # 스크롤 내리기
        for _ in range(int(review_num / 10)):
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", comment_section)
            time.sleep(1)  # 잠시 대기하여 댓글 로딩

        # 댓글 수집
        for i in range(1, review_num):
            try:
                comment_xpath = f'//*[@id="main_pack"]/div[3]/div[2]/div/div/div[4]/div[4]/ul/li[{i}]/div[2]/div/span[2]'
                comment = driver.find_element(By.XPATH, comment_xpath).text
                comments.append(comment)
            except Exception as e:
                print(f"{i}번째 댓글 수집 오류: {e}")
                break  # 댓글이 더 이상 없으면 중단

    except Exception as e:
        print(f"댓글 섹션 열기 오류: {e}")

    driver.execute_script("window.scrollTo(0, 0);")  # 스크롤을 맨 위로

#---------------------------------------------------실관람객 최신댓글수집-------------------------------------------#

    try:
        # "최신순" 클릭
        try:
            latest_order = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[4]/div[2]/div[1]/div/ul/li[2]/a/span')
            latest_order.click()  # 최신순 클릭
        except Exception as e:
            print(f"최신순 클릭 오류: {e}")        
        # 댓글 섹션 찾기
        comment_section = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[4]/div[4]')
        driver.execute_script("arguments[0].scrollIntoView();", comment_section)
        time.sleep(1)

        # 스크롤 내리기
        for _ in range(int(review_num / 10)):
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", comment_section)
            time.sleep(1)  # 잠시 대기하여 댓글 로딩

        # 댓글 수집
        for i in range(1, review_num):
            try:
                comment_xpath = f'//*[@id="main_pack"]/div[3]/div[2]/div/div/div[4]/div[4]/ul/li[{i}]/div[2]/div/span[2]'
                comment = driver.find_element(By.XPATH, comment_xpath).text
                comments.append(comment)
            except Exception as e:
                print(f"{i}번째 댓글 수집 오류: {e}")
                break  # 댓글이 더 이상 없으면 중단

    except Exception as e:
        print(f"댓글 섹션 열기 오류: {e}")

#---------------------------------------------------네티즌 리뷰댓글수집---------------------------------------------#

    # 네티즌 버튼 클릭
    try:       
        driver.execute_script("window.scrollTo(0, 0);")  # 스크롤을 맨 위로
        netizen_button = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[1]/div/div/ul/li[2]/a/span')
        netizen_button.click()
        time.sleep(1)  # 클릭 후 잠시 대기

        # 댓글 섹션 찾기 (수정된 부분)
        comment_section = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[5]/div[4]')
        driver.execute_script("arguments[0].scrollIntoView();", comment_section)
        time.sleep(1)

        # 스크롤 내리기
        for _ in range(int(review_num / 10)):
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", comment_section)
            time.sleep(1)

        # 댓글 수집
        for i in range(1, review_num):
            try:
                comment_xpath = f'//*[@id="main_pack"]/div[3]/div[2]/div/div/div[5]/div[4]/ul/li[{i}]/div[2]/div/span'
                comment = driver.find_element(By.XPATH, comment_xpath).text
                comments.append(comment)
            except Exception as e:
                print(f"{i}번째 댓글 수집 오류: {e}")
                break  # 댓글이 더 이상 없으면 중단

    except Exception as e:
        print(f"댓글 섹션 열기 오류: {e}")

#---------------------------------------------------네티즌 최신댓글수집---------------------------------------------#

    try:       
        driver.execute_script("window.scrollTo(0, 0);")  # 스크롤을 맨 위로       
        # "최신순" 클릭
        try:
            latest_order = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[5]/div[2]/div[1]/div/ul/li[2]/a/span')
            latest_order.click()  # 최신순 클릭
        except Exception as e:
            print(f"최신순 클릭 오류: {e}") 

        # 댓글 섹션 찾기 (수정된 부분)
        comment_section = driver.find_element(By.XPATH, '//*[@id="main_pack"]/div[3]/div[2]/div/div/div[5]/div[4]')
        driver.execute_script("arguments[0].scrollIntoView();", comment_section)
        time.sleep(1)

        # 스크롤 내리기
        for _ in range(int(review_num / 10)):
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", comment_section)
            time.sleep(1)

        # 댓글 수집
        for i in range(1, review_num):
            try:
                comment_xpath = f'//*[@id="main_pack"]/div[3]/div[2]/div/div/div[5]/div[4]/ul/li[{i}]/div[2]/div/span'
                comment = driver.find_element(By.XPATH, comment_xpath).text
                comments.append(comment)
            except Exception as e:
                print(f"{i}번째 댓글 수집 오류: {e}")
                break  # 댓글이 더 이상 없으면 중단
    except Exception as e:
        print(f"댓글 섹션 열기 오류: {e}")

#------------------------------------------------------------------------------------------------------------------#

    # review.txt 파일 경로
    review_file_path = 'data/review.txt'
    
    # 파일이 존재하면 삭제
    if os.path.exists(review_file_path):
        os.remove(review_file_path)

    # 댓글을 review.txt 파일에 저장
    with open(review_file_path, 'w', encoding='utf-8') as f:
        for comment in comments:
            f.write(f"{comment}\n")

    # 드라이버 종료
    time.sleep(5)  # 잠시 대기 후 종료
    driver.quit()

# 영화 제목 입력
movie_title = input("검색할 영화 제목을 입력하세요: ")
review_num = int(input("각 항목당 수집할 댓글 개수을 입력하세요: "))

search_movie_on_naver(movie_title,review_num)
