import pandas as pd
import numpy as np
data = pd.read_excel("C:/Users/lsc/Desktop/melon/final_df.xlsx")
# 주제 번호와 주제 이름을 매핑
topic_dict = {
    1: '자연 속 감정의 표현',
    2: '사랑과 이별',
    3: '일상 속 인간 관계'
}

# 주제 목록을 출력하여 사용자 선택 유도
def select_topic():
    print("\n주제를 선택하세요 ex)1,2,3(번호로 입력해 주세요.)")
    print("1: 자연 속 감정의 표현")
    print("2: 사랑과 이별")
    print("3: 일상 속 인간 관계")
    
    while True:
        try:
            topic_num = int(input("주제 번호를 입력하세요 : "))
            if topic_num in topic_dict:
                return topic_dict[topic_num]
            else:
                print("잘못된 입력입니다. 1, 2, 3 중에서 선택해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

# 원하는 연도와 주제에 맞는 결과를 제공하는 함수
def get_songs_by_year_and_topic(year, topic):
    # 연도와 주제에 맞는 데이터 필터링
    filtered_data = data[(data['Year'] == year) & (data['topic_title'] == topic)]
    
    # 조건에 맞는 가수와 제목만 추출
    if not filtered_data.empty:
        result = filtered_data[['Singer', 'Title','URL']]
        return result
    else:
        return "해당 조건에 맞는 데이터가 없습니다."

# 사용자 입력 받기
input_year = int(input("원하는 연도를 입력하세요: "))
selected_topic = select_topic()

# 함수 호출 및 결과 출력
songs = get_songs_by_year_and_topic(input_year, selected_topic)

# 결과 출력
if not songs.empty:
    # 결과를 recommend_song.xlsx 파일로 저장
    output_path = "recommend/recommend_song.xlsx"
    songs.to_excel(output_path, index=False)
    print(f"\n추천 노래 목록이 '{output_path}'로 저장되었습니다.")
else:
    print("\n해당 조건에 맞는 데이터가 없습니다.")
