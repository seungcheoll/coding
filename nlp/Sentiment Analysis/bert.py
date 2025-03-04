from transformers import TFBertModel, BertTokenizer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

print('Python script_Bert is starting')

# 커스텀 모델 정의(훈련데이터로 받아온 가중치 젹용을 위한 깡통모델)
class MyBertModel(tf.keras.Model):
    def __init__(self):
        super(MyBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('klue/bert-base', from_pt=True)  # KLUE BERT 모델 사용
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류를 위한 Dense 레이어

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = output[1]  # pooled output
        return self.dense(pooled_output)  # Dense 레이어 통과

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

# 커스텀 모델 인스턴스 생성
model = MyBertModel() # 깡통모델

# 더미 입력 생성 (모델 변수 초기화를 위해)
dummy_input = tf.zeros((1, 128), dtype=tf.int32)  # 형태: (배치 크기, 최대 시퀀스 길이)
dummy_attention_mask = tf.zeros((1, 128), dtype=tf.int32)
dummy_token_type_ids = tf.zeros((1, 128), dtype=tf.int32)

model((dummy_input, dummy_attention_mask, dummy_token_type_ids))  # 더미 입력으로 모델 호출하여 변수 생성

# 가중치 불러오기 (깡통모델에 가중치 적용)
model.load_weights("weight/bert_model_weights.h5")

# 입력 데이터의 최대 길이 설정 (학습데이터와 동일한 길이로 설정)
max_seq_len = 128

# 예측 함수
def sentiment_predict(new_sentence):
    # 문장 토큰화
    input_id = tokenizer.encode(new_sentence, max_length=max_seq_len, padding='max_length', truncation=True)
    
    # attention_mask와 token_type_id 생성
    padding_count = input_id.count(tokenizer.pad_token_id)
    attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
    token_type_id = [0] * max_seq_len

    # 배열 형상 변경 및 Tensor로 변환
    input_ids = np.array([input_id], dtype=np.int64)
    attention_masks = np.array([attention_mask], dtype=np.int64)
    token_type_ids = np.array([token_type_id], dtype=np.int64)
    # 각 입력을 리스트로 전달
    score = model.predict([input_ids, attention_masks, token_type_ids])[0][0]

    # 결과 출력
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
    
    return score

# test 실행(가중치와 모델이 잘 설정 되었는지)
sentiment_predict('너무 좋아요.')

#---------------------------------------------------------------------------------------------------------#
#본 실행 코드#
# txt 파일의 내용을 한 줄씩 읽어 리스트에 저장
file_path = 'data/review.txt'  # 파일 경로를 지정하세요.

data = []  # 빈 리스트 초기화

try:
    with open(file_path, 'r', encoding='utf-8') as file:  # 파일 열기
        for line in file:  # 파일의 각 줄에 대해 반복
            stripped_line = line.strip()  # 각 줄의 앞뒤 공백 제거
            if stripped_line:  # 빈 줄이 아닐 경우
                data.append(stripped_line)  # 리스트에 추가
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {e}")

# DataFrame 생성
dict = {'review': data}
df = pd.DataFrame(dict)
df['result'] = pd.Series(dtype='object') # 결과 열 초기화

# 각 리뷰에 대해 감정 예측 수행
for idx, review in enumerate(data):
    print(f"Processing review {idx + 1} of {len(data)}")
    score = sentiment_predict(review)  # sentiment_predict 함수로 예측
    if score > 0.5:
        df.at[idx, 'result'] = '긍정'  # 0.5 초과인 경우 긍정
    else:
        df.at[idx, 'result'] = '부정'  # 0.5 이하인 경우 부정

image_file = "public/sentiment_pie_chart.png"
json_file = "public/sentiment.json" 

# 기존 이미지 삭제 (존재하는 경우)
if os.path.exists(image_file):
    os.remove(image_file)

# # 기존 JSON 삭제 (존재하는 경우)
if os.path.exists(json_file):
    os.remove(json_file)

# 비율 계산
result_counts = df['result'].value_counts()
result_counts_df=pd.DataFrame(result_counts).reset_index(inplace=False)
result_counts_df['Rate']=pd.Series(dtype='object')
result_counts_df.at[0,'Rate'] = np.round(result_counts_df.iloc[0,1]/(result_counts_df.iloc[0,1]+result_counts_df.iloc[1,1]),2)*100
result_counts_df.at[1,'Rate'] = np.round(result_counts_df.iloc[1,1]/(result_counts_df.iloc[0,1]+result_counts_df.iloc[1,1]),2)*100

# 긍정이 항상 위로 오도록
result_counts_df['sentiment'] = pd.Categorical(result_counts_df['result'], categories=['긍정', '부정'], ordered=True)
result_counts_df = result_counts_df.sort_values(by='sentiment').reset_index(drop=True)
result_counts_df=result_counts_df.iloc[:,:3]

#Dataframe을 json형식으로 저장(시각화를 위해)
result_counts_df.to_json(json_file, orient='records', force_ascii=False)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 파이 차트 그리기 (800x600 픽셀로 설정)
plt.figure(figsize=(8, 6))  # 너비: 8인치, 높이: 6인치

plt.pie(result_counts, autopct='%1.1f%%', startangle=140)

# 범례 추가
plt.legend(result_counts.index, title="리뷰 유형", loc="center right")
plt.title('긍정/부정 리뷰 비율', fontweight='bold')
plt.axis('equal')  # 원을 유지하기 위해 비율을 같게 설정

# 그림 저장 (파일 이름 및 형식 설정)(시각화를 위해)
plt.savefig(image_file, dpi=110, bbox_inches='tight')

