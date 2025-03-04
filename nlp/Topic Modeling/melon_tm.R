library(KoNLP)        # 한글 형태소 분석을 위한 라이브러리
library(tidyverse)     # 데이터 조작 및 시각화를 위한 라이브러리
library(dplyr)         # 데이터 프레임을 처리하는 데 필요한 패키지
library(stringr)       # 문자열 처리를 위한 패키지
library(tidytext)      # 텍스트 분석을 위한 패키지
library(readxl)        # Excel 파일을 읽기 위한 패키지
library(topicmodels)   # LDA 모델링을 위한 패키지
library(ldatuning)     # LDA 모델의 최적 K값을 찾기 위한 패키지
library(ggraph)        # 그래프 시각화를 위한 패키지
library(igraph)        # 네트워크 분석 및 시각화를 위한 패키지
library(reshape2)
library(tm)
library(ggplot2)
library(textmineR)

df <- read_excel("melon_data.xlsx") %>% 
  mutate(Lyric = str_replace_all(Lyric, "[^가-힣a-zA-Z0-9.,!?\\s]", " ")) %>%  #한글 영어 특수문자 뺴고 삭제
  mutate(id = row_number())

df$Genre <- sub(",.*", "", df$Genre) # 맨 앞의 장르를 기준으로

df %>% view()

#장르 빈도분석
df_genre <- df %>%
  group_by(Group) %>%
  count(Genre, sort = TRUE)

#빈도 top4
df_top4 <- df_genre %>%
  group_by(Group) %>%
  slice_max(order_by = n, n = 4)

df_filtered <- df_top4 %>% 
  filter(Group <= 1990)

ggplot(df_filtered, aes(x = Genre, y = n, fill = Genre)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste(Genre, n, sep = ": ")), 
            vjust = -0.3, size = 2) + # 막대 위에 레이블 추가
  facet_wrap(~ Group) +
  labs(title = "시대별 Genre 빈도분석", x = "Genre", y = "빈도") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # x축 라벨 기울이기
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬
#----------------------lda 시작---------------------------#
###lda
df_tm <- df[, c(9, 1, 7)] %>%
  mutate(Lyric = str_replace_all(Lyric, "[^가-힣]", " "), #한글빼고 제거
         Lyric = str_squish(Lyric)) %>%
  filter(Lyric != "없음" & nchar(Lyric) > 0) %>%  # '없음'과 빈 셀 제외
  as_tibble()

df_tm %>% view()

#-------데이터 전처리(형태소 추출, 불용어, 유의어)-------#
# 1. 형태소 분석 및 명사/형용사 추출 함수
preprocess_lda <- function(text, stopwords_file = "stopwords_kor.csv") {
  # 불용어 목록 불러오기
  stopwords <- read.csv(stopwords_file, stringsAsFactors = FALSE) %>%
    pull(stopwords_kor)  # 불용어가 'stopwords_kor' 컬럼에 있다고 가정
  # 형태소 분석
  words <- SimplePos22(text) %>%
    str_extract_all("([가-힣]+)/NC") %>%  #명사(NC) / 한글만을 가지고 진행
    unlist() %>%
    str_remove("/.*") %>%  # 품사 태그 제거
    unique() %>%  # 중복 제거(과적합 방지)
    .[str_length(.) > 1]  # 길이가 1인 단어 제외
  
  # 불용어 제거
  words <- words[!words %in% stopwords] 
  
  return(words)
}

# 2. 전처리된 텍스트를 `tm_data`의 Lyric 열에 적용
df_tm2 <- df_tm %>%
  mutate(Lyric_t = sapply(Lyric, preprocess_lda)) %>% 
  select(id, Lyric_t)

df_tm2 %>% view()

##유의어 제거
# 유의어 매핑 테이블 생성
synonyms <- data.frame(
  original = c("가득한", "가득한데", "가득해", "가슴깊이", "가슴속", "거짓말", "계절도", "고백할게", "골목",
               "관심", "기억들", "기억들이", "기억만", "기억속", "꽃송이", "꽃잎", "꽃향기", "그대곁에", "그댄", 
               "그댈", "그리움들이", "그사람", "그사람을", "기도해", "길거리", "길모퉁이", "길목", "남자들", 
               "남자친구", "남잔", "내게로", "내게서", "내겐", "내곁에", "내곁에서", "내곁을", "내꺼", "내마음", 
               "내마음도", "내마음은", "내마음을", "내사랑", "노랜", "노랠", "누군갈", "누군지", "눈물속", 
               "다정한", "달콤하게", "달콤한", "달콤해", "답답한", "답답해", "돌아올거야", "두눈에", "두눈을", 
               "따뜻하게", "따뜻한", "따사로운", "따스", "따스하게", "따스한", "마음들", "마음만", "마음속", 
               "말씀", "말투", "말하기", "말할게", "말해", "매일매일", "매일밤", "머리맡", "머리속", "머리카락", 
               "머린", "머릴", "머릿속", "모두들", "모든걸", "모든것을", "모든것이", "모습들", "목소", "목소릴", 
               "못잊을", "무심한", "무정한", "미련한", "미소만", "미소짓", "미안하단", "미안한", "미안해", 
               "미안해서", "바람결", "바람들", "바람부", "바람소리", "바람속", "바랄게", "반대편", "보고파", 
               "보고픈", "보내긴", "보낼게", "부모님", "부족한", "부족해", "부탁해", "불안한", "불안해", 
               "빗방울", "빗소리", "빗속", "사람들", "사람이", "사랑노래", "사랑따윈", "사랑때문에", "사랑만", 
               "사랑속", "사랑으", "사랑이", "사랑인걸", "사랑하게", "사랑하기", "사랑하긴", "사랑하리", 
               "사랑한", "사랑한단", "사랑할거", "사랑할게", "사랑해", "사랑해도", "사랑해서", "우리사랑", 
               "상상해", "생각들", "생각만", "생각한", "생각해", "생각해도", "생일날", "세상", "세상속", "이세상", 
               "세월가면", "소중한", "소중함", "소중해", "솔직하게", "솔직한", "순간들", "순정", "순진한", 
               "숨소리", "시간들", "시간들이", "시간속", "시작해", "쓸쓸", "쓸쓸하게", "쓸쓸해", "아무말도", 
               "아빠", "아침해", "안개속", "애원해도", "야속한", "약속들", "약속해", "얘기들", "얘기해", "얘길", 
               "어둔", "어둠속", "어린시절", "어린아이", "어린애", "어머님", "어색하게", "어색한", 
               "어색해", "어젯밤", "언제부턴가", "언제쯤", "언젠간", "얼굴들", "없나봐", "없는걸", "없다는걸", 
               "없대", "없을거야", "없을꺼야", "없인", "여름날", "여자들", "여자친구", "여잔", "여잘", "연락처", 
               "연락해", "연인들", "연인들이", "영원토록", "영원하길", "영원한", "영원할거", "영화속", "완벽한", 
               "완벽해", "외로울땐", "외로이", "외톨이", "용기내", "용서해", "우울한", "웃음소리", "원망하기보", 
               "위로해", "위험해", "유리창", "음악속", "이밤도", "이밤에", "이밤이", "이별한", "이상해", "이야기들", 
               "이해해", "익숙", "익숙한", "입맞추고", "자유로운", "잘못된", "장미꽃", "전화번호", "전화해", 
               "조용하게", "조용한", "주윌", "준비한", "중요한", "지루한", "지켜줄꺼야", "진실한", "진심", 
               "진정한", "차오르는", "차올라", "초라", "초라해", "초라해지", "추억들", "추억들이", "추억만", 
               "추억속", "충분해", "친구들", "친구들이", "친굴", "투명한", "특별한", "파도소리", "필요치", "필요한", 
               "필요해", "하늘위로", "하루종일", "하루하루", "하룬", "하룰", "햇살", "행복하게", "행복하기", 
               "행복하길", "행복한", "행복해", "행복해도", "행복해서", "향긋한", "향기로운", "허전", "허전한", 
               "허전해", "호숫가", "화려", "후회할거", "후회해도", "흔적들", "희미", "희미하게", 
               "희미해지"),
  synonym = c("가득", "가득", "가득", "가슴", "가슴", "거짓", "계절", "고백", "골목길", "관심", "기억", "기억", 
              "기억", "기억", "꽃밭", "꽃밭", "꽃밭", "그대", "그대", "그대", "그리움", "사람", "사람", 
              "기도", "길가", "길가", "길가", "남자", "남자", "남자", "내곁", "내곁", "내곁", "내곁", "내곁", 
              "내곁", "내곁", "마음", 
              "마음", "마음", "마음", "사랑", "노래", "노래", "누굴", "누굴", "눈물", "다정", "달콤", "달콤", 
              "달콤", "답답", "답답", "돌아오길", "두눈", "두눈", "따뜻", "따뜻", "따뜻", "따뜻", "따뜻", "따뜻", "마음", 
              "마음", "마음", "말하기", "말하기", "말하기", "말하기", "말하기", "매일", "매일", "머리", "머리", "머리", "머리", "머리",
              "머리", "모든", "모든", "모든", "모든", "모습", "목소리", "목소리", "못잊어", "무시", "무시", "미련", "미소", "미소", 
              "미안", "미안", "미안", "미안", "바람", "바람", "바람", "바람", "바람", "바램","반대", "보고싶다", "보고싶다", "보내다", 
              "보내다","부모", "부족", "부족", "부탁", "불안", "불안", "빗물","빗물","빗물", "사람", "사람", "사랑", "사랑", "사랑", 
              "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", "사랑", 
              "사랑", "사랑", "사랑", "사랑", "사랑", "상상", "생각", "생각", "생각", 
              "생각", "생각", "생일", "세계","세계","세계", "세월", "소중", "소중", "소중", 
              "솔직", "솔직", "순간","순수", "순수", "숨결", "시간", "시간", "시간", "시작", "쓸쓸한", "쓸쓸한", "쓸쓸한", "아무말", 
              "아버지", "아침", "안개", "애원", "야속", "약속", "약속", "얘기", "얘기", "얘기", "어둠", "어둠", 
              "어린", "어린", "어린", "어머니", "어색", "어색", "어색", "어제", "언제", "언제", 
              "언제", "얼굴", "없다", "없다", "없다", "없다", "없다", "없다", "없다", "여름", "여자", "여자", "여자", 
              "여자", "연락", "연락", "연인", "연인", "영원", "영원", "영원", "영원", "영화", "완벽", "완벽", 
              "외로움", "외로움", "외로움", "용기", "용서", "우울", "웃음", "원망", "위로", "위험", "유리", 
              "음악", "이밤", "이밤", "이밤", "이별", "이상한", "이야기", "이해", "익숙함", "익숙함", "입맞춤", 
              "자유", "잘못", "장미", "전화", "전화", "조용", "조용", "주위", "준비", "중요", "지루", "지켜줄거야", 
              "진심", "진심", "진정", "차오르다", "차오르다", "초라한", "초라한", "초라한", "추억", "추억", "추억", "추억", 
              "충분", "친구", "친구", "친구", "투명", "특별", "파도", "필요", "필요", "필요", "하늘", "하루", 
              "하루", "하루", "하루", "햇빛", "행복", "행복", "행복", "행복", "행복", "행복", "행복",
              "향기", "향기", "허전함", "허전함", "허전함", "호수", "화려한", "후회", "후회", 
              "흔적", "희미한","희미한","희미한"),
  stringsAsFactors = FALSE
)

# 유의어 교체 함수
replace_synonyms <- function(text, synonym_df) {
  for(i in 1:nrow(synonym_df)) {
    pattern <- paste0("\\b", synonym_df$original[i], "\\b")  # 단어 경계("\\b")를 사용하여 완전히 일치하는 단어만 교체
    text <- str_replace_all(text, pattern, synonym_df$synonym[i])
  }
  return(text)
}

# Lyric_t 열에 유의어 처리 적용
df_tm2 <- df_tm2 %>%
  mutate(Lyric_t = sapply(Lyric_t, function(x) replace_synonyms(x, synonyms)))

df_tm2$Lyric_t <- sapply(df_tm2$Lyric_t, function(x) paste(x, collapse = " "))

# 결과 확인
df_tm2%>% view()

#---------------------lda모델 dtm생성----------------#
#제거할 빈도수 선택
word_counts <- df_tm2 %>%
  unnest_tokens(word, Lyric_t) %>%
  count(word)

#히스토그램을 통해 5까지의 분포가 엄청많은 것을 파악
#lda는 빈도 기반으로 학습하는 모델이기에 빈도수가 적은 경우가 너무 많으면 필요없는
#정보가 들어가 성능에 영향을 준다. 
ggplot(word_counts, aes(x = n)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  scale_x_log10() +  # x축 로그 스케일 사용 (너무 많은 작은 수를 조정)
  labs(title = "단어 빈도수 분포",
       x = "단어 빈도수",
       y = "단어 개수") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬
## DTM 생성
# 1. 단어 빈도수 계산
word_counts <- df_tm2 %>%
  unnest_tokens(word, Lyric_t) %>%
  count(word) %>%  # 각 단어의 빈도수 계산
  filter(n >= 5)   # 빈도수가 5회 이상인 단어만 필터링
word_counts %>% view()

# 전처리된 텍스트를 lda에 들어가기위해 DTM 형태로 변환
dtm <- df_tm2 %>%
  unnest_tokens(word, Lyric_t) %>%
  filter(word %in% word_counts$word) %>%  # 5회 이상 등장한 단어만 필터링
  count(id, word) %>%  # 문서별 단어 빈도 계산
  cast_dtm(id, word, n)  # DTM 생성

#------------------lda 최적의 k값 찾기-----------------#
# 최적의 k 찾기
result <- FindTopicsNumber(
  dtm,
  topics = seq(2, 20, by = 1),  # 평가할 토픽 수 범위 설정
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",             # Gibbs 샘플링 사용
  control = list(seed = 1234),  # 시드 고정
  mc.cores = 1,                 # 코어 수 (멀티코어 지원 시 2 이상 가능)
  verbose = TRUE
)

# 빈도수가 너무 커지면 토픽간 이질성이 낮아지기에 Griffiths2004,Arun2010은 유의미한 정보를 제공하지 못한다.
# 최적의 토픽 수 시각화
FindTopicsNumber_plot(result) #CaoJuan2009가 낮을수록 Deveaud2014가 높을수록 좋은 k=3

#-----------------------------lda 모델 학습------------------#
# 모델 학습
lda_model <- LDA(dtm,
                 k = 3,
                 method = "Gibbs",
                 control = list(seed = 1234))

#---------------토픽별 중요한 단어 확인해보기---------------#
# LDA 모델에서 단어-토픽 확률(beta) 추출
beta_tidy <- tidy(lda_model, matrix = "beta")
  
beta_tidy <- beta_tidy %>% filter(topic == 3)

# beta값에 대한 그래프 확인인
beta_tidy %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  mutate(topic = paste0("Topic ", topic),
         term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = as.factor(topic))) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  scale_fill_manual(
    values = c("Topic 3" = "#1f77b4",  # 원하는 색상
               "Topic 2" = "#d62728", 
               "Topic 1" = "#2ca02c")  # 필요에 따라 색상 추가
  ) +
  labs(x = NULL, y = expression(beta),
       title = "토픽별 중요 단어어") +
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬

#----------beta값을 이용한 주제 간 관점 차이(1,2)---------------#
# 두 주제의 상위 10개 단어를 추출
beta_tidy <- tidy(lda_model, matrix = "beta")
top_terms_1_2 <- beta_tidy %>%
  filter(topic %in% c(1, 2)) %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup()

# 주제별로 좌측(주제 1), 우측(주제 2) 배치
top_terms_1_2 <- top_terms_1_2 %>%
  mutate(x = ifelse(topic == 1, -beta, beta))  # 주제 1은 음수, 주제 2는 양수로 설정
top_terms_1_2 %>% view()
# 시각화: 두 주제의 단어를 좌우로 배치
ggplot(top_terms_1_2, aes(x = x, y = beta, label = term, color = as.factor(topic), size = beta)) +
  geom_text(show.legend = FALSE) +  # 글자 크기를 beta에 따라 조정
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 1) +  # x = 0에 수직선
  labs(title = "Topic 1 and Topic 2", 
       x = "Position", y = "Beta Value") +
  theme_minimal() +
  scale_x_continuous(labels = NULL) +  # x 축의 값은 제거 (위치만 표시)
  scale_size_continuous(range = c(3, 10)) + # 글자 크기 범위 설정 (3에서 10 사이)
  scale_color_manual(values = c("1" = "#2ca02c", "2" = "#d62728", "3" = "#1f77b4")) + # 토픽 색상 설정
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬

#----------beta값을 이용한 주제 간 관점 차이(1,3)---------------#
# 두 주제의 상위 10개 단어를 추출
top_terms_1_3 <- beta_tidy %>%
  filter(topic %in% c(1, 3)) %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup()

# 주제별로 좌측(주제 1), 우측(주제 3) 배치
top_terms_1_3 <- top_terms_1_3 %>%
  mutate(x = ifelse(topic == 1, -beta, beta))  # 주제 1은 음수, 주제 3는 양수로 설정

# 시각화: 두 주제의 단어를 좌우로 배치
ggplot(top_terms_1_3, aes(x = x, y = beta, label = term, color = as.factor(topic), size = beta)) +
  geom_text(show.legend = FALSE) +  # 글자 크기를 beta에 따라 조정
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 1) +  # x = 0에 수직선
  labs(title = "Topic 1 and Topic 3",
       x = "Position", y = "Beta Value") +
  theme_minimal() +
  scale_x_continuous(labels = NULL) +  # x 축의 값은 제거 (위치만 표시)
  scale_size_continuous(range = c(3, 10)) + # 글자 크기 범위 설정 (3에서 10 사이)
  scale_color_manual(values = c("1" = "#2ca02c", "2" = "#d62728", "3" = "#1f77b4")) + # 토픽 색상 설정
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬

#----------beta값을 이용한 주제 간 관점 차이(2,3)---------------#
# 두 주제의 상위 10개 단어를 추출
top_terms_2_3 <- beta_tidy %>%
  filter(topic %in% c(2, 3)) %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup()

# 주제별로 좌측(주제 2), 우측(주제 3) 배치
top_terms_2_3 <- top_terms_2_3 %>%
  mutate(x = ifelse(topic == 2, -beta, beta))  # 주제 1은 음수, 주제 3는 양수로 설정

# 주제별로 좌측(주제 2), 우측(주제 3) 배치
ggplot(top_terms_2_3, aes(x = x, y = beta, label = term, color = as.factor(topic), size = beta)) +
  geom_text(show.legend = FALSE) +  # 글자 크기를 beta에 따라 조정
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 1) +  # x = 0에 수직선
  labs(title = "Topic 2 and Topic 3",
       x = "Position", y = "Beta Value") +
  theme_minimal() +
  scale_x_continuous(labels = NULL) +  # x 축의 값은 제거 (위치만 표시)
  scale_size_continuous(range = c(3, 10)) + # 글자 크기 범위 설정 (3에서 10 사이)
  scale_color_manual(values = c("1" = "#2ca02c", "2" = "#d62728", "3" = "#1f77b4")) + # 토픽 색상 설정
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬

#--------토픽간 상관관계 그래프(네트워크 그래프)그리기-------#
# 문서-토픽 분포 (gamma 값) 추출
gamma_tidy <- tidy(lda_model, matrix = "gamma")
gamma_tidy %>% view()
# 문서-토픽 분포를 넓은 형식으로 변환
gamma_wide <- gamma_tidy %>%
  pivot_wider(names_from = topic, values_from = gamma, names_prefix = "topic_")
gamma_wide %>% view() 
# 토픽 간 상관계수 계산 (피어슨 상관계수)
cor_matrix <- cor(gamma_wide[-1], use = "complete.obs")

# 상관계수 행렬을 긴 형식으로 변환
cor_df <- melt(cor_matrix)
colnames(cor_df) <- c("topic1", "topic2", "correlation")

# 토픽 간 상관관계를 나타내는 데이터 프레임
cor_df <- cor_df %>% filter(topic1 != topic2)  # 자기 자신과의 상관은 제외

# 그래프 객체 생성
graph <- graph_from_data_frame(cor_df)

# 네트워크 시각화
ggraph(graph, layout = "fr") +
  geom_node_point(size = 8, color = "steelblue") +  # 노드만 표시
  geom_node_text(aes(label = name), vjust = 1.8, size = 5) +  # 노드 텍스트
  scale_y_continuous(limits = c(-1,0)) +  # y축 범위를 0부터 500까지 설정
  theme_minimal() +
  labs(title = "토픽간 상관관계 (LDA)")+
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬

#------노래 가사에 id를 기준으로 topic매칭 및 주제 부여-------#
# DTM에 포함된 id 추출
dtm_ids <- as.integer(dimnames(dtm)[[1]])

# df_tm2를 DTM에 포함된 id로 필터링
df_tm2_filtered <- df_tm2 %>% filter(id %in% dtm_ids)

# 문서별 토픽 확률 추출
topic_probabilities <- posterior(lda_model)$topics

# 각 문서에 대해 가장 높은 확률의 토픽 선택
df_tm2_filtered$assigned_topic <- apply(topic_probabilities, 1, which.max)

# 결과 확인
df_tm2_filtered %>% select(id, assigned_topic) %>% view()

# 원본 데이터에 필터링된 데이터와 병합하여 토픽 번호 추가
final_df <- df %>%
  left_join(df_tm2_filtered %>% select(id, assigned_topic), by = "id")

# 결과 확인
final_df %>% view()

# topic 값에 따라 주제 제목을 부여
final_df <- final_df %>%
  mutate(topic_title = case_when(
    assigned_topic == 1 ~ "자연 속 감정의 표현",
    assigned_topic == 2 ~ "사랑과 이별",
    assigned_topic == 3 ~ "일상 속 인간 관계",
    TRUE ~ "기타"  # 기타 혹은 잘못된 값 처리
  ))

# 결과 확인
final_df %>% select(id,Group,Singer,Title,Genre, assigned_topic, topic_title) %>% view()

#----------------토픽별 장르의 분포를 비교-------------#
# "기타" 주제를 제외하고 Group과 topic_title에 따른 top3 Genre 빈도 계산
Genre_freq <- final_df %>%
  filter(topic_title != "기타") %>%   # "기타" 제거
  count(topic_title, Genre)            # topic_title과 Genre별로 빈도 계산
Genre_freq %>% view()

top4_genre <- Genre_freq %>%
  group_by(topic_title) %>%
  slice_max(order_by = n, n = 4) %>%
  ungroup()

# topic_title 순서 설정
top4_genre$topic_title <- factor(
  top4_genre$topic_title,
  levels = c("자연 속 감정의 표현", "사랑과 이별", "일상 속 인간 관계")
)

# ggplot 코드
ggplot(top4_genre, aes(x = topic_title, y = n, fill = Genre)) +
  geom_bar(stat = "identity", position = "dodge") +  # topic별 Genre 빈도를 막대그래프로 표시
  geom_text(
    aes(label = paste(Genre, n, sep = "\n")),  # Genre와 n을 결합하여 표시
    position = position_dodge(width = 0.9),  # 막대 위치와 맞도록 텍스트 위치 설정
    vjust = -0.3,  # 텍스트가 막대 위에 위치하도록 조정
    size = 3       # 텍스트 크기 설정
  ) +
  labs(
    title = " ",
    x = " ",
    y = "N",
    fill = "Genre"
  ) +
  scale_y_continuous(limits = c(0, 500)) +  # y축 범위를 0부터 500까지 설정
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +  # x축 레이블 기울기 조정
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬


# 결과를 보면 사랑과 이별은 > 발라드 일상 속 사람람과의 관계는 댄스,랩/힙합
# 자연을 배경으로 한 감정의 표현은 록/메탈 성인가요/트로트 포크/블루스에 해당하는 것을 볼 수 있다.

#-------------최종 연대별 노래의 트렌드 분석-----------------#
## 연대별 토픽의 경향성 변화
# Group을 factor로 변환하고 순서 지정
final_df$Group <- factor(final_df$Group, levels = c("1970", "1980", "1990", "2000", "2010", "2020"))

# "기타" 주제를 제외한 Group별 topic_title 빈도 계산
topic_freq <- final_df %>%
  filter(topic_title != "기타") %>%
  count(Group, topic_title)

ggplot(topic_freq, aes(x = Group, y = n, color = topic_title, group = topic_title)) + 
  geom_line(size = 1) +  # 선 그래프 그리기
  geom_point(size = 3) +  # 각 점 표시 (옵션)
  labs(
    title = "연도별 음악 트렌드의 변화",
    x = "Year",
    y = "N",
    color = "Topic Title"
  ) +
  scale_x_discrete(drop = FALSE) +  # 비어 있는 Group도 축에 표시
  scale_color_manual(
    values = c(
      "자연 속 감정의 표현" = "green",
      "사랑과 이별" = "red",
      "일상 속 인간 관계" = "blue"
    )
  ) + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  theme(plot.title = element_text(hjust = 0.5))  # 제목을 가운데 정렬
# 경향성 파악 성인가요/트로트 & 포크/블루스> 발라드 > 댄스 & 랩/힙합 순으로 빈도가 증가하는 것으로 보여진다

# 추천 시스템을 위한 최종 결과 excel로 추출
library(writexl)

# final_df를 엑셀 파일로 저장
write_xlsx(final_df, "final_df.xlsx")

