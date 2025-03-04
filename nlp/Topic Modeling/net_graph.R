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

df <- read_excel("melon_data.xlsx") %>% 
  mutate(Lyric = str_replace_all(Lyric, "[^가-힣a-zA-Z0-9.,!?\\s]", " ")) %>%  #한글 영어 특수문자 뺴고 삭제
  mutate(id = row_number())

df$Genre <- sub(",.*", "", df$Genre) # 맨 앞의 장르를 기준으

df_tm <- df[, c(9, 1, 7)] %>%
  mutate(Lyric = str_replace_all(Lyric, "[^가-힣]", " "), #한글빼고 제거
         Lyric = str_squish(Lyric)) %>%
  filter(Lyric != "없음" & nchar(Lyric) > 0) %>%  # '없음'과 빈 셀 제외
  as_tibble()

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

lda_model <- LDA(dtm,
                 k = 3,
                 method = "Gibbs",
                 control = list(seed = 1234))

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
    assigned_topic == 1 ~ "자연 속 감정의 `표현",
    assigned_topic == 2 ~ "사랑과 이별",
    assigned_topic == 3 ~ "일상 속 인간 관계",
    TRUE ~ "기타"  # 기타 혹은 잘못된 값 처리
  ))

# 1. 불용어와 유의어 제거
df_filtered <- final_df %>%
  filter(!is.na(assigned_topic)) %>%  # 토픽이 할당된 문서만 사용
  mutate(Lyric_clean = sapply(Lyric, preprocess_lda)) %>%  # 불용어 제거
  mutate(Lyric_clean = sapply(Lyric_clean, function(x) replace_synonyms(x, synonyms))) %>%  # 유의어 처리
  mutate(Lyric_clean = sapply(Lyric_clean, function(x) paste(x, collapse = " ")))  # 단어 합치기

# 토픽 1: "자연 속 감정의 표현" - "사랑" 단어 제거
topic_1_pairs <- df_filtered %>%
  filter(assigned_topic == 1) %>%
  unnest_tokens(bigram, Lyric_clean, token = "ngrams", n = 2) %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  filter(word1 != "사랑" & word2 != "사랑") %>%  # "사랑" 단어 제거
  count(word1, word2, sort = TRUE) %>%
  filter(n >= 7)

topic_1_graph <- graph_from_data_frame(topic_1_pairs, directed = FALSE)

ggraph(topic_1_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "green", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  theme_void() +
  labs(title = "토픽 1: 자연 속 감정의 표현",
       subtitle = "빈도 5회 이상 단어쌍 ('사랑' 제외)")



# 토픽 2: "사랑과 이별"
topic_2_pairs <- df_filtered %>%
  filter(assigned_topic == 2) %>%
  unnest_tokens(bigram, Lyric_clean, token = "ngrams", n = 2) %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n >= 10)

topic_2_graph <- graph_from_data_frame(topic_2_pairs, directed = FALSE)

ggraph(topic_2_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "pink", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  theme_void() +
  labs(title = "토픽 2: 사랑과 이별",
       subtitle = "빈도 5회 이상 단어쌍")


# 토픽 3: "일상 속 인간 관계" - "사랑" 단어 제거
topic_3_pairs <- df_filtered %>%
  filter(assigned_topic == 3) %>%
  unnest_tokens(bigram, Lyric_clean, token = "ngrams", n = 2) %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  filter(word1 != "사랑" & word2 != "사랑") %>%  # "사랑" 단어 제거
  count(word1, word2, sort = TRUE) %>%
  filter(n >= 5)

topic_3_graph <- graph_from_data_frame(topic_3_pairs, directed = FALSE)

ggraph(topic_3_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "skyblue", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  theme_void() +
  labs(title = "토픽 3: 일상 속 인간 관계",
       subtitle = "빈도 5회 이상 단어쌍 ('사랑' 제외)")
