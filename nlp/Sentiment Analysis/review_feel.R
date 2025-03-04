library(KoNLP)
library(tidyverse)
library(dplyr)
library(tidytext)
library(tibble)
library(ggplot2)
library(jsonlite)
useNIADic()
useSejongDic()

print("R script is starting")
# 데이터 폴더 경로 설정
data_dir <- "data"

# 데이터 폴더에서 텍스트 파일 목록 가져오기
text_files <- list.files(data_dir, pattern = "\\.txt$", full.names = TRUE)

if (length(text_files) < 1) {
  stop("data 폴더에 텍스트 파일이 없습니다.")
}

# 첫 번째 파일 선택
fil <- text_files[1]
df <- readLines(con = fil, encoding = "UTF-8")

# df를 tibble로 변환
df <- tibble(value = df)

# KoNLP를 사용하여 simplePos22로 토큰화
text_simplePos <- df %>%
  unnest_tokens(input = value,
                output = word,
                token = SimplePos22,
                drop = FALSE)

# word 열을 분리하여 word와 Pos22로 나누기
text_simplePos <- text_simplePos %>%
  separate_rows(word, sep = "\\+") %>%  # + 기준으로 분리하고 행을 확장
  separate(word, into = c("word", "Pos22"), sep = "/")  # / 기준으로 단어와 품사 분리

# 동사, 형용사, 명사 필터링
filtered_text <- text_simplePos %>%
  filter(Pos22 %in% c("nc", "pv", "pa", "px", "mm"))  # 명사, 동사, 형용사에 해당하는 품사 코드

stopwords <- read_csv("Data/stopwords_kor_emot.csv")  # 불용어 파일 읽기

filtered_text <- filtered_text %>%
  filter(!word %in% stopwords$stopwords_kor)  # 불용어 제거


synonyms <- data.frame(
  original = c('재미있', '재밌게', '아름답', '감동적', '사랑법', '재밌었어요', '재밌고', '재밌어요', '잼',
               '재밌다', '재밌었다', '재밌습니다', '좋았습니', '즐겁', '훌륭', '맛있', '맛', '미쳤',
               '사랑영화', '사랑하게', '재', '재밌네요', '재밌는데', '재밌어', '행복했', '개재밋다', '공감가고',
               '멋있다이', '재밌', '재밌네', '재밌다가도', '재밌어서', '재밌었', '재밌었던', '재밌었던거',
               '재밌었어요김고은', '좋겠', '좋았어요ost도', '좋은영화임은', '‘사랑’과', '‘사랑’을', '감동받았습니',
               '감동으', '감동이었습니다후기가좋은게', '감동적이', '감동적이었', '감명', '감명깊', '감사한',
               '감사합니', '감사합니다안', '감사해', '기대안', '기대안했는데', '기대않하고', '기대이상~이제',
               '기대이상이예용', '기대이상입니나지루함', '기대중', '기대치않고', '너무괜찮다', '너무너무너무',
               '너무너뮤노ㅓ무', '너무슬프고재밌어요', '너무재밌게봄', '너무재밌고', '너무적네', '너무좋아요재미있어',
               '따뜻함', '따뜻해서', '사랑들', '사랑스', '사랑이야기', '사랑하자', '사랑한', '잘봤', '잘봤습니',
               '잘봤습니다!', '잘봤어요', '재미도', '재미있~~~당^^', '재미있었', '재밋게', '재밌게봐서',
               '재밌게봤네', '재밌게봤어', '재밌게봤음.', '재밌나....', '재밌나요', '재밌는듯', '재밌다...',
               '재밌다고', '재밌어요.....', '재밌어용.', '재밌었고', '재밌었구요', '재밌었다.', '재밌었다...',
               '재밌었습니', '재밌었습니다', '재밌었습니다호불호가', '재밌었어', '재밌었어요웃기기도', '재밌었지만',
               '좋겠고나답게', '좋구나란', '좋긴', '좋다생각하는데,혼영', '좋습니', '좋아보였습니', '좋았습니다울면서',
               '좋았음밤이나', '좋았음솔직히', '좋었', '좋은건', '좋은세상되냐', '좋은영화', '좋은영화네요연기도',
               '좋은영화를', '좋은영화이다', '좋음동성애자'),
  synonym = c('재미', '재미', '아름다운', '감동', '사랑', '재미', '재미', '재미', '재미', '재미', '재미',
              '재미', '좋다', '즐겁다', '훌륭한', '맛있다', '맛있다', '미쳤다', '사랑', '사랑', '재미',
              '재미', '재미', '재미', '행복', '재미', '공감', '멋있다', '재미', '재미', '재미', '재미',
              '재미', '재미', '재미', '재미', '좋다', '좋다', '좋다', '사랑', '사랑', '감동', '감동',
              '감동', '감동', '감동', '감동', '감동', '감사', '감사', '감사', '감사', '기대', '기대',
              '기대', '기대', '기대', '기대', '기대', '기대', '재미', '재미', '재미', '재미', '재미',
              '재미', '재미', '재미', '따뜻하게', '따뜻하게', '사랑', '사랑', '사랑', '사랑', '사랑',
              '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미',
              '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미', '재미',
              '재미', '재미', '재미', '재미', '재미', '재미', '재미', '좋다', '좋다', '좋다', '좋다',
              '좋다', '좋다', '좋다', '좋다', '좋다', '좋다', '좋다', '좋다', '좋다', '좋다', '좋다',
              '좋다', '좋다'),
  stringsAsFactors = FALSE
)

replace_synonyms <- function(text, synonym_df) {
  for(i in 1:nrow(synonym_df)) {
    pattern <- paste0("\\b", synonym_df$original[i], "\\b")  # 단어 경계("\\b")를 사용하여 완전히 일치하는 단어만 교체
    text <- str_replace_all(text, pattern, synonym_df$synonym[i])
  }
  return(text)
}
# word 열에 유의어 처리 적용
filtered_text <- filtered_text %>%
  mutate(word = sapply(word, function(x) replace_synonyms(x, synonyms)))

dic <- read_csv("Data/knu_sentiment_lexicon.csv")

filtered_text <- filtered_text %>%
  left_join(dic, by = "word") %>%
  mutate(polarity = ifelse(is.na(polarity), 0, polarity))

# 문장별 감성 점수 합산
sentence_scores <- filtered_text %>%
  group_by(value) %>%                  # 문장별로 그룹화
  summarise(score = sum(polarity, na.rm = TRUE))  # 각 문장에 포함된 단어들의 감성 점수를 합산

sentence_scores <- sentence_scores %>%
  mutate(sentiment = ifelse(score > 0, "긍정",
                            ifelse(score < 0, "부정","중립")))

# KoNLP를 사용하여 simplePos22로 토큰화
sentence_scores1 <- sentence_scores %>%
  filter(sentiment == c("긍정","부정")) %>% 
  unnest_tokens(input = value,
                output = word,
                token = SimplePos22,
                drop = FALSE)

# word 열을 분리하여 word와 Pos22로 나누기
sentence_scores1 <- sentence_scores1 %>%
  separate_rows(word, sep = "\\+") %>%  # + 기준으로 분리하고 행을 확장
  separate(word, into = c("word", "Pos22"), sep = "/")  # / 기준으로 단어와 품사 분리

# 동사, 형용사, 명사 필터링
sentence_scores1 <- sentence_scores1 %>%
  filter(Pos22 %in% c("nc", "pv", "pa", "px", "mm"))  # 명사, 동사, 형용사에

sentence_scores1 <- sentence_scores1 %>%
  filter(!word %in% stopwords$stopwords_kor)  # 불용어 제거

# word 열에 유의어 처리 적용
sentence_scores1 <- sentence_scores1 %>%
  mutate(word = sapply(word, function(x) replace_synonyms(x, synonyms)))
sentence_scores1 %>% view()
#-----------------------------------------------------------------#
library(dplyr)
library(stringr)
library(openxlsx)
sentiment_counts <- sentence_scores1 %>%
  filter(!str_detect(word, "ㅠ|ㅜ")) %>%      
  filter(!word %in% c("사람", "영화", "세상", "이해", "인생", "어떻", "생각", "대하", "나오", "게이", "모르", "모든", "김고은", "기대", "청춘")) %>%
  filter(nchar(word) > 1 | word == "화") %>%   # 길이가 1보다 큰 단어 또는 "화"만 선택
  count(word, sentiment, sort = TRUE) %>%
  distinct(word, sentiment, .keep_all = TRUE)  # 중복 제거


top_words <- sentiment_counts %>%
  filter(sentiment %in% c("긍정", "부정")) %>%  
  group_by(sentiment) %>%
  slice_max(order_by = n, n = 10, with_ties = FALSE) %>%  # 동률 제외, 상위 10개 선택
  ungroup()

# 그래프 생성
ggplot(top_words, aes(x = reorder(word, n), y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free") +
  coord_flip() +
  scale_fill_manual(values = c("긍정" = "#87CEEB", "부정" = "#FA8072")) +
  labs(title = "긍정/부정 단어 빈도 분석",
       x = "단어",
       y = "빈도") +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "gray95", color = NA),  # 그래프 배경 흰색
    plot.background = element_rect(fill = "white", color = NA),   # 전체 배경 흰색
    panel.grid = element_line(color = "gray90"),  # 격자선 회색 (필요하면 제거 가능)
    plot.title = element_text(hjust = 0.5, face = "bold")  # 제목 가운데 정렬 및 볼드체
  )

#-----------------------------------------------------------------#
word_counts <- sentence_scores1 %>%
  filter(sentiment %in% c("긍정", "부정")) %>%
  filter(!str_detect(word, "ㅠ|ㅜ")) %>%       # 'ㅠ', 'ㅜ'가 포함된 단어 제거
  filter(word != "영화") %>%                   # '영화' 단어 제거
  filter(nchar(word) > 1 | word == "화") %>%   # 길이가 1보다 큰 단어 또는 "화"만 선택
  count(word, sentiment) %>%
  tidyr::spread(sentiment, n, fill = 0) %>%
  rename(positive = 긍정, negative = 부정)

# 로그 오즈비 계산
word_counts <- word_counts %>%
  mutate(log_odds_ratio = log((positive + 1) / (negative + 1)))

# 상위 10개 단어씩 추출
top_words <- word_counts %>%
  arrange(desc(log_odds_ratio)) %>%
  head(10) %>%
  mutate(category = "긍정") %>%
  bind_rows(
    word_counts %>%
      arrange(log_odds_ratio) %>%
      head(10) %>%
      mutate(category = "부정")
  )

# 그래프 그리기
ggplot(top_words, aes(x = reorder(word, log_odds_ratio), y = log_odds_ratio, fill = category)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("긍정" = "#87CEEB", "부정" = "#FA8072")) +
  labs(
    title = "단어별 긍정-부정 로그오즈비",
    x = "단어",
    y = "로그 오즈비",
    fill = "Sentiment"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "gray95", color = NA),  # 그래프 배경 흰색
    plot.background = element_rect(fill = "white", color = NA),   # 전체 배경 흰색
    panel.grid = element_line(color = "gray90"),  # 격자선 회색 (필요하면 제거 가능)
    plot.title = element_text(hjust = 0.5, face = "bold")  # 제목 가운데 정렬 및 볼드체
  )
#-------------------------------------------------------------#
# 감성 카운트 데이터
sentiment_data <- sentence_scores %>%
  count(sentiment) %>%
  mutate(percentage = n / sum(n) * 100)  # 각 감성의 비율 계산

# 파이 차트 그리기
pie_chart <- ggplot(sentiment_data, aes(x = "", y = n, fill = sentiment)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +  # 파이 차트로 변환
  labs(fill = "리뷰 유형", title = "긍정/부정/중립 리뷰 비율") +
  theme_void() +  # 배경과 축 제거
  geom_text(aes(label = paste0(round(percentage, 1), "%")),   
            position = position_stack(vjust = 0.5)) +  # 각 파이 조각의 중앙에 위치
  theme(
    plot.title = element_text(hjust = 0.5, vjust = -1, size = 16, face = "bold")
  ) +
  scale_fill_manual(values = c(
    "긍정" = "#87CEEB",  # 연한 파란색
    "부정" = "#FA8072",  # 연한 빨간색
    "중립" = "#90EE90"   # 연한 초록색
  ))

image_file <- "public/sentiment_pie_chart_R.png"
json_file <- "public/sentiment_R.json" 

# 기존 이미지 삭제 (존재하는 경우)
if (file.exists(image_file)) {
  file.remove(image_file)
}
# 기존 JSON 삭제 (존재하는 경우)
if (file.exists(json_file)) {
  file.remove(json_file)
}

차트를 파일로 저장
ggsave(image_file, plot = pie_chart, width = 8, height = 6, dpi = 100)

# sentiment_data를를 JSON 파일로 저장
write_json(sentiment_data, json_file, pretty = TRUE)
