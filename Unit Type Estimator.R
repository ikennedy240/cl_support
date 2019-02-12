library(readr)
library(stringr)
library(dplyr)
library(e1071)
cl_data_path = 'data/seattle_sample.csv' # set the path to your CL extract
df <- read_csv(cl_data_path) # read data using readr

Encoding(df$listingText) <- "UTF-16"

# +++----------- Regex Keyword Classifier ----------+++
apart_keywords <- read_csv("resources/apartment_keywords.txt") # grab a keyword file I made
df['unit_type'] <- 'none' # set the base value to none

for(i in 1:dim(apart_keywords)[[1]]){ # loop through types
  # I'm using this one to look at overlap between the regex categories, to find edge cases
  # use one after this for actual categorization
  df$unit_type <- if_else(
    str_detect(df$listingText, regex(apart_keywords[[i,2]], ignore_case=TRUE)),
    if_else(df$unit_type == 'none', apart_keywords[[i,1]], paste(df$unit_type, apart_keywords[[i,1]], sep="+")),
    df$unit_type
  )

}

for(i in 1:dim(apart_keywords)[[1]]){ # loop through types
  # I'm using the loop here to ensure mutual exclusivity here: I've ordered the keyword list so that
  # it starts with broader categories and gets more specific. This is, I think, a simple and useful
  # heuristic, but it's not necessarily perfect
  df$unit_type <- if_else(
    str_detect(df$listingText, regex(apart_keywords[[i,2]], ignore_case=TRUE)),
    apart_keywords[[i,1]],
    df$unit_type
  )
  
}

df %>% group_by(GEOID10) %>% # group by tract
  count(unit_type) %>% # count the unit types
  rename(unit_counts = n) %>% # rename just for readability
  inner_join(df %>% count(GEOID10), by = 'GEOID10') %>% # join with total counts (there must be a better way)
  mutate(proportion = unit_counts/n) %>% # calculate estimated proportions
  write_csv('data/estimated_proportions_by_tract.csv') # write to disk



# +++----------- Naive Bayes Classifier ----------+++

df$unit_type[df$unit_type=='none'] = NA

# Remove all of the matching terms from the listing texts
for(i in 1:dim(apart_keywords)[[1]]){
  df <- mutate(df, listingText_noKW = str_remove_all(listingText, regex(apart_keywords[[i,2]])))
}

# Remove random non-critical terms such as numbers, punctation, and stopwords from the corpus
corp <- Corpus(VectorSource(df$listingText_noKW))
corp <- corp %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corp)
df.train <- df[1:15000,]
df.test <- df[15001:20000,]
dtm.train <- dtm[1:15000,]
dtm.test <- dtm[15001:20000,]
corp.train <- corp[1:15000]
corp.test <- corp[15001:20000]

# Remove terms in <10 listings
dtm.train.nb <- DocumentTermMatrix(corp.train, control=list(dictionary = findFreqTerms(dtm.train, 10)))
dtm.test.nb <- DocumentTermMatrix(corp.test, control=list(dictionary = findFreqTerms(dtm.train, 10)))

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
classifierNB <- naiveBayes(trainNB, df.train$unit_type, laplace=1)
pred <- predict(classifierNB, newdata=testNB)
table("pred"=pred, "actual"=a)
