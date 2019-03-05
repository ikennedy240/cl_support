library(readr)
library(stringr)
library(dplyr)
library(e1071)
library(tm)
library(RTextTools)
library(caret)
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
df <- mutate(df, listingText_noKW=listingText)
# We go in reverse order so we remove more specific keywords before the less specific ones
# e.g. remove 'townhome' before we remove 'home'
for(i in dim(apart_keywords)[[1]]:1){
  df <- mutate(df, listingText_noKW = str_remove_all(listingText_noKW, regex(apart_keywords[[i,2]], ignore_case = TRUE)))
}
# This version is for in place keyword removal, to save space.
for(i in dim(apart_keywords)[[1]]:1){
  df <- mutate(df, listingText = str_remove_all(listingText, regex(apart_keywords[[i,2]], ignore_case = TRUE)))
}

Encoding(df$listingText_noKW) <- "UTF-16"

# Save the version with keyword assigned unit types and the keywords removed from the listing text.
write.csv(df, "data/seattle_sample_kwd.csv")

# For now we will only use things we have the unit type for to test.
df <- df[!is.na(df$unit_type),]

# Remove non-critical terms such as numbers, punctation, and stopwords from the corpus
corp <- Corpus(VectorSource(df$listingText_noKW))
corp <- corp %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

# Split train and test set into the first 3/4 and last 1/4 respectfully
n_train = floor(0.75*nrow(df))
n_test = nrow(df) - n_train

dtm <- DocumentTermMatrix(corp)
df.train <- df[1:n_train,]
df.test <- df[(n_train+1):(n_train+n_test),]
dtm.train <- dtm[1:n_train,]
dtm.test <- dtm[(n_train+1):(n_train+n_test),]
corp.train <- corp[1:n_train]
corp.test <- corp[(n_train+1):(n_train+n_test)]

# Remove terms in <150 (1%) of listings
dtm.train.nb <- DocumentTermMatrix(corp.train, control=list(dictionary = findFreqTerms(dtm.train, 150)))
dtm.test.nb <- DocumentTermMatrix(corp.test, control=list(dictionary = findFreqTerms(dtm.train, 150)))

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
classifierNB <- naiveBayes(trainNB, as.factor(df.train$unit_type), laplace=1)
pred <- predict(classifierNB, newdata=testNB)
View(table("Predictions"=pred, "Actual"=df.test$unit_type))

conf <- confusionMatrix(pred, as.factor(df.test$unit_type))

num_classes <- length(classifierNB$levels)
probs <- read.table(text="", colClasses=append("character", rep("double", num_classes)), col.names=append("word", classifierNB$levels))
for(i in 1:length(classifierNB[["tables"]])){
  probs[nrow(probs)+1,] <- append(attributes(classifierNB$tables)$names[i],
                                  classifierNB$tables[[i]][(num_classes+1):(num_classes*2)])
}

write.csv(probs, "data/word_probs.csv")

write.csv(as.matrix(conf, what="xtabs"), "data/unittypeNB_xtabs.csv")
write.csv(as.matrix(conf, what="overall"), "data/unittypeNB_overall.csv")
write.csv(as.matrix(conf, what="classes"), "data/unittypeNB_accuracies.csv")
