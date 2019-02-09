library(readr)
library(stringr)
library(dplyr)
cl_data_path = 'data/cl_full.csv' # set the path to your CL extract
df <- read_csv(cl_data_path) # read data using readr
apart_keywords <- read_csv("resources/apartment_keywords.txt") # grab a keyword file I made
df['unit_type'] <- 'none' # set the base value to none
for(i in 1:dim(apart_keywords)[[1]]){ # loop through types
  # I'm using the loop here to ensure mutual exclusivity here: I've ordered the keyword list so that
  # it starts with broader categories and gets more specific. This is, I think, a simple and useful
  # heuristic, but it's not necessarily perfect
  df$unit_type <- if_else(
    str_detect(df$listingText, apart_keywords[[i,2]]),apart_keywords[[i,1]],df$unit_type) 
}

df %>% group_by(GEOID10) %>% # group by tract
  count(unit_type) %>% # count the unit types
  rename(unit_counts = n) %>% # rename just for readability
  inner_join(df %>% count(GEOID10), by = 'GEOID10') %>% # join with total counts (there must be a better way)
  mutate(proportion = unit_counts/n) %>% # calculate estimated proportions
  write_csv('data/estimated_proportions_by_tract.csv') # write to disk