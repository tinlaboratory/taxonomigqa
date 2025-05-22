library(tidyverse)

gqa <- read_csv("data/gqa_entities/gqa-lemmas.csv")
gqa_cats <- read_csv('data/gqa_entities/category-membership.csv')

things <- read_csv("data/things-taxonomic-sensitivity/things-lemmas-annotated.csv")


# gqa %>% 
#   left_join(things) %>%
#   write_csv("data/gqa_entities/gqa-lemmas-preannotated.csv")
