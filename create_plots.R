library(ggplot2)
library(dplyr)
library(stringr)

feature_importance <- read.csv('feature_importance.csv')
genre_results <- read.csv('genre_results.csv')

feature_importance %>%
  mutate(var_group = case_when(
    str_detect(var, "review_topic\\d+") ~ "review",
    str_detect(var, "synopsis_topic\\d+") ~ "synopsis",
    str_detect(var, "^[A-Z]") ~ "genre",
    TRUE ~ var
  )) %>%
  mutate(data_source = case_when(
    str_detect(var, "synopsis|box_office|runtime") ~ "movie",
    var_group == "genre" ~ "movie",
    TRUE ~ "review"
  )) ->
  feature_importance

feature_importance %>%
  group_by(var_group) %>%
  summarize(avg_importance = mean(importance)) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(var_group, avg_importance), y = avg_importance, fill = var_group)) +
  geom_col() +
  xlab("Variable Group") +
  ylab("Average Importance") +
  scale_fill_discrete(name = "Variable Group") +
  coord_flip()
ggsave("avgimportance_by_vargroup.png")

feature_importance %>%
  ggplot(aes(x = importance, fill = data_source)) +
  geom_histogram(bins = 60, position = "identity", alpha = 0.6) +
  scale_fill_discrete(name = "Data Source") +
  ggtitle("Feature Importance by Data Source")
ggsave("importance_by_source.png")

feature_importance %>%
  filter(var_group == "genre" | var_group == "review" | var_group == "synopsis") %>%
  ggplot(aes(x = importance, fill = var_group)) +
  geom_histogram(bins = 60, position = "identity", alpha = 0.6) +
  scale_fill_discrete(name = "Variable Group") +
  ggtitle("Feature Importance by Variable Group")
ggsave("importance_by_vargroup.png")

genre_results %>%
  ggplot(aes(x = count, y = accuracy, color = accuracy)) +
  geom_point(size = 3) +
  scale_x_log10() +
  scale_color_gradientn(colors = c("red", "yellow", "green")) +
  ggtitle("Accuracy by Genre Count")
ggsave("accuracy_by_count.png")

genre_results %>%
  ggplot(aes(x = count, y = f1, color = f1)) +
  geom_point(size = 3) +
  scale_x_log10() +
  scale_color_gradientn(colors = c("red", "yellow", "green")) +
  ggtitle("F1-Score by Genre Count")
ggsave("f1_by_count.png")
