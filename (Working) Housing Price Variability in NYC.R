#5205 Final Project

####Text Mining####

# set working directory 
setwd("/Users/xueyining/Downloads")

# Load necessary packages for text mining, data wrangling, and modeling
library(tidyverse)
library(tidytext)
library(ggplot2)
library(widyr)
library(dplyr)
library(scales)
library(quanteda)
library(tidyr)
library(stringr)
library(e1071)
library(igraph)
library(ggraph)
library(glmnet)  
library(Matrix)  
library(Metrics) 

# Step 1: Load and clean housing price data
price_df <- read_csv("rollingsales_manhattan_clean_file.csv")
head(price_df)
str(price_df)

# Convert both neighborhood and building class to lowercase
# Remove numeric prefixes from building class category
# Combine the two text fields into one string for each observation("NEIGHBORHOOD" and "BUILDING.CLASS.CATEGORY")
# Add a unique document ID for modeling later.

df_clean <- price_df %>%
  select(SALE.PRICE, NEIGHBORHOOD, BUILDING.CLASS.CATEGORY) %>%
  filter(SALE.PRICE > 10000) %>%  
  mutate(
    NEIGHBORHOOD = tolower(NEIGHBORHOOD),
    BUILDING.CLASS.CATEGORY = tolower(BUILDING.CLASS.CATEGORY),
    text = paste(
      NEIGHBORHOOD,
      str_remove(BUILDING.CLASS.CATEGORY, "^\\d+\\s+")
    ),
    doc_id = row_number()
  )


# Step 2: Tokenize text into bigrams (2 words phrases)
# Extract 2-word phrases (like "midtown condo") and count how often they appear in each listing.
tokens_bigram <- df_clean %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "^\\d+$"),
         !str_detect(word2, "^\\d+$")) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(doc_id, bigram, sort = TRUE) %>%
  rename(word = bigram) 

# Tokenize into unigrams and lemmatize
# Extract single words and reduce them to their base form (“apartments” → “apartment”)
tokens_unigram <- df_clean %>%
  unnest_tokens(word, text) %>%
  mutate(word = lemmatize_words(word)) %>%
  filter(!word %in% stop_words$word) %>%
  filter(!str_detect(word, "^\\d+$")) %>%
  count(doc_id, word, sort = TRUE)

# Step 3: Combine unigrams and bigrams into a larger token set

tokens_all <- bind_rows(tokens_unigram, tokens_bigram) %>%
  group_by(word) %>%
  filter(n() >= 5) %>%  #Remove rare words
  ungroup()

tokens_all

# Data visualization
# Check the top 20 most frequent words 
tokens_all %>%
  group_by(word) %>%
  summarise(total_n = sum(n)) %>%
  arrange(desc(total_n)) %>%
  slice_max(total_n, n = 20) %>%
  ggplot(aes(x = reorder(word, total_n), y = total_n)) +
  geom_col(fill = "orchid4") +
  coord_flip() +
  labs(
    title = "Top 20 Most Frequent Words (Unigram + Bigram)",
    x = "Word",
    y = "Total Count"
  ) +
  theme_minimal()

# Step 4: Co-occurrence Network Visualization

# We extract co-occurring word pairs (bigrams), filter out stop words,
# then visualize relationships using igraph and ggraph

# Identify the Top 20 Most Frequent Bigrams (two-word combinations) in the dataset
top_bigrams <- df_clean %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(
    !word1 %in% stop_words$word,
    !word2 %in% stop_words$word,
    !str_detect(word1, "^\\d+$"),
    !str_detect(word2, "^\\d+$")
  ) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE) %>%
  slice_max(n, n = 20)

top_bigrams

#Network Visualization
tokens_bigram_seperate = df_clean %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "^\\d+$"),
         !str_detect(word2, "^\\d+$")) %>%
  count(word1, word2, sort = TRUE)

bigram_graph = tokens_bigram_seperate %>%
  filter(n >= 20) %>%
  graph_from_data_frame()

# Highlight top 5% most frequent edges and nodes in red
threshold <- quantile(tokens_bigram_seperate$n, 0.95)
strong_edges <- tokens_bigram_seperate %>%
  filter(n >= threshold)
strong_nodes <- unique(c(strong_edges$word1, strong_edges$word2))

bigram_graph <- tokens_bigram_seperate %>%
  filter(n >= 20) %>%
  graph_from_data_frame()

set.seed(1038) #set seed to 1038 to ensure the reproductivity
plot = ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n,
                     edge_colour = ifelse(n >= threshold, "strong", "normal")),
                 show.legend = FALSE) +
  scale_edge_colour_manual(values = c("strong" = "red", "normal" = "gray70")) +
  geom_node_point(aes(color = name %in% strong_nodes), size = 4) +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "steelblue")) +
  geom_node_text(
    aes(label = name, color = name %in% strong_nodes),
    size = 6,
    repel = TRUE,
    max.overlaps = Inf
  ) +
  theme_void() +
  labs(title = "Bigram Co-occurrence Network (Top Nodes & Edges in Red)")

print(plot)

# Step 5: Build a Document-Term Matrix (TF-IDF weighted)
# TF-IDF downweights very common terms and highlights more informative ones
tokens_tf_idf <- tokens_all %>%
  bind_tf_idf(term = word, document = doc_id, n = n) %>%
  mutate(
    doc_id = as.character(doc_id),
    word = as.character(word),
    tf_idf = as.numeric(tf_idf)
  )

dtm <- tokens_all %>%
  bind_tf_idf(word, doc_id, n) %>%
  cast_dtm(document = doc_id, term = word, value = tf_idf)

# Step 6 : Log transformation on y
# Raw sale prices are heavily right-skewed. 

X <- as.matrix(dtm)
y <- log(df_clean$SALE.PRICE)

# Check if "SALE.PRICE" is heavily right skewed 
# Visualize price distribution (before log)
ggplot(df_clean, aes(x = SALE.PRICE)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "white") +
  labs(title = "Distribution of SALE.PRICE",
       x = "Sale Price",
       y = "Frequency") +
  theme_minimal()

# Visualize price distribution (after log)
ggplot(df_clean, aes(x = log(SALE.PRICE))) +
  geom_histogram(bins = 50, fill = "tomato", color = "white") +
  labs(title = "Distribution of log(SALE.PRICE)",
       x = "log(Sale Price)",
       y = "Frequency") +
  theme_minimal()

skewness(df_clean$SALE.PRICE, na.rm = TRUE)

# Step 7 : Model Comparison
library(glmnet)  
library(Matrix)  
library(Metrics)  

# Model 1: OLS (Ordinary Least Squares)
X_dense <- as.matrix(X)

ols_model <- lm(y ~ ., data = as.data.frame(X_dense))
ols_pred <- predict(ols_model)
ols_rmse <- rmse(y, ols_pred)
ols_r2 <- 1 - sum((y - ols_pred)^2) / sum((y - mean(y))^2)
ols_rmse
ols_r2

# Model 2: Ridge Regression
set.seed(1038)
ridge_model <- cv.glmnet(X, y, alpha = 0)
ridge_pred <- predict(ridge_model, newx = X, s = "lambda.min")
ridge_rmse <- rmse(y, ridge_pred)
ridge_r2 <- 1 - sum((y - ridge_pred)^2) / sum((y - mean(y))^2)
ridge_rmse 
ridge_r2

# Model 3: Lasso Regression
set.seed(1038)
lasso_model <- cv.glmnet(X, y, alpha = 1) #Lasso selects important text 
lasso_pred <- predict(lasso_model, newx = X, s = "lambda.min")
lasso_rmse <- rmse(y, lasso_pred)
lasso_r2 <- 1 - sum((y - lasso_pred)^2) / sum((y - mean(y))^2)
lasso_rmse
lasso_r2

summary(lasso_model)
plot(lasso_model)

# Step 8: Interpret the model result 
# Extract text features (keywords) and their influence on sale price (from the 
# Lasso model)

coef_df <- coef(lasso_model, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("term") %>%
  filter(term != "(Intercept)") %>%
  rename(weight = s1) %>%
  arrange(desc(abs(weight)))

# Preview the top 10 most important features
head(coef_df, 10)

# Top 10 keywords that increase price
coef_df %>% 
  arrange(desc(weight)) %>%
  slice_head(n = 10)

# Top 10 keywords that decrease price
coef_df %>% 
  arrange(weight) %>%
  slice_head(n = 10)

#  Visualize the top 20 most influential keywords
# Clearly show which words (from the text data) are driving prices up or down
coef_df %>%
  slice_max(abs(weight), n = 20) %>%
  ggplot(aes(x = reorder(term, weight), y = weight, fill = weight > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 20 Influential Text Features (Lasso Coefficients)",
       x = "Term",
       y = "Weight (Effect on log(Sale Price))") +
  scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "firebrick")) +
  theme_minimal()


##### Cluster Analysis #####


install.packages(c("dplyr", "tidyr", "readr", "ggplot2", "purrr", "caret"))

# Load required libraries
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(purrr)

# 1. Read cleaned datasets
green <- read_csv("/Users/lijiameng/Desktop/Greenstreets_clean_file.csv")
sales <- read_csv("/Users/lijiameng/Desktop/rollingsales_manhattan_clean_file.csv")

# 2. Summarize total Greenstreets acreage by ZIP code
green_acres <- green %>%
  group_by(ZIPCODE) %>%
  summarise(sum_acres = sum(as.numeric(ACRES), na.rm = TRUE), .groups = "drop")

# 3. Summarize average sale price by ZIP code
df_sales <- sales %>%
  mutate(ZIP.CODE = as.character(ZIP.CODE),
         SALE.PRICE = as.numeric(SALE.PRICE)) %>%
  group_by(ZIP.CODE) %>%
  summarise(average_price = mean(SALE.PRICE, na.rm = TRUE), .groups = "drop")

# 4. Merge acreage and price summaries
df <- df_sales %>%
  rename(ZIPCODE = ZIP.CODE) %>%
  left_join(green_acres, by = "ZIPCODE") %>%
  replace_na(list(sum_acres = 0))

# 5. Scatterplot with regression line
ggplot(df, aes(x = sum_acres, y = average_price)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(
    x = "Total GreenStreets Acreage per ZIP",
    y = "Average Sale Price",
    title = "Greenstreets Acreage vs. Property Prices"
  ) +
  theme_minimal()

# 6. Pearson correlation and simple linear regression
cor_test <- cor.test(df$sum_acres, df$average_price)
print(cor_test)
model1 <- lm(average_price ~ sum_acres, data = df)
summary(model1)

# 7. Regression by FEATURESTATUS types
acres_by_status <- green %>%
  group_by(ZIPCODE, FEATURESTATUS) %>%
  summarise(acres = sum(as.numeric(ACRES), na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = FEATURESTATUS, values_from = acres, values_fill = 0)

df2 <- df_sales %>%
  rename(ZIPCODE = ZIP.CODE) %>%
  left_join(acres_by_status, by = "ZIPCODE") %>%
  replace_na(list(acres = 0))
model2 <- lm(average_price ~ . - ZIPCODE, data = df2)
summary(model2)

# 8. Cluster Analysis on ZIP-level data
#    Combine price and acreage, drop NAs
df_cluster <- df %>% select(sum_acres, average_price) %>% drop_na()

# 8a. Determine optimal k with Elbow method
set.seed(123)
wss <- map_dbl(1:10, ~kmeans(scale(df_cluster), centers = ., nstart = 25)$tot.withinss)

ggplot(data.frame(k = 1:10, wss = wss), aes(x = k, y = wss)) +
  geom_line() +
  geom_point() +
  labs(x = "Number of Clusters", y = "Within-cluster SS", title = "Elbow Method for K-means") +
  theme_minimal()

# 8b. Apply K-means (k=3 as example)
set.seed(123)
k3 <- kmeans(scale(df_cluster), centers = 3, nstart = 25)

# 9. Attach cluster labels back to main df
df <- df %>%
  mutate(cluster = factor(k3$cluster))

# 10. Visualize clusters
ggplot(df, aes(x = sum_acres, y = average_price, color = cluster)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(
    x = "Total GreenStreets Acreage",
    y = "Average Sale Price",
    color = "Cluster",
    title = "K-means Clusters of ZIP Codes"
  ) +
  theme_minimal()

# 11. Cluster centers and summary statistics
cat("Cluster centers (scaled variables):\n")
print(k3$centers)

df %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    mean_acres = mean(sum_acres),
    mean_price = mean(average_price)
  ) %>%
  print()



# --- 1) Split into train / test ------------------------------------------------

set.seed(123)
split   <- sample(nrow(df), size = 0.7 * nrow(df))
train   <- df[ split, ]
test    <- df[-split, ]

# --- 2) Normalize (remove DV first!) -----------------------------------------

library(caret)
trainMinusDV <- subset(train, select = -c(average_price))
testMinusDV  <- subset(test,  select = -c(average_price))

preproc   <- preProcess(trainMinusDV)
trainNorm <- predict(preproc, trainMinusDV)
testNorm  <- predict(preproc, testMinusDV)

# --- 3) K‑means clustering ----------------------------------------------------

set.seed(123)
km       <- kmeans(trainNorm, centers = 2, nstart = 25)
clusterTrain <- km$cluster
clusterTest  <- predict(km, newdata = testNorm)   # use the same k-means centroids

train$cluster <- clusterTrain
test$cluster  <- clusterTest

# split by cluster membership :contentReference[oaicite:3]{index=3}
train1 <- subset(train, cluster == 1)
train2 <- subset(train, cluster == 2)
test1  <- subset(test,  cluster == 1)
test2  <- subset(test,  cluster == 2)

# --- 4A) Predict with per‑cluster linear models -------------------------------

lm1   <- lm(average_price ~ sum_acres, data = train1)
lm2   <- lm(average_price ~ sum_acres, data = train2)

pred1 <- predict(lm1, newdata = test1)
pred2 <- predict(lm2, newdata = test2)

# combine and compute SSE :contentReference[oaicite:4]{index=4}
predOverall   <- c(pred1,     pred2)
priceOverall  <- c(test1$average_price, test2$average_price)
sseClusters   <- sum((predOverall - priceOverall)^2)

# baseline single‐model SSE
linear    <- lm(average_price ~ sum_acres, data = train)
predLinear<- predict(linear, newdata = test)
sseLinear <- sum((predLinear - test$average_price)^2)

cat("SSE – single model:", sseLinear,    "\n",
    "SSE – clusters   :", sseClusters,   "\n")

# --- 4B) Predict with per‑cluster regression trees ---------------------------

library(rpart)
tree1 <- rpart(average_price ~ sum_acres, data = train1, minbucket = 10)
tree2 <- rpart(average_price ~ sum_acres, data = train2, minbucket = 10)

pred1_t <- predict(tree1, newdata = test1)
pred2_t <- predict(tree2, newdata = test2)

sse1_t  <- sum((test1$average_price - pred1_t)^2)
sse2_t  <- sum((test2$average_price - pred2_t)^2)
sseTrees<- sse1_t + sse2_t                # :contentReference[oaicite:5]{index=5}

cat("SSE – trees:", sseTrees, "\n")

set.seed(1706)
split = createDataPartition(y=rollingsales_manhattan_clean$SALE.PRICE,p = 0.7,list = F,groups = 100)
train = rollingsales_manhattan_clean[split,]
test = rollingsales_manhattan_clean[-split,]
trainMinusDV = subset(train,select=-c(SALE.PRICE))
testMinusDV = subset(test,select=-c(SALE.PRICE))
preproc = preProcess(trainMinusDV)
trainNorm = predict(preproc,trainMinusDV)
testNorm = predict(preproc,testMinusDV)
trainNorm[] <- lapply(trainNorm, function(x) if(is.character(x)) as.factor(x) else x)
distances <- daisy(trainNorm, metric = "gower")
clusters = hclust(d = distances,method = 'ward.D2')
clusters$height
library(dendextend)
plot(color_branches(as.dendrogram(clusters),k = 3,groupLabels = F))
clusterGroups = cutree(clusters,k=4)
library(mclust)
clus = Mclust(trainNorm)
summary(clus)
clus4 = Mclust(trainNorm, G = 4)
bic = sapply(1:9,FUN=function(x){
  -Mclust(trainNorm,G=x)$bic
})
dat = data.frame(clusters=1:9,bic)
ggplot(dat,aes(x=clusters,y=bic))+
  geom_line(color='steelblue',size=1.4)+
  scale_x_continuous(breaks=1:9,minor_breaks = 1:9)
mcluster = Mclust(trainNorm,G=4)


#####Spatial Analysis#####


library(tidyverse)
install.packages("sf")
library(sf)  # for dealing spatial data
install.packages("geosphere")
library(geosphere)  # for calculating distance

setwd("/Users/liuzeyu/Documents")
house <- read_csv("rollingsales_manhattan_clean_file.csv")
parks <- read_csv("Greenstreets_clean_file.csv")

str(house)
str(parks)


install.packages("tidygeocoder")
library(tidygeocoder)
library(dplyr)

#
house <- house %>%
  mutate(full_address = paste(ADDRESS, ZIP.CODE, "New York, NY", sep = ", "))

# 
batch_geocode <- function(df) {
  geocode(df, address = full_address, method = "census")
}

# ）
house <- house %>% mutate(batch = ceiling(row_number() / 500))

# 
library(purrr)
geo_list <- house %>%
  group_split(batch) %>%
  map(batch_geocode)

# Merging results
house_geo_full <- bind_rows(geo_list)

house_sf <- house_geo_full %>%
  filter(!is.na(lat), !is.na(long)) %>%
  st_as_sf(coords = c("long", "lat"), crs = 4326)

nrow(house_geo_full)
summary(house_geo_full[c("lat", "long")])

parks_fixed <- parks %>%
  filter(str_detect(multipolygon, "^MULTIPOLYGON \\(\\(\\(")) %>%  # Matching
  filter(str_detect(multipolygon, "\\)\\)\\)$"))  # 

parks_sf <- st_as_sf(parks_fixed, wkt = "multipolygon", crs = 4326)

nrow(parks_sf)

install.packages("leaflet")
library(leaflet)

leaflet(house_sf) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%  # Background Map
  addCircleMarkers(
    radius = 2,
    color = "blue",
    stroke = FALSE,
    fillOpacity = 0.5,
    popup = ~paste("Price: $", SALE.PRICE)
  ) %>%
  setView(lng = -74.02, lat = 40.70, zoom = 12)  # Manhatten Looks


# Static Graph
library(ggplot2)

ggplot(house_sf) +
  geom_sf(aes(color = log(SALE.PRICE)), size = 0.8, alpha = 0.7) +
  scale_color_viridis_c(option = "C") +
  labs(title = "Distribution Manhantenn Price）",
       color = "log(housing price)") +
  theme_minimal()


library(ggplot2)

leaflet() %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  
  # Graphs
  addPolygons(data = parks_sf,
              fillColor = "forestgreen",
              color = "darkgreen",  # 
              weight = 1,
              fillOpacity = 0.4,
              group = "Greenstreets") %>%
  
  # Blue Points of Housing price
  addCircleMarkers(data = house_sf,
                   radius = 3,
                   color = "blue",
                   stroke = FALSE,
                   fillOpacity = 0.5,
                   popup = ~paste0("Price: $", format(SALE.PRICE, big.mark = ",")),
                   group = "House Sales") %>%
  
  # 
  addLayersControl(
    overlayGroups = c("Greenstreets", "House Sales"),
    options = layersControlOptions(collapsed = FALSE)
  ) %>%
  
  # View: Manhantten Focused
  setView(lng = -73.98, lat = 40.76, zoom = 12)


dist_matrix <- st_distance(house_sf, parks_sf)

sum(!st_is_valid(parks_sf))
parks_sf <- st_make_valid(parks_sf)

nearest_index <- st_nearest_feature(house_sf, parks_sf)

nearest_parks <- parks_sf[nearest_index, ]

house_sf$dist_to_park <- st_distance(house_sf, nearest_parks, by_element = TRUE) %>%
  as.numeric()


library(ggplot2)

ggplot(house_sf, aes(x = dist_to_park, y = SALE.PRICE)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  scale_y_log10() +
  labs(
      title = "Relationship Between Housing Price and Distance to Nearest Greenstreet",
      x = "Distance to Nearest Greenstreet (meters)",
      y = "Sale Price (log scale)"
    )
  theme_minimal()

  
house_sf_filtered <- house_sf %>% filter(SALE.PRICE > 10000)

ggplot(house_sf_filtered, aes(x = dist_to_park, y = SALE.PRICE)) +
  geom_point(alpha = 0.5, color = "orange") +
  scale_y_log10() +
  labs(
    title = "Relationship Between Housing Price and Distance to Nearest Greenstreet",
    x = "Distance to Nearest Greenstreet (meters)",
    y = "Sale Price (log scale)"
  ) +
  theme_minimal()

model <- lm(log(SALE.PRICE) ~ dist_to_park, data = house_sf_filtered)
summary(model)

model2 <- lm(log(SALE.PRICE) ~ dist_to_park + NEIGHBORHOOD + BUILDING.CLASS.CATEGORY, data = house_sf_filtered)
summary(model2)

model_interact <- lm(
  log(SALE.PRICE) ~ dist_to_park * NEIGHBORHOOD + BUILDING.CLASS.CATEGORY,
  data = house_sf_filtered
)

summary(model_interact)


library(broom)
coefs <- tidy(model_interact) %>%
  filter(str_detect(term, "dist_to_park:NEIGHBORHOOD")) %>%
  mutate(
    neighborhood = str_remove(term, "dist_to_park:NEIGHBORHOOD"),
    significance = ifelse(p.value < 0.05, "Significant", "Not Significant")
  )

ggplot(coefs, aes(x = reorder(neighborhood, estimate), y = estimate, fill = significance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Marginal Effect of Greenstreet Distance on Price (by Neighborhood)",
       x = "Neighborhood",
       y = "Effect of Distance to Park (log price scale)") +
  scale_fill_manual(values = c("Significant" = "darkgreen", "Not Significant" = "gray")) +
  theme_minimal()

str(parks_sf)


library(stringr)

parks_sf <- parks_sf %>%
  mutate(
    park_type = case_when(
      str_detect(SITENAME, regex("greenstreet", ignore_case = TRUE)) ~ "Greenstreet",
      str_detect(SITENAME, regex("mall|median", ignore_case = TRUE)) ~ "Mall",
      str_detect(SITENAME, regex("triangle", ignore_case = TRUE)) ~ "Triangle",
      str_detect(SITENAME, regex("plaza", ignore_case = TRUE)) ~ "Plaza",
      str_detect(SITENAME, regex("refuge|sitting", ignore_case = TRUE)) ~ "SittingArea",
      TRUE ~ "Other"
    )
  )

nearest_idx <- st_nearest_feature(house_sf_filtered, parks_sf)

#
house_sf_filtered$nearest_park_type <- parks_sf$park_type[nearest_idx]

library(ggplot2)

ggplot(house_sf_filtered, aes(x = nearest_park_type, y = SALE.PRICE)) +
  geom_boxplot(fill = "forestgreen", alpha = 0.6) +
  scale_y_log10() +
  labs(
    title = "Distribution of Housing Prices by Nearest Greenstreet Type",
    x = "Nearest Greenstreet Type",
    y = "Sale Price (log scale)"
  ) +
  theme_minimal()

model_type <- lm(
  log(SALE.PRICE) ~ nearest_park_type + NEIGHBORHOOD + BUILDING.CLASS.CATEGORY,
  data = house_sf_filtered
)

summary(model_type)


house_sf_filtered$nearest_park_type <- factor(house_sf_filtered$nearest_park_type)

house_sf_filtered$nearest_park_type <- relevel(house_sf_filtered$nearest_park_type, ref = "Mall")

model_alt <- lm(
  log(SALE.PRICE) ~ nearest_park_type + NEIGHBORHOOD + BUILDING.CLASS.CATEGORY,
  data = house_sf_filtered
)

summary(model_alt)

library(mgcv)

gam_model <- gam(log(SALE.PRICE) ~ s(dist_to_park) + NEIGHBORHOOD + BUILDING.CLASS.CATEGORY,
                 data = house_sf_filtered)
summary(gam_model)
plot(gam_model, se = TRUE, rug = TRUE, shade = TRUE,
     ylab = "Effect on log(SALE.PRICE)", xlab = "Distance to Park (m)")

