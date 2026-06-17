# ==============================================================================
# 0. LIBRARIES & SETUP
# ==============================================================================
# Install missing packages if necessary:
# install.packages(c("dplyr", "tidyr", "ggplot2", "reshape2", "bestNormalize", 
#                    "vegan", "factoextra", "caret", "randomForest", "psych", 
#                    "pROC", "MLmetrics", "cluster", "mclust"))

library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(bestNormalize)
library(vegan)
library(factoextra)
library(caret)
library(randomForest)
library(psych)
library(pROC)
library(MLmetrics)
library(cluster)
library(mclust)

# Set global seed for reproducibility
set.seed(467)

# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================
data <- read.csv("dataset.csv") 

# --- Check for Missing and Duplicate Values ---
print("Missing values per column:")
print(colSums(is.na(data)))
print("Duplicate values per column:")
print(sapply(data, function(x) sum(duplicated(x))))

# --- Remove Duplicates ---
# Removing duplicates based on track_id
data <- data[!duplicated(data$track_id), ]

# --- Duration Filtering (1st - 99th Percentile) ---
lower <- quantile(data$duration_ms, 0.01, na.rm = TRUE)
upper <- quantile(data$duration_ms, 0.99, na.rm = TRUE)
data <- data[data$duration_ms >= lower & data$duration_ms <= upper, ]

# --- Popularity Transformation (Threshold: 50) ---
# Converting popularity into a binary target (0 vs 1)
colnames(data)[colnames(data) == "popularity"] <- "popularity_column"
data <- data %>%
  mutate(popularity = ifelse(popularity_column <= 50, 0, 1)) %>%
  select(-popularity_column)

# --- Sampling ---
# Subsetting 10,000 rows for computational efficiency
data <- data[sample(nrow(data), 10000), ]
print("Structure of sampled data:")
str(data)

# ==============================================================================
# 2. EDA: VISUALIZING DISTRIBUTIONS
# ==============================================================================
numerical_cols <- c("duration_ms", "danceability", "energy", "loudness", 
                    "tempo", "speechiness", "acousticness", "instrumentalness", 
                    "liveness", "valence")

# --- 2.1 Histograms (Numerical Variables) ---
data_long <- data %>%
  select(all_of(numerical_cols)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

p_hist <- ggplot(data_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = "Distributions of Numerical Variables", x = "", y = "Frequency")

print(p_hist)
# CHECK: Ensure directory exists (it is created later in script, so let's create it here too just in case)
dir.create("plots_2", showWarnings = FALSE)
ggsave("plots_2/0_eda_hist.png", p_hist, width = 10, height = 8)

# --- 2.2 Bar Plots (Categorical Variables) ---
categorical_cols <- c("explicit", "key", "mode", "time_signature", "track_genre", "popularity")
data_cat_long <- data %>%
  mutate(across(all_of(categorical_cols), as.character)) %>%
  select(all_of(categorical_cols)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

# Plotting all categorical variables except 'track_genre' (too many levels)
ggplot(data_cat_long %>% filter(variable != "track_genre"), aes(x = value)) +
  geom_bar(fill = "darkblue", color = "black") +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  theme_minimal() +
  labs(title = "Distributions of Categorical Variables", x = "", y = "Count")

# ==============================================================================
# 3. EDA: CORRELATION ANALYSIS
# ==============================================================================
numerical_data <- data[, numerical_cols]
correlation_matrix <- cor(numerical_data, use = "pairwise.complete.obs")

# Preparing data for Heatmap
melted_correlation <- melt(correlation_matrix, varnames = c("Var1","Var2"), value.name = "corr")
melted_correlation$Var1 <- factor(melted_correlation$Var1, levels = numerical_cols)
melted_correlation$Var2 <- factor(melted_correlation$Var2, levels = numerical_cols)
# Subset to show upper triangle only
melted_correlation <- subset(melted_correlation, as.integer(Var1) <= as.integer(Var2))

p_corr <- ggplot(melted_correlation, aes(x = Var1, y = Var2, fill = corr)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", corr), color = abs(corr) > 0.6), size = 3.2) +
  scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation") +
  scale_color_manual(values = c("black", "white"), guide = "none") +
  coord_fixed() +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), 
        axis.title = element_blank()) +
  labs(title = "Correlation Heatmap (Upper Triangle)")

print(p_corr)
ggsave("plots_2/0b_eda_corr.png", p_corr, width = 8, height = 8)

# ==============================================================================
# 4. INITIAL PCA PROJECTION (Pre-Cleaning)
# ==============================================================================
pca_res <- prcomp(data[, numerical_cols], scale. = TRUE)
pca_df <- as.data.frame(pca_res$x)

# Visualize based on Popularity Status
if(length(unique(data$popularity)) <= 2) { 
  pca_df$pop_status <- ifelse(data$popularity == 1, "1: Popular", "0: Not Popular") 
} else {
  threshold <- median(data$popularity, na.rm = TRUE) 
  pca_df$pop_status <- ifelse(data$popularity > threshold, "1: Popular", "0: Not Popular")
}
pca_df$pop_status <- factor(pca_df$pop_status, levels = c("1: Popular", "0: Not Popular"))

ggplot(pca_df, aes(x = PC1, y = PC2, color = pop_status)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = c("1: Popular" = "red", "0: Not Popular" = "grey70")) + 
  theme_minimal() +
  labs(title = "PCA Projection (Before Outlier Removal)", color = "Status",
       x = paste0("PC1 (", round(100 * summary(pca_res)$importance[2,1], 1), "%)"),
       y = paste0("PC2 (", round(100 * summary(pca_res)$importance[2,2], 1), "%)")) -> p_pca_init

print(p_pca_init)
ggsave("plots_2/0e_pca_initial.png", p_pca_init, width = 8, height = 6)

# ==============================================================================
# 5. NORMALIZATION & MAHALANOBIS (OUTLIER DETECTION)
# ==============================================================================
# --- Applying bestNormalize ---
data_transformed <- data %>%
  select(all_of(numerical_cols)) %>%
  mutate(across(everything(), ~ {
    obj <- bestNormalize(., allow_orderNorm = TRUE)
    obj$x.t
  }))

# --- Mahalanobis Distance Calculation (Pre-Clean) ---
data_clean_for_m <- data_transformed %>% drop_na() %>% as.matrix()
m_dist <- mahalanobis(data_clean_for_m, colMeans(data_clean_for_m), cov(data_clean_for_m))

# --- Multivariate Q-Q Plot (Blue Points) ---
n <- length(m_dist)
p <- ncol(data_clean_for_m)
df_qq_m <- data.frame(observed = sort(m_dist), theoretical = qchisq(ppoints(n), df = p))

p_qq1 <- ggplot(df_qq_m, aes(x = theoretical, y = observed)) +
  geom_point(alpha = 0.5, color = "darkblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "Multivariate Q-Q Plot (Before Outlier Removal)", 
       subtitle = "Checking Mahalanobis Distance")

print(p_qq1)
ggsave("plots_2/0c_qq_initial.png", p_qq1, width = 6, height = 6)

# --- Individual Q-Q Plots ---
data_transformed %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(sample = value)) +
  stat_qq(alpha = 0.4, size = 0.8) +
  stat_qq_line(color = "red") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = "Individual Q-Q Plots (After Best Normalization)")

# ==============================================================================
# 6. OUTLIER REMOVAL & FEATURE ENGINEERING
# ==============================================================================
# Define cutoff (Chi-square distribution, p=0.001)
cutoff <- qchisq(1 - 0.001, df = ncol(data_clean_for_m))
is_outlier <- m_dist > cutoff
data_trimmed <- data[!is_outlier, ]

# Feature Engineering: Convert Instrumentalness to Binary
data_trimmed <- data_trimmed %>%
  mutate(instrumental_bin = ifelse(instrumentalness > 0.5, 1, 0))

# ==============================================================================
# 7. FINAL NORMALIZATION & VERIFICATION
# ==============================================================================
# Updated numerical column list (replaced instrumentalness with binary version)
numerical_cols_new <- c("duration_ms", "danceability", "energy", "loudness", "tempo", 
                        "speechiness", "acousticness", "liveness", "valence", "instrumental_bin")

# Re-Normalize on the Trimmed Data
data_final_transformed <- data_trimmed %>%
  select(all_of(numerical_cols_new)) %>%
  mutate(across(everything(), ~ {
    obj <- bestNormalize(., allow_orderNorm = TRUE)
    obj$x.t
  }))

# Final Mahalanobis Check
data_matrix_final <- as.matrix(data_final_transformed)
m_dist_final <- mahalanobis(data_matrix_final, colMeans(data_matrix_final), cov(data_matrix_final))

# --- Final Multivariate Q-Q Plot (Green Points) ---
n_final <- length(m_dist_final)
p_final <- ncol(data_matrix_final)
df_qq_final <- data.frame(observed = sort(m_dist_final), theoretical = qchisq(ppoints(n_final), df = p_final))

p_qq2 <- ggplot(df_qq_final, aes(x = theoretical, y = observed)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "Multivariate Q-Q Plot (Final Cleaned Data)", 
       subtitle = "After Outlier Trimming & Binary Transformation")

print(p_qq2)
ggsave("plots_2/0d_qq_final.png", p_qq2, width = 6, height = 6)

# --- Merge Transformed Data Back ---
data_final <- data_trimmed %>%
  select(-all_of(numerical_cols)) %>% # Remove old numerical columns
  bind_cols(data_final_transformed)   # Add new normalized columns

# Update global data variable to point to cleaned data
data <- data_final
print("Final Data Structure:")
str(data)

# ==============================================================================
# 8. PERMANOVA ANALYSIS
# ==============================================================================
# Clean column names (fix potential issues from bind_cols)
data <- data %>%
  rename(instrumental_bin = contains("instrumental_bin") %>% first())

final_vars <- c("duration_ms", "danceability", "energy", "loudness", "tempo", 
                "speechiness", "acousticness", "liveness", "valence", "instrumental_bin")

# Subsampling for PERMANOVA (Computationally expensive)
set.seed(467)
n_sample <- min(2000, nrow(data))
idx <- sample(seq_len(nrow(data)), n_sample)
data_sample <- data[idx, ]

X_sample <- data_sample %>% select(all_of(final_vars))
group_sample <- as.factor(data_sample$popularity)

# Distance Matrix (Euclidean)
dist_mat <- dist(X_sample, method = "euclidean")

# Run PERMANOVA
print("Calculating PERMANOVA, please wait...")
set.seed(467)
permanova_pop <- adonis2(dist_mat ~ group_sample, permutations = 199)

print("--- PERMANOVA RESULTS (BY POPULARITY) ---")
print(permanova_pop)

# ==============================================================================
# 9. DETAILED PCA ANALYSIS
# ==============================================================================
# Creaate plots_2 directory if it doesn't exist
dir.create("plots_2", showWarnings = FALSE)

pca_cols <- c("duration_ms", "danceability", "energy", "loudness", 
              "speechiness", "acousticness", "instrumental_bin", "liveness", 
              "valence", "tempo")

pca_data_full <- data %>% select(all_of(pca_cols))

# Run PCA (Scale = TRUE)
pca_final <- prcomp(pca_data_full, scale. = TRUE)

cat("### PCA Variance Summary ###\n")
summary(pca_final)

# --- Scree Plot ---
p_scree <- fviz_eig(pca_final, addlabels = TRUE, ylim = c(0, 40), 
         main = "Scree Plot: Proportion of Variance Explained")
print(p_scree)
ggsave("plots_2/1_pca_scree.png", p_scree, width = 8, height = 6)

# --- PCA Biplot ---
scores <- as.data.frame(pca_final$x)
scores$popularity <- as.factor(data$popularity)

loadings <- as.data.frame(pca_final$rotation)
loadings$variable <- rownames(loadings)

mult <- 5 # Arrow scaling factor

p_biplot <- ggplot() +
  geom_point(data = scores, aes(x = PC1, y = PC2, color = popularity), 
             alpha = 0.2, size = 1) +
  geom_segment(data = loadings, 
               aes(x = 0, y = 0, xend = PC1 * mult, yend = PC2 * mult),
               arrow = arrow(length = unit(0.2, "cm")), 
               color = "#D7191C", linewidth = 0.8) +
  geom_text(data = loadings, 
            aes(x = PC1 * mult * 1.1, y = PC2 * mult * 1.1, label = variable),
            color = "black", fontface = "bold", size = 3.5) +
  scale_color_manual(values = c("0" = "green", "1" = "brown"),
                     labels = c("Not Popular", "Popular")) +
  theme_minimal() +
  labs(title = "PCA Biplot: Samples and Variable Loadings",
       x = paste0("PC1 (", round(100 * summary(pca_final)$importance[2,1], 1), "%)"),
       y = paste0("PC2 (", round(100 * summary(pca_final)$importance[2,2], 1), "%)"))

print(p_biplot)
ggsave("plots_2/2_pca_biplot.png", p_biplot, width = 8, height = 6)

# --- Loadings Heatmap (First 7 PCs) ---
n_pcs_selected <- 7
melted_loadings_7 <- pca_final$rotation[, 1:n_pcs_selected] %>%
  as.data.frame() %>%
  mutate(Variable = rownames(.)) %>%
  pivot_longer(cols = -Variable, names_to = "PC", values_to = "Loading")

p_pca_heat <- ggplot(melted_loadings_7, aes(x = PC, y = Variable, fill = Loading)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Loading, 2)), size = 3) +
  scale_fill_gradient2(low = "#E46726", mid = "white", high = "#00468B", 
                       midpoint = 0, limits = c(-1, 1), name = "Loading") +
  theme_minimal() +
  labs(title = paste("PCA Loadings Heatmap (First", n_pcs_selected, "PCs)"),
       subtitle = "Representing 80%+ of total cumulative variance",
       x = "Principal Components", y = "Original Audio Features")

print(p_pca_heat)
ggsave("plots_2/3_pca_loadings_heatmap.png", p_pca_heat, width = 8, height = 6)

# --- Variable Contributions (PC1) ---
p_contrib <- fviz_contrib(pca_final, choice = "var", axes = 1, fill = "steelblue", color = "black")
print(p_contrib)
ggsave("plots_2/4_pca_contrib_pc1.png", p_contrib, width = 8, height = 6)

# ==============================================================================
# 10. FACTOR ANALYSIS (EXPLORATORY)
# ==============================================================================
# Note: Generating the factor analysis model first
fa_data <- data %>% select(all_of(pca_cols))
# Using 4 factors as an example, with Varimax rotation
fa_result <- fa(fa_data, nfactors = 4, rotate = "varimax")

loadings_df <- as.data.frame(unclass(fa_result$loadings))
loadings_df$Variable <- rownames(loadings_df)

loadings_long <- loadings_df %>%
  pivot_longer(cols = -Variable, names_to = "Factor", values_to = "Loading")

p_fa_heat <- ggplot(loadings_long, aes(x = Factor, y = Variable, fill = Loading)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(Loading, 2), 
                alpha = abs(Loading) > 0.3), 
            size = 4, fontface = "bold") +
  scale_fill_gradient2(low = "#D7191C", mid = "white", high = "#2C7BB6", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation") +
  scale_alpha_manual(values = c(0.3, 1), guide = "none") +
  theme_minimal(base_size = 14) +
  labs(title = "Factor Loadings Heatmap",
       subtitle = "Variable associations with Latent Factors",
       x = "Latent Factors", y = "Musical Features") +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(face = "bold"))

print(p_fa_heat)
ggsave("plots_2/5_factor_analysis_heatmap.png", p_fa_heat, width = 8, height = 6)

# ==============================================================================
# 11. DISCRIMINATION AND CLASSIFICATION (LDA & RF)
# ==============================================================================
# We will use both parametric (LDA) and non-parametric (Random Forest) methods.
# Crucially, we will handle CLASS IMBALANCE by down-sampling the majority class
# in the training set. This discriminates based on features, not prior probabilities.

# --- 11.1 Data Preparation (Balanced) ---
class_cols <- c(pca_cols, "popularity")
discrim_data <- data %>% select(all_of(class_cols))
discrim_data$popularity <- factor(discrim_data$popularity, levels = c(0, 1), 
                                  labels = c("NotPopular", "Popular"))

set.seed(467)
# Split into Train (80%) and Test (20%) - Stratified
train_idx <- createDataPartition(discrim_data$popularity, p = 0.8, list = FALSE)
train_raw <- discrim_data[train_idx, ]
test_data  <- discrim_data[-train_idx, ]

# Downsample Training Data (50-50 Balance)
set.seed(467)
train_balanced <- downSample(x = train_raw[, -ncol(train_raw)],
                             y = train_raw$popularity)
colnames(train_balanced)[ncol(train_balanced)] <- "popularity" # Rename 'Class' to 'popularity'

print("--- Balanced Training Set Counts ---")
print(table(train_balanced$popularity))

# --- 11.2 Linear Discriminant Analysis (LDA) ---
library(MASS)
model_lda <- lda(popularity ~ ., data = train_balanced)

# Predict on Test Set (Unbalanced - Real World Scenario)
pred_lda_raw <- predict(model_lda, test_data)
pred_lda_class <- pred_lda_raw$class

print("--- LDA CONFUSION MATRIX (Test Set) ---")
cm_lda <- confusionMatrix(pred_lda_class, test_data$popularity, positive = "Popular")
print(cm_lda)

# --- 11.3 Random Forest (Balanced Learning) ---
print("Training Balanced Random Forest...")
set.seed(467)
model_rf_bal <- randomForest(
  popularity ~ ., 
  data = train_balanced, 
  ntree = 500,
  importance = TRUE
)

# Predict on Test Set
pred_rf <- predict(model_rf_bal, newdata = test_data)

print("--- RANDOM FOREST CONFUSION MATRIX (Test Set) ---")
cm_rf <- confusionMatrix(pred_rf, test_data$popularity, positive = "Popular")
print(cm_rf)

# Comparison Metrics
acc_lda <- cm_lda$overall["Accuracy"]
kap_lda <- cm_lda$overall["Kappa"]
sens_lda <- cm_lda$byClass["Sensitivity"] # Recall for Popular
spec_lda <- cm_lda$byClass["Specificity"] 
acc_rf <- cm_rf$overall["Accuracy"]
kap_rf <- cm_rf$overall["Kappa"]
sens_rf <- cm_rf$byClass["Sensitivity"]

print(paste("LDA Sensitivity (Popular):", round(sens_lda, 4)))
print(paste("RF Sensitivity (Popular):", round(sens_rf, 4)))

# --- 11.4 Feature Importance Plot (RF) ---
imp_df <- as.data.frame(importance(model_rf_bal))
imp_df$Variable <- rownames(imp_df)

# We use MeanDecreaseGini implies contribution to node purity
p_imp <- ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "darkred", width = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance (Random Forest)",
       subtitle = "Variable contribution to separating Popular/Not Popular",
       x = "Audio Features", y = "Mean Decrease Gini")

print(p_imp)
ggsave("plots_2/11_rf_importance.png", p_imp, width = 8, height = 6)


# ==============================================================================
# 13. CLUSTERING (SMALL SAMPLE ANALYSIS) - FIXED
# ==============================================================================
# HATA ????Z??M??: Target ve Feature'lar??n sat??r say??s??n??n e??it olmas?? garanti alt??na al??nd??.

set.seed(467)
# 1. ??rneklem Al (100 sat??r)
mini_idx <- sample(nrow(data), 100)
mini_data_raw <- data[mini_idx, ]

# 2. Sadece Clustering i??in gerekli kolonlar?? ve Popularity'i se??
features_clust <- c("duration_ms", "danceability", "energy", "loudness", 
                    "speechiness", "acousticness", "liveness", "valence", "tempo", 
                    "instrumental_bin")

# Ge??ici bir veri seti olu??tur (Features + Target)
temp_df <- mini_data_raw %>% 
  dplyr::select(all_of(features_clust), popularity) %>%
  na.omit() # E??er NA varsa hem feature hem target'tan ayn?? anda siler

# 3. Target ve Feature'lar?? AYNI temp_df ??zerinden ay??r (B??ylece uzunluklar e??it olur)
mini_target <- ifelse(temp_df$popularity == 1, "Popular", "NotPopular")
mini_target <- factor(mini_target, levels = c("Popular", "NotPopular"))

# Sadece Feature'lar?? al ve Scale et
df_mini_scaled <- temp_df %>% 
  dplyr::select(-popularity) %>% 
  scale()

# --- Hierarchical Clustering (HAC) ---
dist_mat_mini <- dist(df_mini_scaled, method = "euclidean")
hc_res <- hclust(dist_mat_mini, method = "ward.D2")

p_dend <- fviz_dend(hc_res, k = 2, cex = 0.6, k_colors = c("#2E9FDF", "#E7B800"),
          rect = TRUE, main = "Dendrogram (Sample of 100 Tracks)")
print(p_dend)
ggsave("plots_2/6_cluster_dendrogram.png", p_dend, width = 10, height = 6)

# --- K-Means Clustering ---
set.seed(123)
km_res <- kmeans(df_mini_scaled, centers = 2, nstart = 25)

p_kmeans <- fviz_cluster(km_res, data = df_mini_scaled,
             geom = "point", ellipse.type = "convex", 
             main = "K-Means Results (k = 2)")
print(p_kmeans)
ggsave("plots_2/7_cluster_kmeans.png", p_kmeans, width = 8, height = 6)

# --- Clustering Performance (ARI & Silhouette) ---
grp_hc <- cutree(hc_res, k = 2)

# Adjusted Rand Index (Art??k hata vermez ????nk?? uzunluklar e??itlendi)
ari_km <- adjustedRandIndex(km_res$cluster, mini_target)
ari_hc <- adjustedRandIndex(grp_hc, mini_target)

# Silhouette Score
sil_km <- silhouette(km_res$cluster, dist_mat_mini)
sil_hc <- silhouette(grp_hc, dist_mat_mini)

results_clust <- data.frame(
  Method = c("K-Means (100 sample)", "HAC (100 sample)"),
  Silhouette_Score = c(mean(sil_km[, 3]), mean(sil_hc[, 3])),
  ARI_vs_Popularity = c(ari_km, ari_hc)
)

print("--- CLUSTERING REPORT (N=100) ---")
print(results_clust)

# ==============================================================================
# 14. CANONICAL CORRELATION ANALYSIS (CCA)
# ==============================================================================
# Hypothesis: Physical Intensity (X) vs Emotional Vibe (Y)
X_vars <- c("energy", "loudness", "tempo")
Y_vars <- c("danceability", "valence", "acousticness")

data_cca <- data %>% dplyr::select(all_of(c(X_vars, Y_vars))) %>% na.omit()
X <- as.matrix(data_cca[, X_vars])
Y <- as.matrix(data_cca[, Y_vars])

cca_res <- cancor(X, Y)

print("--- Canonical Correlations (Strength of Relationship) ---")
print(cca_res$cor)

# Visualize the 1st Dimension
CC1_X <- X %*% cca_res$xcoef[, 1]
CC1_Y <- Y %*% cca_res$ycoef[, 1]

cca_df <- data.frame(Intensity = CC1_X, Vibe = CC1_Y)

p_cca <- ggplot(cca_df, aes(x = Intensity, y = Vibe)) +
  geom_point(alpha = 0.1, color = "purple") +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  theme_minimal() +
  labs(title = "Canonical Correlation Analysis",
       subtitle = paste("Max Correlation:", round(cca_res$cor[1], 3)),
       x = "Set 1: Intensity (Energy/Loudness/Tempo)",
       y = "Set 2: Vibe (Dance/Valence/Acoustic)")

print(p_cca)
ggsave("plots_2/8_cca_plot.png", p_cca, width = 8, height = 6)

