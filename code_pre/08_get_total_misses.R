suppressPackageStartupMessages({
  library(ANTsR)
  library(ANTsRCore) # for cluster labeling
  library(extrantsr) # for bias correction, skull-stripping, and registration
  library(tidyverse)
  library(here)
})

if (Sys.getenv("LSB_JOBINDEX") == "") {
  i <- 56
} else {
  i <- as.numeric(Sys.getenv("LSB_JOBINDEX"))
}

data_dir <- file.path(here(), "data/processed_05")

get_n_missing <- function(lesion_labels, coords_image) {
  coords_image_dil <- iMath(coords_image, "GD", 2)
  coords_image_clusters <- labelClusters(coords_image_dil)
  n_coords <- max(coords_image_clusters)
  n_missing <- n_coords - sum(unique(coords_image_clusters[lesion_labels != 0]) != 0)
  print(c(n_coords, n_missing))
  return(c(n_coords, n_missing))
}

count_missing_lesions <- function(i, data_dir) {
  print(i)
  subject <- list.files(data_dir)
  subject_files <- list.files(file.path(data_dir, subject[i]),
                              full.names = TRUE)
  
  cvs_path <- subject_files[grepl("cvs_coords", subject_files)]
  if (length(cvs_path) == 0) {
    contains_cvs <- 0
    contains_lesions <- 0
  } else {
    contains_cvs <- 1
    if (grepl("cvs_coords_nl", cvs_path)) {
      contains_lesions <- 0
    } else {
      contains_lesions <- 1
    }
  }
  
  prl_path <- subject_files[grepl("prl_coords.nii.gz", subject_files)]
  
  prl_coords <- check_ants(prl_path)
  lesion_labels <- check_ants(file.path(data_dir, subject[i], "prob_05.nii.gz"))
  
  if (contains_cvs) {
    cvs_coords <- check_ants(cvs_path)
    cvs_missing <- get_n_missing(lesion_labels, cvs_coords == 3)

    lesion_missing <- get_n_missing(lesion_labels,  cvs_coords != 0 | prl_coords != 0)
  } else {
    lesion_missing <- get_n_missing(lesion_labels, prl_coords != 0)
    cvs_missing <- c(NA, NA)
  }
  
  prl_missing <- get_n_missing(lesion_labels, prl_coords == 2)
  
  return(cbind(subject[i], 
               as.data.frame(matrix(c(contains_lesions, contains_cvs, lesion_missing, prl_missing, cvs_missing),
                                    nrow = 1, ncol = 8))))
}

missing_lesions <- as.data.frame(matrix(rep(0, 9 * 97), ncol = 9, nrow = 97))
for (i in 1:97) {
  missing_lesions[i, ] <- count_missing_lesions(i, data_dir)
}
missing_lesions <- missing_lesions %>% 
  rename(contains_lesions = V2, contains_cvs = V3, 
         total_lesions = V4, missing_lesions = V5, 
         total_prls = V6, missing_prls = V7, 
         total_cvss = V8, missing_cvss = V9)
write.csv(missing_lesions, here("data/csvs/missing_lesions.csv"), row.names = FALSE)

ms_status <- read.csv(here("data/csvs/cavs_ms_status.csv")) %>% 
  mutate(site = as.numeric(str_sub(subject, 2, 2)))

missing_lesions <- missing_lesions %>% 
  rename(subject = V1) %>% 
  left_join(ms_status, by = "subject")

missing_lesions %>% 
  group_by(ms_diagnosis) %>% 
  summarize(sum_lesions = sum(V5, na.rm = TRUE),
            sum_prls = sum(V7, na.rm = TRUE),
            sum_cvss = sum(V9, na.rm = TRUE),
            mean_lesions = mean(V5, na.rm = TRUE),
            mean_prls = mean(V7, na.rm = TRUE),
            mean_cvss = mean(V9, na.rm = TRUE))

missing_sum <- colSums(missing_lesions[, 2:9], na.rm = TRUE); missing_sum
les_covered <- missing_sum[3] - missing_sum[4]; les_covered
les_covered / missing_sum[3]
prl_covered <- missing_sum[5] - missing_sum[6]; prl_covered
prl_covered / missing_sum[5]
cvs_covered <- missing_sum[7] - missing_sum[8]; cvs_covered
cvs_covered / missing_sum[7]

n_candidates <- rep(0, 97)
for (i in 1:97) { 
  subject <- list.files(data_dir)
  lesion_labels <- check_ants(file.path(data_dir, subject[i], "prob_05.nii.gz"))
  n_candidates[i] <- length(unique(lesion_labels)) - 1
}
sum(n_candidates)
