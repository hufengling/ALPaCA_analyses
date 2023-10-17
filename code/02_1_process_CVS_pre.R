library(readxl)
library(tidyverse)
library(parallel)
library(ANTsR)
library(ANTsRCore)
library(extrantsr)
library(here)

#####
process_sheet <- function(path, sheet_num, keep_all = F) {
  df <- read_xlsx(path, sheet_num)[, 1:3]
  names(df) <- c("coords", "location", "cvs_status")
  
  if (keep_all) {
    df <- df %>% 
      filter(cvs_status %in% c("Positive", "Negative", "Excluded"))
  } else {
    df <- df %>% 
      filter(cvs_status == "Positive")
  }
  df <- df %>% 
    filter(!is.na(coords)) %>% 
    mutate(cvs_id = case_when(
      cvs_status == "Negative" ~ 2,
      cvs_status == "Excluded" ~ 3,
      cvs_status == "Positive" ~ 1))
  coords_df <- df$coords %>% 
    str_replace_all(",", " ") %>% 
    str_squish() %>%
    str_split_fixed(" ", 3) %>% 
    as.data.frame() %>% 
    rename(x_coord = "V1",
           y_coord = "V2",
           z_coord = "V3") %>% 
    mutate_if(is.character,as.numeric)
  
  return(cbind(df, coords_df))
}

combine_dataframes <- function(df_list) {
  unique_names <- unique(names(df_list))
  combined_dfs <- list()
  
  for (name in unique_names) {
    df_subset <- df_list[names(df_list) == name]
    combined_dfs[[name]] <- do.call(rbind, df_subset)
  }
  
  return(combined_dfs)
}

label_seg_nifti <- function(excel_list, index, 
                            seg_1_dir, seg_2_dir, flair_dir, combined_dir) {
  print(index)
  subject_id <- names(excel_list)[index]
  subject_id <- str_split_fixed(subject_id, " ", 2)[1]
  dir.create(file.path(combined_dir, subject_id), 
             showWarnings = FALSE, recursive = TRUE)
  coord_df <- excel_list[[index]]
  seg_1_path <- file.path(seg_1_dir, subject_id)
  seg_2_path <- file.path(seg_2_dir, subject_id)
  flair_path <- file.path(flair_dir, subject_id, 
                          "N4/FLAIR_aligned_POST_N4.nii.gz")
  seg_1 <- NULL
  seg_2 <- NULL
  
  flair <- check_ants(flair_path)
  
  if (file.exists(seg_1_path)) {
    print("Lynn exists")
    seg_1 <- check_ants(list.files(seg_1_path, 
                                   full.names = T))
    seg_1 <- seg_1 * 2 # Lynn's segmentations are 0/1, but should default to 2 = non-CVS
    seg_1_lab <- labelClusters(seg_1, maxThresh = 10, minClusterSize = 1)
    if (mean(table(seg_1_lab[seg_1_lab != 0])) < 3) { # If average coordinate marker is small, dilate by 1
      seg_1 <- iMath(seg_1, "GD", 1)
      seg_1_lab <- iMath(seg_1_lab, "GD", 1)
    }
  }
  
  if (file.exists(seg_2_path)) {
    print("Carly exists")
    seg_2_path <- list.files(file.path(seg_2_path, "Results"), 
                             full.names = T)
    seg_2 <- check_ants(seg_2_path[grepl(".nii", seg_2_path)][1])
    seg_2_lab <- labelClusters(seg_2, maxThresh = 10, minClusterSize = 1) # All of Carly's segmentations are at least 3 diameter
  }
  
  cvs_image <- antsImageClone(flair)
  cvs_image[cvs_image != 0] <- 0
  cvs_image_lab <- antsImageClone(cvs_image)
  
  if (!is.null(seg_2)) {
    cvs_image <- cvs_image + seg_2
    cvs_image_lab <- cvs_image + seg_2_lab
  }
  
  if (!is.null(seg_1)) {
    seg_1_lab <- seg_1_lab * 1000 # Label clusters from seg_2 and clusters from seg_1 separately
    cvs_image[cvs_image == 0] <- seg_1[cvs_image == 0]
    cvs_image_lab[cvs_image_lab == 0] <- seg_1_lab[cvs_image_lab == 0]
  }
  
  if (nrow(coord_df) != 0) {
    coord_df <- coord_df[order(coord_df$cvs_id, decreasing = TRUE), ] # Make sure that CVS are written last so they overlap normal lesion)
    for (i in 1:nrow(coord_df)) {
      coord <- c(coord_df$x_coord[i], coord_df$y_coord[i], coord_df$z_coord[i])
      if (any(coord > dim(flair) + 1) | any(coord < c(1, 1, 1))) next # If too close to either edge, skip (or data entry error)
      cluster_label <- as.numeric(cvs_image_lab[coord[1], coord[2], coord[3]])
      if (cluster_label == 0) { # if the point is not currently labeled, make a 3x3x3 box with the ID
        cvs_image[(coord[1] - 1):(coord[1] + 1), 
                  (coord[2] - 1):(coord[2] + 1), 
                  (coord[3] - 1):(coord[3] + 1)] <- coord_df$cvs_id[i]
      }
      
      if (cluster_label != 0) { # if the point is already labeled, relabel that cluster to the correct ID
        cvs_image[cvs_image_lab == cluster_label] <- coord_df$cvs_id[i]
      }
    }
  }
  
  seg_exists_indicator <- !is.null(seg_1) | !is.null(seg_2)
  
  # return(c(as.numeric(seg_exists_indicator), 
  #             max(labelClusters(cvs_image, minClusterSize = 1)),
  #             max(labelClusters(cvs_image == 1, minClusterSize = 1))))
  
  
  if (seg_exists_indicator) {
    antsImageWrite(cvs_image, file.path(combined_dir, subject_id,
                                        "cvs_combined.nii.gz"))
  } else {
    antsImageWrite(cvs_image, file.path(combined_dir, subject_id,
                                        "cvs_combined_nl.nii.gz"))
  }
  
  output <- c(as.numeric(seg_exists_indicator),
              max(labelClusters(cvs_image, maxThresh = 10, minClusterSize = 1)),
              max(labelClusters(cvs_image == 1, minClusterSize = 1)))
  print(output)
  return(output)
}

####
out_dir <- here("data/gold_standard/CVS_combined_pre")
key <- read_xlsx(here("data/csvs/Key.xlsx"))
names(key) <- c("random_id", "patient_id", "pre_post")

randomized_path <- here("data/csvs/randomized_CVS.xlsx")
randomized_sheets <- excel_sheets(randomized_path) %>% as.numeric()
remaining_path <- here("data/csvs/remaining_CVS.xlsx")
remaining_sheets <- excel_sheets(remaining_path)

randomized_list <- lapply(6:length(excel_sheets(randomized_path)), # First 5 sheets were for analysis
                          function(i) {
                            process_sheet(randomized_path, i, keep_all = T)
                          })
remaining_list <- lapply(1:length(excel_sheets(remaining_path)), 
                         function(i) {
                           process_sheet(remaining_path, i, keep_all = T)
                         })

key_id_pair <- left_join(
  data.frame(random_id = randomized_sheets[6:length(randomized_sheets)]), key)

names(randomized_list) <- paste(key_id_pair[, 2], key_id_pair[, 3]) %>% str_trim()
names(remaining_list) <- remaining_sheets %>% str_trim()

post_randomized_list <- combine_dataframes(randomized_list[grepl("Pre", names(randomized_list))])
post_remaining_list <- combine_dataframes(remaining_list[grepl("Pre", names(remaining_list))])

all_list <- append(post_randomized_list, post_remaining_list)

###
seg_1_dir <- here("data/gold_standard/CVS_lynn")
seg_2_dir <- here("data/gold_standard/CVS_carly")
flair_dir <- here("data/gold_standard/FLAIR_QMENTA")
combined_dir <- here("data/gold_standard/CVS_combined_pre")

n_lesions <- lapply(1:length(all_list), function(i) {
  label_seg_nifti(all_list, i,
                  seg_1_dir, seg_2_dir, flair_dir, combined_dir)
})

n_lesions_df <- do.call(rbind, n_lesions) %>% as.data.frame()
write.csv(n_lesions_df, "/home/fengling/Documents/prl/data/csvs/n_lesions_pre.csv")
