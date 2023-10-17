#bsub -q taki_normal -J "competitors[1-97]" -M 40000 -o ~/Documents/prl/stdout/competitors.txt bash -c "Rscript /home/fengling/Documents/prl/code/05_get_competitor_preds.R"
suppressPackageStartupMessages({
  library(ANTsR)
  library(ANTsRCore) # for cluster labeling
  library(extrantsr) # for bias correction, skull-stripping, and registration
  library(tidyverse)
})


proj_dir <- "/home/fengling/Documents/prl"
data_dir <- "/home/fengling/Documents/prl/data/processed_05/"
subjects <- list.files(file.path(data_dir))

if (Sys.getenv("LSB_JOBINDEX") == "") {
  i <- 62
} else {
  i <- as.numeric(Sys.getenv("LSB_JOBINDEX"))
}

get_competitor_pred <- function (subjects, data_dir, subject_ind) {
  competitor_df <- data.frame(matrix(nrow = 0, ncol = 5))
  subject <- subjects[subject_ind]
  print(subject)
  
  cvs_path <- file.path(data_dir, subject, "cvs_masks.nii.gz")
  prl_path <- file.path(data_dir, subject, "aprl_masks.nii.gz")
  
  lesion_labels <- check_ants(file.path(data_dir, subject, "prob_05.nii.gz"))
  mimosa <- check_ants(file.path(data_dir, subject, "prob.nii.gz"))
  if (file.exists(cvs_path)) {
    cvs <- check_ants(cvs_path) 
  } else {
    cvs <- NULL
  }
  if (file.exists(prl_path)) {
    prl <- check_ants(prl_path)
    prl_csv <- read.csv(file.path(data_dir, subject, "aprl_preds.csv"))
  } else {
    prl <- NULL
    prl_csv <- NULL
  }
  
  n_lesions <- max(lesion_labels)
  print(paste0("Total of ", n_lesions, " lesions"))
  
  competitor_df <- data.frame(matrix(numeric(n_lesions * 5), ncol = 5))
  for (j in 1:n_lesions) {
    tmp_lesion_mask <- lesion_labels == j
    mimosa_pred <- max(mimosa * tmp_lesion_mask)
    
    if (!is.null(prl)) {
      prl_id <- unique(prl[tmp_lesion_mask])
      prl_id <- prl_id[prl_id != 0]
      if (is_empty(prl_id)) {
        prl_pred <- 0
      } else {
        if (length(prl_id) > 1) {
          print(j)
          warning(paste0("Multiple APRL lesions identified in lesion ", j, " ", subject))
          masked_prl <- prl[tmp_lesion_mask]
          masked_vec <- masked_prl[masked_prl != 0]
          lesion_counts <- table(masked_vec)
          print(lesion_counts)
          prl_id <- as.numeric(names(lesion_counts)[which.max(lesion_counts)])
        }
        prl_pred <- prl_csv[prl_csv$X == prl_id, 3]
      }
    } else {
      prl_pred <- 0
    }
    cvs_pred <- if (!is.null(cvs)) max(cvs * tmp_lesion_mask) else 0
    
    competitor_df[j, ] <- c(j, mimosa_pred, prl_pred, cvs_pred, subject)
  }
  
  competitor_df <- competitor_df %>% rename(lesion_id = X1, 
                                            mimosa_pred = X2,
                                            aprl_pred = X3,
                                            acvs_pred = X4,
                                            subject = X5)
  competitor_df$lesion_id <- as.numeric(competitor_df$lesion_id)
  return(competitor_df)
}

competitor_preds <- get_competitor_pred(subjects, data_dir, i)

write.csv(competitor_preds, 
          file.path(proj_dir, 
                    paste0("cv_output/competitors/competitor_", i, ".csv")), 
          row.names = FALSE)
