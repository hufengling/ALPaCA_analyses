#bsub -q taki_normal -J "label_predictions[1-97]" -M 40000 -o ~/Documents/prl/stdout/label_predictions.txt bash -c "Rscript /home/fengling/Documents/prl/code/07_label_predictions.R"
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

pred_dir <- file.path(here(), "data/test_predictions")
data_dir <- file.path(here(), "data/processed_05")
test_preds <- read.csv(file.path(here(), "cv_output/test_predictions.csv"))

make_predictions <- function (index, pred_df, data_dir, pred_dir) {
  subjects <- unique(pred_df$subject)
  print(index)
  individual <- subjects[index]
  subject_lesions <- pred_df[pred_df$subject == individual, ]
  subject_lesions <- subject_lesions %>% 
    mutate(true_label = ifelse(
      !is.na(lesion_true) & !is.na(cvs_true), 
      case_when(
        lesion_true == 0 & prl_true == 0 & cvs_true == 0 ~ 0, # not lesion
        lesion_true == 1 & prl_true == 0 & cvs_true == 0 ~ 1, # standard lesion
        lesion_true == 1 & prl_true == 1 & cvs_true == 0 ~ 2, # PRL
        lesion_true == 1 & prl_true == 0 & cvs_true == 1 ~ 3, # CVS
        lesion_true == 1 & prl_true == 1 & cvs_true == 1 ~ 4 # PRL and CVS
      ),
      ifelse(
        !is.na(cvs_true), # CVS is still NA
        case_when(
          prl_true == 0 & cvs_true == 0 ~ 0, # not PRL or CVS
          prl_true == 1 & cvs_true == 0 ~ 12, # PRL
          prl_true == 0 & cvs_true == 1 ~ 13, # CVS
          prl_true == 1 & cvs_true == 1 ~ 14 # PRL and CVS
        ),
        case_when( # Lesion and CVS are both NA
          prl_true == 0 ~ 0,
          prl_true == 1 ~ 22 # PRL or both PRL and CVS
        )
      )
    )) %>% 
    mutate(proposed_label = case_when(
      lesion_yn == 0 & prl_yn == 0 & cvs_yn == 0 ~ 0, # not lesion
      lesion_yn == 1 & prl_yn == 0 & cvs_yn == 0 ~ 1, # standard lesion
      lesion_yn == 1 & prl_yn == 1 & cvs_yn == 0 ~ 2, # PRL
      lesion_yn == 1 & prl_yn == 0 & cvs_yn == 1 ~ 3, # CVS
      lesion_yn == 1 & prl_yn == 1 & cvs_yn == 1 ~ 4 # PRL and CVS
    )) %>% 
    mutate(competitor_label = case_when(
      mimosa_yn == 0 & aprl_yn == 0 & acvs_yn == 0 ~ 0, # not lesion
      mimosa_yn == 1 & aprl_yn == 0 & acvs_yn == 0 ~ 1, # standard lesion
      mimosa_yn == 1 & aprl_yn == 1 & acvs_yn == 0 ~ 2, # PRL
      mimosa_yn == 1 & aprl_yn == 0 & acvs_yn == 1 ~ 3, # CVS
      mimosa_yn == 1 & aprl_yn == 1 & acvs_yn == 1 ~ 4 # PRL and CVS
    ))
  
  lesion_labels <- check_ants(file.path(data_dir, individual, "prob_05.nii.gz"))
  true_ants <- antsImageClone(lesion_labels) * 0
  proposed_ants <- antsImageClone(lesion_labels) * 0
  competitor_ants <- antsImageClone(lesion_labels) * 0
  
  n_lesions <- max(lesion_labels)
  for (j in 1:n_lesions) {
    lesion_preds <- subject_lesions[subject_lesions$lesion_id == j, ]
    tmp_lesion_mask <- lesion_labels == j
    
    true_ants <- true_ants + tmp_lesion_mask * lesion_preds[["true_label"]]
    proposed_ants <- proposed_ants + tmp_lesion_mask * lesion_preds[["proposed_label"]]
    competitor_ants <- competitor_ants + tmp_lesion_mask * lesion_preds[["competitor_label"]]
  }
  
  pred_path <- file.path(pred_dir, individual)
  dir.create(pred_path,
             recursive = TRUE, showWarnings = FALSE)
  antsImageWrite(true_ants, file.path(pred_path,
                                      "true.nii.gz"))
  antsImageWrite(proposed_ants, file.path(pred_path,
                                          "proposed.nii.gz"))
  antsImageWrite(competitor_ants, file.path(pred_path,
                                            "competitor.nii.gz"))
}

for (i in 1:97) {
  make_predictions(i, test_preds, data_dir, pred_dir)
}
