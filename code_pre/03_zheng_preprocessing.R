# bsub -q taki_normal -J "preprocess[1-97]" -M 40000 -o ~/Documents/prl/stdout/preprocess.txt bash -c "Rscript /home/fengling/Documents/prl/code/03_zheng_preprocessing.R"

# Load libraries --------------------------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(neurobase) # for reading and writing nifti files
  library(mimosa) # for performing lesion segmentation
  library(fslr) # for smoothing and tissue class segmentation
  library(parallel) # for working in parallel
  library(pbmcapply) # for working in parallel
  library(WhiteStripe)
  library(stringr)
  library(neurobase)
  library(ANTsR)
  library(ANTsRCore) # for cluster labeling
  library(extrantsr) # for bias correction, skull-stripping, and registration
  library(caret)
  library(pbapply)
  library(tidyverse)
  library(neuroim)
  library(here)
})
source(here("code/00_lesion_helper_functions.R"))
source(here("code/00_preprocessing_helper_functions.R"))

if (Sys.getenv("LSB_JOBINDEX") == "" | Sys.getenv("LSB_JOBINDEX") == "0") {
  i <- 87
} else {
  i <- as.numeric(Sys.getenv("LSB_JOBINDEX"))
}

# Setup --------------------------------------------------------------------------------------------------
skip_exists <- TRUE
skip_all <- FALSE

patients <- list.files(here("data/Zheng_data"))
patient <- patients[i]
Zheng_path <- list.files(here("data/Zheng_data", patient),
                         recursive = TRUE, full.names = TRUE
)
Abby_path <- list.files(here("data/CAVS_Pilot_Abby", patient),
                        recursive = TRUE, full.names = TRUE
)
t1_brain_path <- Zheng_path[grepl("3D_T1_MPRAGE_n4_brain.nii.gz/3D_T1_MPRAGE_n4_brain.nii.gz", Zheng_path)]
t1_n4_path <- Zheng_path[grepl("n4_corrected/3D_T1_MPRAGE_n4.nii.gz", Zheng_path)]
t2_n4_path <- Zheng_path[grepl("n4_corrected/3D_T2_FLAIR_n4.nii.gz", Zheng_path)]
epi_path <- Abby_path[grepl("EPI_GAD.nii.gz", Abby_path)]
phase_path <- Zheng_path[which(grepl("before_processing/1-pha_pre.nii.gz", Zheng_path))]

reg_out_dir <- here("data/processed_05/", patient)
dir.create(reg_out_dir, showWarnings = FALSE, recursive = TRUE)

# EPI and Phase N4 correction -------------------------------------------------
if (!skip_all) {
  n4_exists <- all(sapply(c(file.path(reg_out_dir, "prob.nii.gz"),
                            file.path(reg_out_dir, "t1_final.nii.gz"),
                            file.path(reg_out_dir, "flair_final.nii.gz"),
                            file.path(reg_out_dir, "epi_final.nii.gz"),
                            file.path(reg_out_dir, "phase_final.nii.gz"),
                            file.path(reg_out_dir, "mask_final.nii.gz"),
                            file.path(reg_out_dir, "t1_n4_bet.nii.gz"),
                            file.path(reg_out_dir, "flair_n4_bet.nii.gz"),
                            file.path(reg_out_dir, "epi_n4_bet.nii.gz"),
                            file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz")),
                          file.exists))
  if (skip_exists & n4_exists) {
    cat("Skipping preprocessing of raw images and MIMoSA probability masks.")
    epi <- read_rpi(epi_path[which(grepl(p, epi_path))])
    epi_n4 <- oro2ants(bias_correct(epi, correction = "N4", 
                                    reorient = F, verbose = F)) # Need epi_n4 for lesion/CVS registration
  } else {
    epi <- read_rpi(epi_path[which(grepl(p, epi_path))])
    phase <- read_rpi(phase_path[which(grepl(p, phase_path))])
    epi_n4 <- oro2ants(bias_correct(epi, correction = "N4", 
                                    reorient = F, verbose = F))
    phase_n4 <- oro2ants(bias_correct(phase, correction = "N4", reorient = F,
                                      verbose = F))
    t1_n4_brain <- check_ants(t1_brain_path[which(grepl(p, t1_brain_path))])
    brain_mask <- t1_n4_brain != 0
    brain_mask <- oro2ants(orient_rpi(ants2oro(brain_mask))$img)
    t1_n4 <- check_ants(t1_n4_path[which(grepl(p, t1_n4_path))])
    flair_n4 <- check_ants(t2_n4_path[which(grepl(p, t2_n4_path))])
    
    ## Register
    t1_to_epi <- registration(
      filename = t1_n4,
      template.file = epi_n4,
      typeofTransform = "Rigid", 
      remove.warp = FALSE, verbose = F
    )
    
    flair_to_epi <- registration(
      filename = flair_n4,
      template.file = epi_n4,
      typeofTransform = "Rigid", 
      remove.warp = FALSE, verbose = F
    )
    
    phase_n4 <- antsCopyImageInfo(epi_n4, phase_n4)
    
    t1_reg <- antsApplyTransforms(
      fixed = epi_n4, moving = t1_n4,
      transformlist = c(t1_to_epi$fwdtransforms),
      interpolator = "lanczosWindowedSinc"
    )
    
    brain_mask <- antsApplyTransforms(
      fixed = epi_n4, moving = brain_mask,
      transformlist = c(t1_to_epi$fwdtransforms),
      interpolator = "nearestNeighbor"
    )
    
    flair_reg <- antsApplyTransforms(
      fixed = epi_n4, moving = flair_n4,
      transformlist = c(flair_to_epi$fwdtransforms),
      interpolator = "lanczosWindowedSinc"
    )
    
    ## Apply brain mask and WhiteStripe data
    epi_dist <- c(mean(epi_n4[brain_mask]), sd(epi_n4[brain_mask]))
    epi_final <- ((epi_n4 - epi_dist[1]) / epi_dist[2]) * brain_mask
    
    tmp <- ants2oro(phase_n4 * brain_mask)
    ind <- whitestripe(tmp, "T2", stripped = TRUE, verbose = F)
    phase_n4_bet_ws <- oro2ants(
      whitestripe_norm(tmp, ind$whitestripe.ind)
    )
    
    phase_dist <- c(mean(phase_n4[brain_mask]), sd(phase_n4[brain_mask]))
    phase_final <- ((phase_n4 - phase_dist[1]) / phase_dist[2]) * brain_mask
    
    tmp <- ants2oro(t1_reg * brain_mask)
    ind <- whitestripe(tmp, "T1", stripped = TRUE, verbose = F)
    t1_final <- oro2ants(
      whitestripe_norm(tmp, ind$whitestripe.ind)
    )
    
    tmp <- ants2oro(flair_reg * brain_mask)
    ind <- whitestripe(tmp, "T2", stripped = TRUE, verbose = F)
    flair_final <- oro2ants(
      whitestripe_norm(tmp, ind$whitestripe.ind)
    )
    
    # Prepare data for MIMoSA
    mimosa <- mimosa_data(
      brain_mask = ants2oro(brain_mask),
      FLAIR = ants2oro(flair_final),
      T1 = ants2oro(t1_final),
      gold_standard = NULL, normalize = "no",
      cores = 1, verbose = F
    )
    mimosa_df <- mimosa$mimosa_dataframe
    cand_voxels <- mimosa$top_voxels
    tissue_mask <- mimosa$tissue_mask
    
    # Apply model to test image
    load("/home/fengling/Documents/prl/data/mimosa_model.RData")
    predictions_WS <- predict(mimosa_model, mimosa_df, type = "response")
    predictions_nifti_WS <- niftiarr(cand_voxels, 0)
    predictions_nifti_WS[cand_voxels == 1] <- predictions_WS
    probmap <- oro2ants(
      fslsmooth(predictions_nifti_WS,
                sigma = 1.25,
                mask = tissue_mask, retimg = TRUE, smooth_mask = TRUE,
                verbose = F
      )
    ) # probability map
    
    flair_dist <- c(mean(flair_final[brain_mask]), sd(flair_final[brain_mask]))
    flair_final <- ((flair_final - flair_dist[1]) / flair_dist[2]) * brain_mask
    
    t1_dist <- c(mean(t1_final[brain_mask]), sd(t1_final[brain_mask]))
    t1_final <- ((t1_final - t1_dist[1]) / t1_dist[2]) * brain_mask
    
    antsImageWrite(t1_reg * brain_mask, file.path(reg_out_dir, "t1_n4_bet.nii.gz"))
    antsImageWrite(flair_reg * brain_mask, file.path(reg_out_dir, "flair_n4_bet.nii.gz"))
    antsImageWrite(epi_n4 * brain_mask, file.path(reg_out_dir, "epi_n4_bet.nii.gz"))
    antsImageWrite(phase_n4_bet_ws, file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz"))
    antsImageWrite(probmap, file.path(reg_out_dir, "prob.nii.gz"))
    antsImageWrite(t1_final, file.path(reg_out_dir, "t1_final.nii.gz"))
    antsImageWrite(flair_final, file.path(reg_out_dir, "flair_final.nii.gz"))
    antsImageWrite(epi_final, file.path(reg_out_dir, "epi_final.nii.gz"))
    antsImageWrite(phase_final, file.path(reg_out_dir, "phase_final.nii.gz"))
    antsImageWrite(brain_mask, file.path(reg_out_dir, "mask_final.nii.gz"))
    
    rm(
      probmap, t1_final, flair_final, epi_final, phase_final, brain_mask,
      predictions_WS, predictions_nifti_WS, mimosa_df, cand_voxels, mimosa, tmp, ind,
      t1_reg, flair_reg, epi, phase,
      phase_n4, t1_n4_brain, t1_n4, flair_n4, t1_to_epi, flair_to_epi, tissue_mask
    )
  }
}

# Also register CVS coordinates --------------------------------------------------------------------------------------------------
if (!skip_all) {
  cat("Mapping CVS")
  contains_lesions <- FALSE
  cvs_dir <- file.path("/home/fengling/Documents/prl/data/CVS_combined", p)
  if (file.exists(cvs_dir)) {
    flair_cvs_path <- file.path("/home/fengling/Documents/prl/data/FLAIR_QMENTA/", p, 
                                "N4/FLAIR_aligned_N4.nii.gz")
    cvs_coords_path <- list.files(cvs_dir, full.names = TRUE)
    contains_lesions <- !grepl("cvs_combined_nl", cvs_coords_path)
    
    flair_cvs <- check_ants(flair_cvs_path)
    cvs_coords <- check_ants(cvs_coords_path)
    
    cvs_to_epi <- registration(
      filename = flair_cvs,
      template.file = epi_n4,
      typeofTransform = "Rigid", remove.warp = FALSE, verbose = F
    )
    
    cvs_reg <- antsApplyTransforms(
      fixed = epi_n4, moving = cvs_coords,
      transformlist = cvs_to_epi$fwdtransforms,
      interpolator = "nearestNeighbor"
    )
    
    cvs_orig <- antsImageClone(cvs_reg)
    cvs_reg[cvs_orig == 2 | cvs_orig == 5] <- 1 # standard, non-CVS lesion
    cvs_reg[cvs_orig == 3 | cvs_orig == 4 | cvs_orig == 6] <- 2 # possible CVS lesion or excluded (may contain confluent with CVS or multiple veins)
    cvs_reg[cvs_orig == 1] <- 3 # definite CVS lesion
    
    if (contains_lesions) {
      antsImageWrite(cvs_reg, file.path(reg_out_dir, "cvs_coords.nii.gz"))
    } else {
      antsImageWrite(cvs_reg, file.path(reg_out_dir, "cvs_coords_nl.nii.gz"))
    }
    rm(cvs_reg, flair_cvs, cvs_coords, cvs_to_epi, cvs_orig)
  }
}

# Register phase --------------------------------------------------------------------------------------------------
# PRL coordinates are in phase space, but we need them to be in epi space.
# So need to put PRL coords in phase space, then reorient them along with phase.
if (!skip_all) {
  cat("Mapping PRLs")
  prl_coords <- read.csv("/home/fengling/Documents/prl/data/csvs/CAVS_pilot_prls.csv")
  
  prl_coords <- prl_coords %>%
    mutate(
      x_coord = str_split(coordinate, "-", simplify = TRUE)[, 1],
      y_coord = str_split(coordinate, "-", simplify = TRUE)[, 2],
      z_coord = str_split(coordinate, "-", simplify = TRUE)[, 3]
    ) %>%
    filter(coordinate != "") %>%
    mutate(prl_status = ifelse(prl_status == "yes", 1, 0))
  
  phase <- read_rpi(phase_path[which(grepl(p, phase_path))])
  phase_nii <- readnii(phase_path[which(grepl(p, phase_path))])
  
  subject_prl_coords <- filter(prl_coords, subject == p)
  prl_mat <- array(0, dim = dim(phase_nii))
  
  if (nrow(subject_prl_coords) != 0) {
    for (j in 1:nrow(subject_prl_coords)) {
      coord <- as.numeric(subject_prl_coords[j, 5:7])
      prl_mat[coord[1], coord[2], coord[3]] <- subject_prl_coords[j, 4] + 1
    }
  }
  
  prl_nii <- phase_nii
  prl_nii@.Data <- prl_mat
  prl_rpi <- oro2ants(orient_rpi(prl_nii)$img)
  
  epi_final <- check_ants(file.path(reg_out_dir, "epi_final.nii.gz"))
  antsCopyImageInfo(epi_final, prl_rpi)
  
  prl_rpi <- iMath(prl_rpi, "GD", 2)
  
  antsImageWrite(prl_rpi, paste0(reg_out_dir, "/prl_coords.nii.gz"))
  
  rm(prl_rpi, prl_nii, phase_nii, prl_mat, subject_prl_coords, prl_coords)
}

# Label Lesions ------------------------------------------------------------------------------------------------
if (!skip_all) {
  cat("Labeling lesions")
  
  prob <- check_ants(file.path(reg_out_dir, "prob.nii.gz"))
  prl_coords <- check_ants(file.path(reg_out_dir, "prl_coords.nii.gz"))
  
  cvs_exists <- file.exists(file.path(reg_out_dir, "cvs_coords.nii.gz"))
  if (cvs_exists) {
    cvs_coords <- check_ants(file.path(reg_out_dir, "cvs_coords.nii.gz"))
  } else {
    cvs_coords <- NULL
  }
  
  prob_05 <- make_binary_mask(prob, 0.05)
  
  prob_30 <- make_binary_mask(prob, 0.30)
  
  if (sum(prob_05) == 0) {
    prob_05_labeled <- antsImageClone(prob_05)
    lesion_type <- antsImageClone(prob_05)
  }
  
  if (sum(prob_05) > 0) {
    prob_05_labeled <- oro2ants(label_lesion(
      ants2oro(prob_05),
      ants2oro(prob), mincluster = 30
    ))
    
    lesion_type <- annotate_lesion_mask(prob_05_labeled, cvs_exists,
                                        prl_coords, cvs_coords, contains_lesions)
  }
  
  antsImageWrite(prob_05_labeled, file.path(reg_out_dir, "prob_05.nii.gz"))
  antsImageWrite(lesion_type, file.path(reg_out_dir, "lesion_labels.nii.gz"))
}

#
contains_lesions <- FALSE
prob <- check_ants(file.path(reg_out_dir, "prob.nii.gz"))
prob_05_labeled <- check_ants(file.path(reg_out_dir, "prob_05.nii.gz"))

prl_coords <- check_ants(file.path(reg_out_dir, "prl_coords.nii.gz"))
cvs_exists <- file.exists(file.path(reg_out_dir, "cvs_coords.nii.gz")) | file.exists(file.path(reg_out_dir, "cvs_coords_nl.nii.gz"))
if (cvs_exists) {
  if (file.exists(file.path(reg_out_dir, "cvs_coords.nii.gz"))) {
    contains_lesions <- TRUE
    cvs_coords <- check_ants(file.path(reg_out_dir, "cvs_coords.nii.gz"))
  }
  if (file.exists(file.path(reg_out_dir, "cvs_coords_nl.nii.gz")))
    cvs_coords <- check_ants(file.path(reg_out_dir, "cvs_coords_nl.nii.gz"))
} else {
  cvs_coords <- NULL
}

if (sum(prob_05_labeled) > 0) {
  lesion_type <- annotate_lesion_mask(prob_05_labeled, cvs_exists,
                                      prl_coords, cvs_coords, contains_lesions)
  antsImageWrite(lesion_type, file.path(reg_out_dir, "lesion_labels.nii.gz"))
}

# Make .30 thresholded probability mask for APRL -------------------------------------------------
if (!skip_all) {
  if (sum(prob_30) == 0) {
    prob_30_labeled <- antsImageClone(prob_30)
  }
  
  if (sum(prob_30) > 0) {
    prob_30_labeled <- oro2ants(label_lesion(
      ants2oro(prob_30),
      ants2oro(prob), mincluster = 100
    ))
  }
  
  antsImageWrite(prob_30_labeled, file.path(reg_out_dir, "prob_30.nii.gz"))
  
  cat("Done")
}

# Label lesions that were not covered by MIMoSA -----------------------------------------------
subject_files <- list.files(file.path(reg_out_dir), full.names = TRUE)
lesion_labels <- check_ants(file.path(reg_out_dir, "lesion_labels.nii.gz"))
cvs_path <- subject_files[grepl("cvs_coords", subject_files)]
if (length(cvs_path) == 0) {
  contains_cvs <- 0
  contains_lesions <- 0
  cvs_coords <- NULL
} else {
  cvs_coords <- check_ants(cvs_path)
  contains_cvs <- 1
  if (grepl("cvs_coords_nl", cvs_path)) {
    contains_lesions <- 0
  } else {
    contains_lesions <- 1
  }
}

prl_path <- subject_files[grepl("prl_coords.nii.gz", subject_files)]
prl_coords <- check_ants(prl_path)

if (is.null(cvs_coords)) {
  prl_coords[lesion_labels != 0] <- 0
  prl_coords[prl_coords == 1] <- 1100
  prl_coorsd[prl_coords == 2] <- 1110
  lesion_labels_true <- lesion_labels + prl_coords
} else {
  prl_coords[lesion_labels != 0] <- 0
  cvs_coords[lesion_labels != 0] <- 0
  tmp <- antsImageClone(prl_coords)
  tmp[tmp != 0] <- 0
  
  tmp[prl_coords > 0 | cvs_coords > 0] <- 1100
  tmp[prl_coords != 2 & cvs_coords == 3] <- 1101
  tmp[prl_coords == 2 & cvs_coords != 3] <- 1110
  tmp[prl_coords == 2 & cvs_coords == 3] <- 1111
  
  lesion_labels_true <- lesion_labels + tmp
}

antsImageWrite(lesion_labels_true, file.path(reg_out_dir, "lesion_labels_true.nii.gz"))

