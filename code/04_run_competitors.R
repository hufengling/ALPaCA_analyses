#bsub -q taki_normal -J "run_competitors[1-97]" -M 40000 -o ~/Documents/prl/stdout/run_competitors.txt bash -c "Rscript /home/fengling/Documents/prl/code/04_run_competitors.R"
suppressPackageStartupMessages({
  library(neurobase) # for reading and writing nifti files
  library(oro.nifti)
  library(mimosa) # for performing lesion segmentation
  library(fslr) # for smoothing and tissue class segmentation
  library(parallel) # for working in parallel
  library(pbmcapply) # for working in parallel
  library(WhiteStripe)
  library(stringr)
  library(ANTsR)
  library(ANTsRCore) # for cluster labeling
  library(extrantsr) # for bias correction, skull-stripping, and registration
  library(caret)
  library(pbapply)
  library(tidyverse)
  library(neuroim)
  library(RIA)
  library(Rfast)
  library(stats)
})

pretrainedmodel = readRDS("/home/fengling/Documents/prl/data/fit.rf.sm.fo.orig.10c3mask.rds")
source("/home/fengling/Documents/prl/code/00_lesion_helper_functions.R")

# Setup --------------------------------------------------------------------------------------------------
processed_path <- "/home/fengling/Documents/prl/data/processed_05"
patient <- list.files(processed_path)

if (Sys.getenv("LSB_JOBINDEX") == "" | Sys.getenv("LSB_JOBINDEX") == "0") {
  i <- 2
} else {
  i <- as.numeric(Sys.getenv("LSB_JOBINDEX"))
}
p <- patient[i]
reg_out_dir <- file.path(processed_path, p)

epi <- readnii(file.path(reg_out_dir, "epi_n4_bet.nii.gz"))
t1 <- readnii(file.path(reg_out_dir, "t1_n4_bet.nii.gz"))
flair <- readnii(file.path(reg_out_dir, "flair_n4_bet.nii.gz"))
#phase <- readnii(file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz"))

lesion_labels <- readnii(file.path(reg_out_dir, "prob_30.nii.gz"))
mask <- readnii(file.path(reg_out_dir, "mask_final.nii.gz"))
prob <- readnii(file.path(reg_out_dir, "prob.nii.gz"))

cvs <- centralveins(epi = epi, t1 = t1, flair = flair, mask = mask,
                    probmap = prob, binmap = lesion_labels != 0,
                    parallel = T, cores = 4,
                    biascorrected = T, skullstripped = T, registered = T,
                    c3d = F)
if (!is.null(cvs)) {
  writenii(cvs$cvs.probmap, file.path(reg_out_dir, "cvs_masks"))
  write.csv(cvs$cvs.biomarker, file.path(reg_out_dir, "cvs_biomarker.csv"),
            row.names = FALSE)
}

findprls_out <- findprls(lesmask = lesion_labels,
                         phasefile = file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz"),
                         pretrainedmodel = pretrainedmodel)
if (!is.null(findprls_out)) {
  leslabels_img = findprls_out$leslabels
  writenii(leslabels_img, file.path(reg_out_dir, "aprl_masks"))
  ria.df = findprls_out$ria.df
  preds = findprls_out$preds
  write.csv(preds, file.path(reg_out_dir, "aprl_preds.csv"))
}
