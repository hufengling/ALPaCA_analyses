# bsub -q taki_normal -J "make_predictions[1-97]" -M 40000 -o ~/Documents/prl/stdout/make_predictions_R.txt bash -c "Rscript /home/fengling/Documents/prl/code/09_make_predictions_with_rotation.R"

library(tidyverse)
library(ANTsR)
library(extrantsr)
library(torch)
library(here)
library(ALPaCA)

if (Sys.getenv("LSB_JOBINDEX") == "" | Sys.getenv("LSB_JOBINDEX") == "0") {
  i <- 5
} else {
  i <- as.numeric(Sys.getenv("LSB_JOBINDEX"))
}

make_predictions_gs <- function(ants_list = NULL,
                                t1 = NULL, flair = NULL, epi = NULL, phase = NULL, labeled_candidates = NULL,
                                gold_standard = NULL, model_id,
                                n_patches = 20, rotate_patches = TRUE,
                                verbose = FALSE) {
  # Error checking
  if (is.null(ants_list)) { # Make sure all images are provided
    if (any(is.null(t1),
            is.null(flair),
            is.null(epi),
            is.null(phase),
            is.null(labeled_candidates))) {
      stop("Images must either be provided via ants_list, or images must be provided for each of t1, flair, epi, phase, and labeled_candidates")
    }
    
    t1 <- check_ants(t1)
    flair <- check_ants(flair)
    epi <- check_ants(epi)
    phase <- check_ants(phase)
    labeled_candidates <- check_ants(labeled_candidates)
    gold_standard <- check_ants(gold_standard)
  }
  
  if (!is.null(ants_list)) { # Make sure all images are provided
    if (!all(c("t1", "flair", "epi", "phase", "labeled_candidates") %in% names(ants_list))) {
      stop("If images are provided via ants_list, ants_list must be a named list with items: t1, flair, epi, phase, labeled_candidates. Output from preprocess_images() function can be directly used with return_image = TRUE.")
    }
    t1 <- check_ants(ants_list$t1)
    flair <- check_ants(ants_list$flair)
    epi <- check_ants(ants_list$epi)
    phase <- check_ants(ants_list$phase)
    labeled_candidates <- check_ants(ants_list$labeled_candidates)
  }
  
  if (n_patches < 1) {
    stop("n_patches must be a positive integer.")
  }
  
  # If there are no lesions, don't have to return anything
  if (sum(labeled_candidates) == 0) {
    warning("No lesion candidates detected.")
    return(NULL)
  }
  
  # Load CV models
  models_list <- list(jit_load(here(paste0("prl_pytorch/trace_models/autoencoder_", model_id, ".pt"))),
                      jit_load(here(paste0("prl_pytorch/trace_models/predictor_", model_id, ".pt"))))
  
  n_lesions <- max(labeled_candidates)
  # Pre-allocate memory
  prediction_tensor <- torch_zeros(c(n_lesions, 3))
  std_tensor <- torch_zeros_like(prediction_tensor)
  time_vec <- c()
  gold_standard_vec <- rep(0, n_lesions)
  
  if (verbose) {
    print("Running patches through ALPaCANet")
  }
  for (candidate_id in 1:n_lesions) {
    if (verbose) {
      print(paste0("Making predictions for lesion ", candidate_id,
                   " of ", n_lesions))
    }
    # Get indexes within lesion indexed by candidate_id
    candidate_coords <- which(labeled_candidates == candidate_id, arr.ind = TRUE)
    under_zero <- apply(candidate_coords - 12, 1, function(i) { # Check if patch bleeds into "nothing"
      any(i < 0)
    })
    over_dim <- apply(candidate_coords + 11, 1, function(i) { # Check if patch bleeds into "nothing" on other side
      any(i[1] > dim(t1)[1],
          i[2] > dim(t1)[2],
          i[3] > dim(t1)[3])
    })
    candidate_coords <- candidate_coords[!under_zero & !over_dim, ] # Check that all patches fit inside the images
    
    max_coords <- min(n_patches, nrow(candidate_coords)) # Sample some of the candidate_coords
    # If there are no coords with full patch, just guess 0 for everything
    if (max_coords == 0) {
      warning(paste0("No full patches could be extracted for lesion ", candidate_id, ". Default prediction of 0."))
      prediction_tensor[candidate_id, ] <- torch_zeros(c(1, 3))
      std_tensor[candidate_id, ] <- torch_zeros(c(1, 3))
      next
    }
    
    if (max_coords < n_patches & rotate_patches) { # If we are rotating patches, there is less dependence and resampling same point is still useful
      max_coords <- n_patches
      random_inds <- candidate_coords[sample(1:nrow(candidate_coords),
                                             n_patches, replace = TRUE), ]
    } else {
      random_inds <- candidate_coords[sample(1:nrow(candidate_coords), max_coords), ]
    }
    starts <- random_inds - 12
    ends <- random_inds + 11
    
    all_patch <- torch_zeros(c(max_coords, 4, 24, 24, 24)) # Pre-allocate memory
    for (patch_id in 1:max_coords) {
      all_patch[patch_id, , , ,] <- extract_patch(candidate_id,  # Extract patches centered at the candidate_coords above. Rotate and flip patches if desired to decrease dependency
                                                  starts[patch_id, ],
                                                  ends[patch_id, ],
                                                  t1, flair, epi, phase,
                                                  labeled_candidates,
                                                  rotate_patches = rotate_patches)
    }
    
    encoder <- models_list[[1]]$encoder # Extract encoder
    predictor <- models_list[[2]] # Extract predictor
    
    start_time <- Sys.time()
    with_no_grad({ # Run patches through model
      output <- predictor(encoder(all_patch))
    })
    end_time <- Sys.time()
    time_vec <- c(time_vec, end_time - start_time)
    
    all_output <- output
    
    prediction_tensor[candidate_id, ] <- torch_mean(all_output, dim = 1, # Get the mean prediction for all coordinates and models
                                                    keepdim = TRUE)
    std_tensor[candidate_id, ] <- torch_std(all_output, dim = 1, # Get the standard deviation of all predictions for a sense of uncertainty
                                            keepdim = TRUE)
    gold_standard_vec[candidate_id] <- gold_standard[candidate_coords[1, 1],
                                                     candidate_coords[1, 2],
                                                     candidate_coords[1, 3]]
  }
  
  # Convert torch tensor to dataframe
  predictions <- as.data.frame(as.matrix(prediction_tensor))
  std <- as.data.frame(as.matrix(std_tensor))
  names(predictions) <- c("Lesion", "PRL", "CVS")
  names(std) <- c("Lesion", "PRL", "CVS")
  
  return(list(predictions = predictions,
              std = std,
              labels = gold_standard_vec,
              avg_time = mean(time_vec)))
}

cv_splits <- read.csv(here("cv_output/cv_df.csv"))
subject_id <- cv_splits[i, 2]

for (model_id in 1:10) {
  t1_path <- here(paste0("data/processed_05/", subject_id, "/t1_final.nii.gz"))
  flair_path <- here(paste0("data/processed_05/", subject_id, "/flair_final.nii.gz"))
  epi_path <- here(paste0("data/processed_05/", subject_id, "/epi_final.nii.gz"))
  phase_path <- here(paste0("data/processed_05/", subject_id, "/phase_final.nii.gz"))
  labeled_path <- here(paste0("data/processed_05/", subject_id, "/prob_05.nii.gz"))
  gold_standard_path <- here(paste0("data/processed_05/", subject_id, "/lesion_labels.nii.gz"))
  output_list <- make_predictions_gs(t1 = t1_path, flair = flair_path, 
                                     epi = epi_path, phase = phase_path,
                                     labeled_candidates = labeled_path,
                                     gold_standard = gold_standard_path,
                                     model_id = model_id, verbose = TRUE)
  
  # Process output labels
  labels_df <- data.frame(do.call(rbind, 
                                  str_split(output_list$labels, ""))[, -1])
  weights_df <- data.frame(labels_df != 9)
  labels_df[labels_df == 9] <- 0
  weights_df[weights_df == TRUE] <- 1
  weights_df[weights_df == FALSE] <- 0

  
  # Make as dataframe
  model_output <- cbind(output_list$predictions, output_list$std,
                        labels_df, weights_df, 
                        subject_id)

  write.csv(model_output, here(paste0("cv_output/split_", model_id, 
                                      "/subject_", i, ".csv")))
}
