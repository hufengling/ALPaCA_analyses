make_binary_mask <- function(img, threshold) {
  img_thresh <- antsImageClone(img)
  img_thresh[img >= threshold] <- 1
  img_thresh[img < threshold] <- 0
  
  return(img_thresh)
}

get_type <- function(id_vec) {
  counts <- table(id_vec)
  if (length(counts) == 1) {
    return(id_vec[1])
  }
  if ("0" %in% names(counts)) {
    counts <- counts[-which(names(counts) == "0")] # exclude 0
  }
  return(as.numeric(names(counts)[which.max(counts)]))
}

annotate_lesion_mask <- function(labeled_mask, cvs_exists, 
                                 prl_coords, cvs_coords = NULL, contains_lesions) {
  lesion_type <- antsImageClone(labeled_mask)
  n_lesions <- max(unique(labeled_mask))
  browser()
  if (n_lesions > 0) {
    for (i in 1:n_lesions) {
      print(i)
      is_lesion <- 0
      is_prl <- 0
      is_cvs <- 0
      
      if (!cvs_exists) {
        is_lesion <- 9
        is_cvs <- 9
      }
      if (!contains_lesions) { # if standard lesions were not marked, can't make conclusions on them.
        is_lesion <- 9
      }
      
      # Annotate lesions according to PRL dataset
      prl_id <- prl_coords[labeled_mask == i]
      prl_type <- max(prl_id) # get_type(prl_id)
      
      if (prl_type == 1) {
        is_lesion <- 1 # If lesion is identified in PRL or CVS segmentations, 
        # can overwrite "contains_lesions = F"
      }
      if (prl_type == 2) {
        is_prl <- 1
        is_lesion <- 1
      }
      
      if (cvs_exists) {
        cvs_id <- cvs_coords[labeled_mask == i]
        cvs_type <- max(cvs_id) # get_type(cvs_id)
        
        if (cvs_type == 2 | cvs_type == 1) { # If marked, can overwrite is_lesion to be 1
          is_lesion <- 1
        }
        if (cvs_type == 3) {
          is_lesion <- 1
          is_cvs <- 1
        }
      }
      
      # Label each lesion as code "1LPC" where "L" is lesion identity, "P" is PRL identity, "C" is CVS identity
      # leading 1 so the lesion is not lost if some/all are 0
      lesion_type[labeled_mask == i] <- as.numeric(paste0(1, is_lesion, is_prl, is_cvs))
    }
  }
  
  return(lesion_type)
}
