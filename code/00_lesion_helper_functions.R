#' fuzzySpatialCMeansSegmentation
#'
#' Fuzzy spatial c-means for image segmentation.
#'
#' Image segmentation using fuzzy spatial c-means as described in
#'
#' Chuang et al., Fuzzy c-means clustering with spatial information for image
#' segmentation.  CMIG: 30:9-15, 2006.
#'
#' @param image image to be segmented.
#' @param mask optional mask image.  Otherwise, the entire image is used.
#' @param numberOfClusters number of segmentation clusters
#' @param m fuzziness parameter (default = 2).
#' @param p membership importance parameter (default = 1).
#' @param q spatial constraint importance parameter (default = 1).
#' \code{q = 0} is equivalent to conventional fuzzy c-means.
#' @param radius neighborhood radius (scalar or array) for spatial
#' constraint.
#' @param maxNumberOfIterations iteration limit (default = 20).
#' @param convergenceThreshold Convergence between iterations is measured
#' using the Dice coefficient (default = 0.02).
#' @param verbose print progress.
#' @return list containing segmentation and probability images
#'
#' @author NJ Tustison
#'
#' @examples
#' image <- antsImageRead(getANTsRData("r16"))
#' mask <- getMask(image)
#' fuzzy <- fuzzySpatialCMeansSegmentation(image, mask)
#'
#' @export fuzzySpatialCMeansSegmentation

fuzzySpatialCMeansSegmentation <- function(image, mask = NULL,
                                           numberOfClusters = 4,
                                           m = 2, p = 1, q = 1, radius = 2,
                                           maxNumberOfIterations = 20,
                                           convergenceThreshold = 0.02,
                                           verbose = FALSE) {
  if (is.null(mask)) {
    mask <- antsImageClone(image) * 0 + 1
  }
  
  x <- image[mask != 0]
  
  v <- seq(
    from = 0, to = 1,
    length.out = numberOfClusters + 2
  )[2:(numberOfClusters + 1)]
  v <- v * (max(x) - min(x)) + min(x)
  cc <- length(v)
  
  if (verbose == TRUE) {
    cat("Initial cluster centers: ", v, "\n")
  }
  
  xx <- matrix(data = 0, nrow = cc, ncol = length(x))
  for (i in seq.int(cc))
  {
    xx[i, ] <- x
  }
  
  if (length(radius) == 1) {
    radius <- rep(radius, image@dimension)
  }
  
  segmentation <- antsImageClone(image) * 0
  probabilityImages <- NULL
  
  iter <- 0
  diceValue <- 0
  while (iter < maxNumberOfIterations && diceValue < 1.0 - convergenceThreshold) {
    # Update membership values
    
    xv <- matrix(data = 0, nrow = cc, ncol = length(x))
    for (k in seq.int(cc))
    {
      xv[k, ] <- abs(x - v[k])
    }
    
    u <- matrix(data = 0, nrow = nrow(xv), ncol = ncol(xv))
    for (i in seq.int(cc))
    {
      n <- xv[i, ]
      
      d <- n * 0
      for (k in seq.int(cc))
      {
        d <- d + (n / xv[k, ])^(2 / (m - 1))
      }
      u[i, ] <- 1 / d
    }
    u[is.nan(u)] <- 1
    
    
    # Update cluster centers
    
    v <- rowSums((u^m) * xx, na.rm = TRUE) / rowSums(u^m, na.rm = TRUE)
    
    if (verbose == TRUE) {
      cat("Updated cluster centers: ", v, "\n")
    }
    
    # Spatial function
    
    h <- matrix(data = 0, nrow = nrow(u), ncol = ncol(u))
    for (i in seq.int(cc))
    {
      uImage <- antsImageClone(image) * 0
      uImage[mask != 0] <- u[i, ]
      uNeighborhoods <- getNeighborhoodInMask(uImage, mask, radius)
      h[i, ] <- colSums(uNeighborhoods, na.rm = TRUE)
    }
    
    # u prime
    
    d <- rep(0, ncol(u))
    for (k in seq.int(cc))
    {
      d <- d + (u[k, ]^p) * (h[k, ]^q)
    }
    
    probabilityImages <- list()
    uprime <- matrix(data = 0, nrow = nrow(u), ncol = ncol(u))
    for (i in seq.int(cc))
    {
      uprime[i, ] <- (u[i, ]^p) * (h[i, ]^q) / d
      uprimeImage <- antsImageClone(image) * 0
      uprimeImage[mask != 0] <- uprime[i, ]
      probabilityImages[[i]] <- uprimeImage
    }
    
    tmpSegmentation <- antsImageClone(image) * 0
    tmpSegmentation[mask != 0] <- max.col(t(uprime))
    
    diceValue <- labelOverlapMeasures(segmentation, tmpSegmentation)$MeanOverlap[1]
    iter <- iter + 1
    
    if (verbose == TRUE) {
      cat("Iteration ", iter, " (out of ", maxNumberOfIterations, "):  ",
          "Dice overlap = ", diceValue, "\n",
          sep = ""
      )
    }
    segmentation <- tmpSegmentation
  }
  return(list(
    segmentationImage = segmentation,
    probabilityImages = probabilityImages
  ))
}

#' @title 3D Image Gradient
#' @description This function returns the gradient images for a 3D array or NIfTI volume.
#' @param image a 3D array or image of class \code{nifti}
#' @param mask an array or \code{nifti} mask of voxels for which the gradient will be calculated,
#' if \code{NULL} the gradient will be run for the full array.
#' Note that mask should be in the same space as the image volume
#' @param which a string specifying the gradient direction that should be returned; either "all" for a list of x, y, and z gradient volumes, or "x", "y", or "z" for a single volume with the given gradient
#' @param radius an integer specifying radius of the neighborhood (in voxels) for which the gradient should be calculated
#'
#' @return Either a list of three gradient volumes or a single gradient volume, in either array or NIfTI format based on what was input.
#' @examples \dontrun{
#' library(neurobase)
#' epi <- readnii("path/to/epi")
#' gradients <- gradient(image = epi, which = "all")
#' }
#' @export
#' @importFrom oro.nifti is.nifti
gradient <- function(image, mask = NULL, which = "all", radius = 1) {
  if (radius >= min(dim(image))) {
    stop("Radius larger than smallest image dimension")
  }
  if (is.nifti(image)) {
    if (which == "all") {
      dx <- image
      dx@.Data[1:radius, , ] <- 0
      dx@.Data[dim(image)[1] - radius + 1, , ] <- 0
      dx@.Data[((1 + radius):(dim(image)[1] - radius)), , ] <-
        (image@.Data[((1 + 2 * radius):dim(image)[1]), , ] -
           image@.Data[(1:(dim(image)[1] - 2 * radius)), , ]) / (2 * radius)
      dx@.Data[mask@.Data == 0] <- 0
      
      dy <- image
      dy@.Data[, 1:radius, ] <- 0
      dy@.Data[, dim(image)[2] - radius + 1, ] <- 0
      dy@.Data[, ((1 + radius):(dim(image)[2] - radius)), ] <-
        (image@.Data[, ((1 + 2 * radius):dim(image)[2]), ] -
           image@.Data[, (1:(dim(image)[2] - 2 * radius)), ]) / (2 * radius)
      dy@.Data[mask@.Data == 0] <- 0
      
      dz <- image
      dz@.Data[, , 1:radius] <- 0
      dz@.Data[, , dim(image)[3] - radius + 1] <- 0
      dz@.Data[, , ((1 + radius):(dim(image)[3] - radius))] <-
        (image@.Data[, , ((1 + 2 * radius):dim(image)[3])] -
           image@.Data[, , (1:(dim(image)[3] - 2 * radius))]) / (2 * radius)
      dz@.Data[mask@.Data == 0] <- 0
      
      return(list(Dx = dx, Dy = dy, Dz = dz))
    } else if (which == "x") {
      dx <- image
      dx@.Data[1:radius, , ] <- 0
      dx@.Data[dim(image)[1] - radius + 1, , ] <- 0
      dx@.Data[((1 + radius):(dim(image)[1] - radius)), , ] <-
        (image@.Data[((1 + 2 * radius):dim(image)[1]), , ] -
           image@.Data[(1:(dim(image)[1] - 2 * radius)), , ]) / (2 * radius)
      dx@.Data[mask@.Data == 0] <- 0
      return(dx)
    } else if (which == "y") {
      dy <- image
      dy@.Data[, 1:radius, ] <- 0
      dy@.Data[, dim(image)[2] - radius + 1, ] <- 0
      dy@.Data[, ((1 + radius):(dim(image)[2] - radius)), ] <-
        (image@.Data[, ((1 + 2 * radius):dim(image)[2]), ] -
           image@.Data[, (1:(dim(image)[2] - 2 * radius)), ]) / (2 * radius)
      dy@.Data[mask@.Data == 0] <- 0
      return(dy)
    } else if (which == "z") {
      dz <- image
      dz@.Data[, , 1:radius] <- 0
      dz@.Data[, , dim(image)[3] - radius + 1] <- 0
      dz@.Data[, , ((1 + radius):(dim(image)[3] - radius))] <-
        (image@.Data[, , ((1 + 2 * radius):dim(image)[3])] -
           image@.Data[, , (1:(dim(image)[3] - 2 * radius))]) / (2 * radius)
      dz@.Data[mask@.Data == 0] <- 0
      return(dz)
    }
  } else if (is.array(image)) {
    if (which == "all") {
      dx <- image
      dx[1:radius, , ] <- 0
      dx[dim(image)[1] - radius + 1, , ] <- 0
      dx[((1 + radius):(dim(image)[1] - radius)), , ] <-
        (image[((1 + 2 * radius):dim(image)[1]), , ] -
           image[(1:(dim(image)[1] - 2 * radius)), , ]) / (2 * radius)
      dx[mask == 0] <- 0
      
      dy <- image
      dy[, 1:radius, ] <- 0
      dy[, dim(image)[2] - radius + 1, ] <- 0
      dy[, ((1 + radius):(dim(image)[2] - radius)), ] <-
        (image[, ((1 + 2 * radius):dim(image)[2]), ] -
           image[, (1:(dim(image)[2] - 2 * radius)), ]) / (2 * radius)
      dy[mask == 0] <- 0
      
      dz <- image
      dz[, , 1:radius] <- 0
      dz[, , dim(image)[3] - radius + 1] <- 0
      dz[, , ((1 + radius):(dim(image)[3] - radius))] <-
        (image[, , ((1 + 2 * radius):dim(image)[3])] -
           image[, , (1:(dim(image)[3] - 2 * radius))]) / (2 * radius)
      dz[mask == 0] <- 0
      
      return(list(Dx = dx, Dy = dy, Dz = dz))
    } else if (which == "x") {
      dx <- image
      dx[1:radius, , ] <- 0
      dx[dim(image)[1] - radius + 1, , ] <- 0
      dx[((1 + radius):(dim(image)[1] - radius)), , ] <-
        (image[((1 + 2 * radius):dim(image)[1]), , ] -
           image[(1:(dim(image)[1] - 2 * radius)), , ]) / (2 * radius)
      dx[mask == 0] <- 0
      return(dx)
    } else if (which == "y") {
      dy <- image
      dy[, 1:radius, ] <- 0
      dy[, dim(image)[2] - radius + 1, ] <- 0
      dy[, ((1 + radius):(dim(image)[2] - radius)), ] <-
        (image[, ((1 + 2 * radius):dim(image)[2]), ] -
           image[, (1:(dim(image)[2] - 2 * radius)), ]) / (2 * radius)
      dy[mask == 0] <- 0
      return(dy)
    } else if (which == "z") {
      dz <- image
      dz[, , 1:radius] <- 0
      dz[, , dim(image)[3] - radius + 1] <- 0
      dz[, , ((1 + radius):(dim(image)[3] - radius))] <-
        (image[, , ((1 + 2 * radius):dim(image)[3])] -
           image[, , (1:(dim(image)[3] - 2 * radius))]) / (2 * radius)
      dz[mask == 0] <- 0
      return(dz)
    }
  } else {
    print("Image must be array or NifTI")
  }
}

#' @title 3D Image Hessian
#' @description This function returns the eigenvalues of the hessian matrices for a 3D array or NIfTI volume.
#' @param image a 3D array or image of class \code{nifti}
#' @param mask an array or \code{nifti} mask of voxels for which vesselness will be calculated,
#' with more selective masking improving speed significantly.
#' Note that mask should be in the same space as the image volume
#' @param radius an integer specifying radius of the neighborhood (in voxels) for which the hessian should be calculated
#' @param parallel is a logical value that indicates whether the user's computer
#' is Linux or Unix (i.e. macOS), and should run the code in parallel
#' @param cores if parallel = TRUE, cores is an integer value that indicates how many cores
#' the function should be run on
#'
#' @return A list of three eigenvalue volumes.
#' @examples \dontrun{
#' library(neurobase)
#' epi <- readnii("path/to/epi")
#' mask <- epi != 0
#' hesseigs <- hessian(image = epi, mask = mask)
#' }
#' @export
#' @importFrom pbmcapply pbmclapply
#' @importFrom pbapply pblapply
#' @importFrom parallel detectCores
hessian <- function(image, mask, radius = 1, parallel = FALSE, cores = 2) {
  print("Getting derivatives")
  grads <- gradient(image, which = "all", radius = radius)
  gx <- grads$Dx
  gy <- grads$Dy
  gz <- grads$Dz
  rm(grads)
  
  gradsx <- gradient(gx, which = "all", radius = radius)
  gxx <- gradsx$Dx
  gxy <- gradsx$Dy
  gxz <- gradsx$Dz
  rm(gx, gradsx)
  
  gradsy <- gradient(gy, which = "all", radius = radius)
  gyx <- gradsy$Dx
  gyy <- gradsy$Dy
  gyz <- gradsy$Dz
  rm(gy, gradsy)
  
  gradsz <- gradient(gz, which = "all", radius = radius)
  gzx <- gradsz$Dx
  gzy <- gradsz$Dy
  gzz <- gradsz$Dz
  rm(gz, gradsz)
  
  print("Creating hessian matrices")
  bigmat <- cbind(
    as.vector(gxx[mask == 1]), as.vector(gxy[mask == 1]), as.vector(gxz[mask == 1]),
    as.vector(gyx[mask == 1]), as.vector(gyy[mask == 1]), as.vector(gyz[mask == 1]),
    as.vector(gzx[mask == 1]), as.vector(gzy[mask == 1]), as.vector(gzz[mask == 1])
  )
  
  rm(gxx, gxy, gxz, gyx, gyy, gyz, gzx, gzy, gzz)
  
  biglist <- split(bigmat, row(bigmat))
  biglist <- lapply(biglist, matrix, nrow = 3, byrow = T)
  
  rm(bigmat)
  
  getevals <- function(matrix) {
    thiseig <- eigen(matrix, only.values = TRUE)$values
    sort <- order(abs(thiseig))
    return(thiseig[sort])
  }
  
  print("Calculating eigenvalues")
  if (parallel == TRUE) {
    result <- matrix(unlist(pbmclapply(biglist, getevals, mc.cores = cores)),
                     ncol = 3, byrow = T
    )
  } else if (parallel == FALSE) {
    result <- matrix(unlist(pblapply(biglist, getevals)), ncol = 3, byrow = T)
  }
  e1 <- mask
  e1[mask == 1] <- result[, 1]
  e2 <- mask
  e2[mask == 1] <- result[, 2]
  e3 <- mask
  e3[mask == 1] <- result[, 3]
  
  return(list(eigval1 = e1, eigval2 = e2, eigval3 = e3))
}

#' @title Distinct Lesion Centers
#' @description This function finds the centers of distinct lesions based on a lesion probability map. The method is described in Dworkin et al., (2018).
#' @param probmap a 3D array or image of class \code{nifti}, containing the probability that each voxel is a lesion voxel
#' @param binmap a 3D array or \code{nifti} mask in which voxels are classified as either lesion voxels or not lesion voxels.
#' Note that mask should be in the same space as the probmap volume
#' @param c3d a logical value reflecting whether or not the Convert3D imaging toolbox is installed.
#' @param minCenterSize an integer value representing the minimum number of connected voxels that can be considered a lesion center
#' @param radius an integer specifying radius of the neighborhood (in voxels) for which the hessian should be calculated.
#' @param parallel is a logical value that indicates whether the user's computer
#' is Linux or Unix (i.e. macOS), and should run the code in parallel
#' @param cores if parallel = TRUE, cores is an integer value that indicates how many cores
#' the function should be run on
#'
#' @importFrom ANTsRCore labelClusters
#' @importFrom extrantsr ants2oro oro2ants
#' @return A list containing lesioncenters (a nifti file with labeled lesion centers) and lesioncount (an integer value representing the number of distinct lesions)
#' @examples \dontrun{
#' library(neurobase)
#' lesion.probs <- readnii("path/to/probabilitymap")
#' centers <- lesioncenters(
#'   probmap = lesion.probs, binmap = lesion.probs > 0.30,
#'   parallel = TRUE, cores = 4
#' )
#' }
#' @export
#' @references J.D. Dworkin, K.A. Linn, I. Oguz, G.M. Fleishman, R. Bakshi, G. Nair, P.A. Calabresi, R.G. Henry, J. Oh, N. Papinutto, D. Pelletier, W. Rooney, W. Stern, N.L. Sicotte, D.S. Reich, R.T. Shinohara. An automated statistical technique for counting distinct multiple sclerosis lesions. American Journal of Neuroradiology, 2018; 39, 626-633.
lesioncenters <- function(probmap, binmap, c3d = T, minCenterSize = 10, radius = 1, parallel = F, cores = 2) {
  scale <- ceiling((1 / mean(probmap@pixdim[2:4]))^3)
  if (c3d == T) {
    if (file.exists("/Applications/ITK-SNAP.app/Contents/bin/c3d")) {
      c3d_path <- "/Applications/ITK-SNAP.app/Contents/bin/c3d"
    } else {
      stop("Cannot find path to Convert3D\n
           If it is already installed via ITK-SNAP, please specify the correct path in the function call.\n
           If not, please download the software at http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3.")
    }
    tempprob <- tempfile(pattern = "file", tmpdir = tempdir(), fileext = ".nii.gz")
    writenii(probmap, tempprob)
    tempeigs <- tempfile(pattern = "file", tmpdir = tempdir())
    system(paste0(c3d_path, " ", tempprob, " -hesseig ", scale, " -oo ", paste0(tempeigs, "%02d.nii.gz")))
    
    phes1 <- readnii(paste0(tempeigs, "00.nii.gz"))
    phes2 <- readnii(paste0(tempeigs, "01.nii.gz"))
    phes3 <- readnii(paste0(tempeigs, "02.nii.gz"))
  } else if (c3d == F) {
    phes <- hessian(probmap, mask = binmap, radius, parallel, cores)
    phes1 <- phes$eigval1
    phes2 <- phes$eigval2
    phes3 <- phes$eigval3
  } else {
    stop("'c3d' must be TRUE or FALSE")
  }
  clusmap <- ants2oro(labelClusters(oro2ants(binmap), minClusterSize = 20 * scale))
  
  les <- clusmap
  les[les != 0] <- 1
  les[phes1 > 0 | phes2 > 0 | phes3 > 0] <- 0
  les <- ants2oro(labelClusters(oro2ants(les), minClusterSize = minCenterSize * scale))
  
  return(list(lesioncenters = les, lesioncount = max(les)))
}

#' @title Lesion Labelling
#' @description This function is a helper function for lesion_identification(). It takes in a lesion segmentation mask and a NIfTI image with identified lesion centers and returns a lesion label map.
#' @param lesmask Lesion segmentation mask. Given a probability threshold, automatically binarizes lesion probability map into a segmentation mask.
#' @param centers Lesion center map. Provided by lesiontools::lesioncenters().
#' @export
#' @import lesiontools
#' @import Rfast
#' @return A NIfTI with labels for each identified lesion.

getleslabels <- function(lesmask, centers) {
  #### knn on mimosa segmentations ####
  inds.lab <- which(lesmask == 1 & centers > 0, arr.ind = TRUE) # labeled indices
  inds.cand <- which(lesmask == 1 & centers == 0, arr.ind = T) # candidate indices
  num.candvox <- nrow(inds.cand)
  
  pairwisedists <- Rfast::dista(inds.cand, inds.lab, k = 1, index = T)
  
  lesmask[inds.lab] <- centers[inds.lab]
  lesmask[inds.cand] <- centers[inds.lab[pairwisedists, ]]
  return(lesmask)
}

## Relabel splitted lesions
relabel <- function(split_lesion, s) {
  for (i in 1:length(s)) {
    split_lesion[split_lesion == s[i]] <- i
  }
  return(split_lesion)
}

# ## Split confluent lesions
# split_confluent_new <- function(i, labeled_image, lesion_center_image) {
#   centers_in_label <- lesion_center_image[labeled_image == i]
#   n_centers <- unique(centers_in_label[centers_in_label != 0])
#   
#   if (length(n_centers) == 0) {
#     return(labeled_image == i)
#   }
#   
#   lesion <- labeled_image == i
#   split_lesion <- getleslabels(
#     lesion,
#     lesion_center_image * lesion
#   )
#   s <- unique(split_lesion[split_lesion != 0])
#   for (i in 1:length(s)) {
#     split_lesion[split_lesion == s[i]] <- i
#   }
#   return(split_lesion)
# }

split_confluent_new = function(label,i, conf_split){
  lesion = label == i
  voxel_table = table(conf_split * lesion)
  uniqueness = length(voxel_table) -1
  if(uniqueness > 1){
    split_lesion = getleslabels(lesion, conf_split * lesion)
    v_t = table(split_lesion)
    s = as.integer(names(v_t))[2:length(v_t)]
    split_lesion = relabel(split_lesion, s)
  }else{split_lesion = lesion}
  return(split_lesion)
}


## Label Lesions
label_lesion <- function(mimosa_mask, prob_map, mincluster = 100) {
  labeled_img <- label_mask(mimosa_mask == 1)
  size_control <- table(labeled_img)
  size_control <- size_control[size_control > mincluster]
  if (length(size_control) == 1) {
    zero_mask <- antsImageClone(oro2ants(mimosa_mask))
    zero_mask[zero_mask == 1] <- 0
    return(zero_mask)
  }
  lesion_count <- seq(1, length(size_control) - 1)
  les_split <- lesioncenters(
    probmap = prob_map, binmap = mimosa_mask,
    c3d = F, minCenterSize = mincluster / 10, radius = 1,
    parallel = F, cores = 2
  )$lesioncenters
  subimg <- lapply(lesion_count, split_confluent_new,
                   label = labeled_img, conf_split = les_split
  )
  
  for (i in 1:length(subimg)) {
    mask <- subimg[[i]]
    if (i == 1) {
      ct <- max(subimg[[1]])
      sum_mask <- mask
      next
    } else {
      add <- (mask > 0) * ct
      mask <- mask + add
      ct <- max(mask)
      sum_mask <- sum_mask + mask
    }
  }
  return(sum_mask)
}

#' @title MS Lesion Center Identification + Lesion Labelling
#' @description This function takes in a lesion probability map, a lesion segmentation mask, and a T2*-phase image for a single subject and identifies + classifies lesions as PRL or not.
#' @param probmap Lesion probability map. We recommend using lesion segmentation algorithm MIMoSA.
#' @param lesmask Lesion segmentation mask. Given a probability threshold, automatically binarizes lesion probability map into a segmentation mask.
#' @export
#' @import lesiontools
#' @import Rfast
#' @return A list with two NIfTI files: one with the identified lesion centers, and one with labels for each identified lesion.

lesion_identification <- function(probmap, lesmask) {
  ## get lesion centers ##
  lescents_obj <- lesioncenters(
    probmap = probmap,
    binmap = lesmask,
    c3d = F,
    radius = 2,
    parallel = F
  ) # get centers of dilated mask
  lescents_img <- lescents_obj$lesioncenters # get centers of dilated mask as a nifti
  
  ## get lesion labels ##
  leslabels <- getleslabels(lesmask = lesmask, centers = lescents_img)
  
  # subset out small lesions
  leslabels_big <- leslabels
  for (i in 1:max(leslabels)) {
    if (sum(leslabels == i) < 100) {
      leslabels_big[leslabels_big == i] <- 0
    }
  }
  
  # relabel lesions
  leslabels.names <- names(table(leslabels_big))
  numles <- length(leslabels.names) - 1
  if (leslabels.names[length(leslabels.names)] != numles) {
    for (i in 2:(numles + 1)) {
      leslabels_big[leslabels_big == leslabels.names[i]] <- (i - 1)
    }
  }
  
  return(list(lescents = lescents_img, leslabels = leslabels_big))
}

#' @title Distance to Mask Boundary
#' @description This function finds the distance of each voxel to the nearest boundary in a given mask.
#' @param mask a 3D array or image of class \code{nifti}, containing a binary mask where 1 represents structure.
#'
#' @return A new image in which voxels have been assigned their distance to the nearest boundary.
#' @examples \dontrun{
#' library(neurobase)
#' lesion.mask <- readnii("path/to/mask")
#' dtb <- dtboundary(mask = lesion.mask)
#' }
#' @export
dtboundary <- function(mask) {
  get.d.boundary.exact.balloon <- function(v, mask, d.max = 30) {
    if (mask[v[1], v[2], v[3]] == 0) print("ERROR! - voxel outside of mask...")
    inf <- 1000
    balloon.empty <- TRUE
    r <- 1
    # expand balloon
    while (balloon.empty) {
      balloon <- 1 - mask[(v[1] - r):(v[1] + r), (v[2] - r):(v[2] + r), (v[3] - r):(v[3] + r)]
      # If balloon had reached edge
      if (sum(balloon > 0)) {
        which.outside <- which(balloon > 0, arr.ind = TRUE)
        d.out <- min(sqrt((which.outside[, 1] - (r + 1))^2 + (which.outside[, 2] - (r + 1))^2 + (which.outside[, 3] - (r + 1))^2))
        balloon.empty <- FALSE
      } else {
        if (r <= d.max) {
          r <- r + 1
        } else {
          balloon.empty <- FALSE
          d.out <- inf
        }
      }
    }
    return(d.out)
  }
  which.mask.arrind <- which(mask > 0, arr.ind = TRUE)
  # For each voxel in the mask
  min.d <- rep(0, dim(which.mask.arrind)[1])
  for (i in 1:(dim(which.mask.arrind)[1])) {
    # Get minimum distance to boundary
    min.d[i] <- get.d.boundary.exact.balloon(which.mask.arrind[i, ], mask)
  }
  mask[mask > 0] <- min.d
  return(mask)
}

#' @title Central Vein Detection
#' @description This function obtains the probability that each lesion in a subject's deep white-matter has a central vein.
#' @param epi a T2*-EPI volume of class \code{nifti}.
#' @param t1 a T1-weighted volume of class \code{nifti}.
#' @param flair a T2-FLAIR volume of class \code{nifti}.
#' @param probmap an image of class \code{nifti}, containing the probability that each voxel
#' is a lesion voxel.
#' If a probability map is not included, the MIMoSA model will be applied (Valcarcel et al., 2017).
#' @param binmap a \code{nifti} mask in which voxels are classified as either lesion voxels
#' or not lesion voxels.
#' Note that mask should be in the same space as the probmap volume.
#' @param parallel is a logical value that indicates whether the user's computer
#' is Linux or Unix (i.e. macOS), and should run the code in parallel.
#' @param cores if parallel = TRUE, cores is an integer value that indicates how many cores
#' the function should be run on.
#' @param skullstripped a logical value reflecting whether or not the images have already been skull-stripped.
#' @param biascorrected a logical value reflecting whether or not the images have already been bias-corrected.
#' @param c3d a logical value reflecting whether or not the Convert3D imaging toolbox is installed.
#'
#' @importFrom ANTsRCore labelClusters
#' @importFrom neurobase niftiarr
#' @import mimosa
#' @importFrom extrantsr ants2oro oro2ants bias_correct registration fslbet_robust
#' @importFrom stats predict
#' @importFrom fslr fslsmooth fast fslerode
#' @return A list containing candidate.lesions (a nifti file with labeled lesions evaluated for CVS),
#' cvs.probmap (a nifti file in which candidate lesions are labeled with their CVS probability), and
#' cvs.biomarker (a numeric value representing the average CVS probability of a subject's lesions).
#' @examples \dontrun{
#' library(neurobase)
#' epi <- readnii("path/to/epi")
#' flair <- readnii("path/to/flair")
#' t1 <- readnii("path/to/t1")
#' cvs <- centralveins(
#'   epi = epi, t1 = t1, flair = flair,
#'   parallel = TRUE, cores = 4, c3d = T
#' )
#' }
#' @export
centralveins <- function(epi, t1, flair, mask,
                         probmap = NULL, binmap = NULL,
                         parallel = F, cores = 2,
                         skullstripped = F, biascorrected = F, registered = F,
                         c3d = F, use_fast = T) {
  if (biascorrected == F) {
    epi <- bias_correct(epi, correction = "N4", reorient = F)
    t1 <- bias_correct(t1, correction = "N4", reorient = F)
    flair <- bias_correct(flair, correction = "N4", reorient = F)
  }
  if (registered == F) {
    flair <- registration(
      filename = flair, template.file = t1, typeofTransform = "Rigid",
      remove.warp = FALSE, outprefix = "fun"
    )
    flair <- flair$outfile
  }
  if (skullstripped == F) {
    t1_ss <- fslbet_robust(t1, correct = F)
    epi_ss <- fslbet_robust(epi, correct = F)
    flair_ss <- flair
    flair_ss[t1_ss == 0] <- 0
  } else {
    t1_ss <- t1
    epi_ss <- epi
    flair_ss <- flair
  }
  if (is.null(probmap)) {
    mimosa_data <- mimosa_data(
      brain_mask = mask,
      FLAIR = flair_ss,
      T1 = t1_ss,
      normalize = "Z",
      cores = cores,
      verbose = TRUE
    )
    mimosa_df <- mimosa_data$mimosa_dataframe
    mimosa_cm <- mimosa_data$top_voxels
    rm(mimosa_data)
    
    predictions <- predict(mimosa::mimosa_model_No_PD_T2,
                           newdata = mimosa_df, type = "response"
    )
    probmap <- niftiarr(mask, 0)
    probmap[mimosa_cm == 1] <- predictions
    probmap <- fslsmooth(probmap,
                         sigma = 1.25, mask = mask,
                         retimg = TRUE, smooth_mask = TRUE
    )
  }
  if (is.null(binmap)) {
    binmap <- probmap
    binmap[probmap >= 0.3] <- 1
    binmap[probmap < 0.3] <- 0
  }
  if (sum(binmap) == 0) {
    warning("No lesions detected")
    return(NULL)
  }
  
  frangi <- frangi(
    image = epi_ss, mask = epi_ss != 0,
    parallel = parallel, cores = cores, c3d = c3d
  )
  frangi[frangi < 0] <- 0
  if (registered == F) {
    regs <- labelreg(epi, t1, frangi)
    epi_t1 <- regs$imagereg
    frangi_t1 <- regs$labelreg
  } else {
    epi_t1 <- epi
    frangi_t1 <- frangi
  }
  
  les <- lesioncenters(probmap, binmap,
                       parallel = parallel,
                       cores = cores, c3d = c3d)
  
  csf <- fast(t1, opts = "--nobias") # Doesn't work in my testing (FH)
  # csf <- fuzzySpatialCMeansSegmentation(oro2ants(t1),
  #                                       mask = oro2ants(mask),
  #                                       numberOfClusters = 3
  # )$segmentationImage # Substitute for fast (1 = CSF)
  csf[csf != 1] <- 0
  csf <- ants2oro(labelClusters(csf, minClusterSize = 300))
  csf[csf > 0] <- 1
  csf <- (csf != 1)
  csf <- fslerode(csf, kopts = paste("-kernel boxv", 3), verbose = TRUE)
  csf <- (csf == 0) # End up with mask of large CSF clusters that are dilated
  
  labels <- les$lesioncenters # Already labeled...? FH
  # labels <- ants2oro(labelClusters(oro2ants(les$lesioncenters), minClusterSize = 27))
  if (sum(labels) == 0) {
    warning("No lesions detected")
    return(NULL)
  }
  for (j in 1:max(labels)) {
    if (sum(csf[labels == j]) > 0) {
      labels[labels == j] <- 0
    }
  }
  
  les <- labels > 0
  if (sum(les) == 0) {
    warning("No lesions detected")
    return(NULL)
  }
  dtb <- dtboundary(les)
  
  labels <- ants2oro(labelClusters(oro2ants(les), minClusterSize = 27))
  probles <- labels
  avprob <- NULL
  maxles <- max(labels)
  for (j in 1:maxles) {
    frangsub <- frangi[labels == j]
    centsub <- dtb[labels == j]
    coords <- which(labels == j, arr.ind = T)
    prod <- frangsub * centsub
    score <- sum(prod)
    nullscores <- NULL
    for (k in 1:1000) {
      samp <- sample(1:length(centsub))
      centsamp <- centsub[samp]
      coordsamp <- coords[samp, ]
      sampprod <- frangsub * centsamp
      sampscore <- sum(sampprod)
      nullscores <- c(nullscores, sampscore)
    }
    lesprob <- sum(nullscores < score) / length(nullscores)
    avprob <- c(avprob, lesprob)
    probles[labels == j] <- lesprob
    
    print(paste0("Done with lesion ", j, " of ", maxles))
  }
  
  return(list(candidate.lesions = labels, 
              cvs.probmap = probles, 
              cvs.biomarker = mean(avprob)))
}

#' @title Frangi Vesselness Filter
#' @description This function returns a vesselness map for a 3D array or NIfTI volume. This vesselness measure is based on the method described in Frangi et al., (1998).
#' @param image a 3D array or image of class \code{nifti}
#' @param mask an array or \code{nifti} mask of voxels for which vesselness will be calculated,
#' with more selective masking improving speed significantly.
#' Note that mask should be in the same space as the image volume
#' @param radius an integer specifying radius of the neighborhood (in voxels) for which the vesselness should be calculated.
#' Note that this value essentially serves as the scale of the vessel objects
#' @param color a string specifying whether vessels will appear darker ("dark") or brighter ("bright") than their surroundings
#' @param parallel is a logical value that indicates whether the user's computer
#' is Linux or Unix (i.e. macOS), and should run the code in parallel
#' @param cores if parallel = TRUE, cores is an integer value that indicates how many cores
#' the function should be run on
#' @param c3d a logical value reflecting whether or not the Convert3D imaging toolbox is installed.
#' @param min.scale if c3d==T, the minimum scale in mm of the structures being found.
#' @param max.scale if c3d==T, the maximum scale in mm of the structures being found.
#'
#' @importFrom neurobase readnii writenii
#' @return A 3D volume of the Frangi vesselness scores.
#' @examples \dontrun{
#' library(neurobase)
#' epi <- readnii("path/to/epi")
#' mask <- epi != 0
#' veins <- frangi(
#'   image = epi, mask = mask, radius = 1,
#'   color = "dark", parallel = TRUE, cores = 4
#' )
#' }
#' @export
#' @references A.F. Frangi, W.J. Niessen, K.L. Vincken, M.A. Viergever (1998). Multiscale vessel enhancement filtering. In Medical Image Computing and Computer-Assisted Intervention - MICCAI'98, W.M. Wells, A. Colchester and S.L. Delp (Eds.), Lecture Notes in Computer Science, vol. 1496 - Springer Verlag, Berlin, Germany, pp. 130-137.
frangi <- function(image, mask, radius = 1, color = "dark",
                   parallel = FALSE, cores = 2, c3d = F,
                   min.scale = 0.5, max.scale = 0.5) {
  if (c3d == F) {
    eigvals <- hessian(image, mask, radius, parallel, cores)
    
    print("Calculating vesselness measure")
    l1 <- eigvals$eigval1
    l2 <- eigvals$eigval2
    l3 <- eigvals$eigval3
    l1 <- as.vector(l1[mask == 1])
    l2 <- as.vector(l2[mask == 1])
    l3 <- as.vector(l3[mask == 1])
    rm(eigvals)
    
    al1 <- abs(l1)
    al2 <- abs(l2)
    al3 <- abs(l3)
    
    Ra <- al2 / al3
    Ra[!is.finite(Ra)] <- 0
    Rb <- al1 / sqrt(al2 * al3)
    Rb[!is.finite(Rb)] <- 0
    
    S <- sqrt(al1^2 + al2^2 + al3^2)
    A <- 2 * (.5^2)
    B <- 2 * (.5^2)
    C <- 2 * (.5 * max(S))^2
    
    rm(al1, al2, al3)
    
    eA <- 1 - exp(-(Ra^2) / A)
    eB <- exp(-(Rb^2) / B)
    eC <- 1 - exp(-(S^2) / C)
    
    rm(Ra, Rb, S, A, B, C)
    
    vness <- eA * eB * eC
    
    rm(eA, eB, eC)
    
    if (color == "dark") {
      vness[l2 < 0 | l3 < 0] <- 0
      vness[!is.finite(vness)] <- 0
    } else if (color == "bright") {
      vness[l2 > 0 | l3 > 0] <- 0
      vness[!is.finite(vness)] <- 0
    }
    
    image[mask == 1] <- vness
    return(image)
  } else {
    tempinv <- tempfile(pattern = "file", tmpdir = tempdir(), fileext = ".nii.gz")
    if (color == "dark") {
      writenii(-1 * image, tempinv)
    } else {
      writenii(image, tempinv)
    }
    tempvein <- tempfile(pattern = "file", tmpdir = tempdir(), fileext = ".nii.gz")
    system(paste0("c3d ", tempinv, " -hessobj 1 ", min.scale, " ", max.scale, " -oo ", tempvein))
    
    veinmask <- readnii(tempvein)
    return(veinmask)
  }
}

# only takes in disc = F
findprls <- function(lesmask, phasefile, pretrainedmodel) {
  if (sum(lesmask) == 0) {
    return(NULL)
  }
  # run ria feature extraction
  ria.obj <- extract_ria(phasefile = phasefile, leslabels = lesmask, disc = F)
  ria.df <- as.data.frame(ria.obj)
  
  # rename variable names to match the ones saved in the model
  names.temp <- as.character(names(ria.df))
  # names.temp = sapply(X = names.temp, function(X){gsub(pattern = "orig.", replacement = "", x = X)})
  names.temp <- sapply(X = names.temp, function(X) {
    gsub(pattern = "%", replacement = ".", x = X)
  })
  names(ria.df) <- names.temp
  
  return(list(leslabels = lesmask, ria.df = ria.df, preds = stats::predict(pretrainedmodel, newdata = ria.df, type = "prob")))
}
#'
#' #' @title MS Lesion Identification and PRL Classification with Pre-trained Model
#' #' @description This function takes in a lesion probability map, a lesion segmentation mask, and a T2*-phase image for a single subject and identifies + classifies lesions as PRL or not.
#' #' @param probmap Lesion probability map. We recommend using lesion segmentation algorithm MIMoSA.
#' #' @param lesmask Lesion segmentation mask. Given a probability threshold, automatically binarizes lesion probability map into a segmentation mask.
#' #' @param phasefile Location of the T2*-phase image
#' #' @param disc Calculate discretized versions of first order radiomic features?
#' #' @export
#' #' @import lesiontools
#' #' @import RIA
#' #' @import Rfast
#' #' @importFrom stats predict
#' #' @return A list with 3 objects: a NIfTI file of a lesion label map, a dataframe containing all radiomic features, and a vector containing lesion-wise probability of being a PRL
#'
#' findprls = function(probmap, lesmask, phasefile, disc = T, pretrainedmodel = NULL){
#'   # run lesion identification code
#'   lesident <- lesion_identification(probmap = probmap,
#'                                     lesmask = lesmask)
#'
#'   leslabels.out = lesident$leslabels
#'   print("lesion identification done!")
#'
#'   # run ria feature extraction
#'   ria.obj = extract_ria(phasefile = phasefile, leslabels = leslabels.out, disc = disc)
#'   ria.df = as.data.frame(ria.obj)
#'   print("radiomic feature extraction done!")
#'
#'   # rename variable names to match the ones saved in the model
#'   names.temp = as.character(names(ria.df))
#'   #names.temp = sapply(X = names.temp, function(X){gsub(pattern = "orig.", replacement = "", x = X)})
#'   names.temp = sapply(X = names.temp, function(X){gsub(pattern = "%", replacement = ".", x = X)})
#'   names(ria.df) = names.temp
#'
#'   if (is.null(pretrainedmodel)) {
#'     if(disc == T){
#'       pretrainedmodel = prlr::prlmodel_orig_disc8_disc64
#'     } else{
#'       pretrainedmodel = prlr::prlmodel_orig
#'     }
#'   }
#'
#'   return(list(leslabels = leslabels.out, ria.df = ria.df, preds = stats::predict(pretrainedmodel, newdata = ria.df, type = "prob")))
#'
#' }

#' @title Radiomic Feature Extraction
#' @description This function extracts radiomic features from a given T2*-phase image and a lesion label map
#' @param phasefile Location of the T2*-phase image
#' @param leslabels Lesion label map, generated by our lesion identification method
#' @param disc Calculate discretized versions of first order radiomic features?
#' @import RIA
#' @import neurobase
#' @return A dataframe containing radiomic features for each identified lesion

extract_ria <- function(phasefile, leslabels, disc = T) {
  numles <- max(leslabels)
  
  return.stats <- list()
  for (i in 1:numles) {
    RIA.image <- RIA::load_nifti(
      filename = phasefile,
      crop_in = FALSE,
      switch_z = FALSE,
      reorient_in = FALSE,
      replace_in = FALSE,
      reorient = FALSE, verbose_in = FALSE
    )
    image <- neurobase::readnii(phasefile)
    image[leslabels != i] <- NA
    RIA.image$data$orig <- image
    
    first.order.orig <- RIA:::first_order(RIA.image)
    
    if (disc == T) {
      RIA.image <- RIA:::discretize(RIA.image, bins_in = c(8, 64), equal_prob = TRUE)
      first.order.disc <- RIA:::first_order(RIA.image, use_type = "discretized")
      
      stats.orig <- unlist(first.order.orig$stat_fo)
      stats.disc <- unlist(first.order.disc$stat_fo)
      return.stats[[i]] <- c(stats.orig, stats.disc)
    } else {
      stats.orig <- unlist(first.order.orig$stat_fo)
      return.stats[[i]] <- stats.orig
    }
  }
  all.stats <- do.call(rbind, return.stats)
  return(all.stats)
}
