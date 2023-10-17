library(lesiontools)

cvs <- central_veins(epi, t1_n4, flair_n4, verbose = TRUE)

cvs <- central_veins(epi, t1_n4, flair_n4, verbose = FALSE)#,
                     prob_map = check_ants("./data/processed_05/01-005/prob.nii.gz"),
                     bin_map = check_ants("./data/processed_05/01-005/prob_30.nii.gz"), verbose = TRUE)

uni <- unique(tmp)
tmp_copy <- antsImageClone(tmp)
for (i in 1:length(uni)) {
  tmp_copy[tmp == uni[i]] <- i
}

subjects <- list.files(here("data/processed_05"), full.names = TRUE)
unique_list <- lapply(subjects, function(path) {
  tmp <- unique(check_ants(file.path(path, "lesion_labels.nii.gz")))
  print(tmp)
  tmp
})
