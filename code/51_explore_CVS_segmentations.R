library(pdist)
library(RcppHungarian)

proj_dir <- "/home/fengling/Documents/prl"
out_dir <- "/home/fengling/Documents/prl/data/CVS_combined"
key <- read_xlsx(file.path(proj_dir, "data/csvs/Key.xlsx"))
names(key) <- c("random_id", "patient_id", "pre_post")

randomized_path <- file.path(proj_dir, "data/csvs/randomized_CVS.xlsx")
randomized_sheets <- excel_sheets(randomized_path) %>% as.numeric()
remaining_path <- file.path(proj_dir, "data/csvs/remaining_CVS.xlsx")
remaining_sheets <- excel_sheets(remaining_path)

randomized_list <- lapply(6:length(excel_sheets(randomized_path)), # First 5 sheets were for analysis
                          function(i) {
                            process_sheet(randomized_path, i, keep_all = T)
                          })
remaining_list <- lapply(1:length(excel_sheets(remaining_path)), 
                         function(i) {
                           process_sheet(remaining_path, i, keep_all = T)
                         })
########################
names(randomized_list) <- paste(key_id_pair[, 2], key_id_pair[, 3]) %>% str_trim()
for (i in 1:length(remaining_sheets)) {
  remaining_sheets[i] <- ifelse(grepl("Pre", remaining_sheets[i]), remaining_sheets[i], paste(remaining_sheets[i], "Post"))
}
names(remaining_list) <- remaining_sheets

combine_dataframes <- function(df_list) {
  names_df <- str_split_fixed(names(df_list), " ", 2)
  unique_names <- unique(names_df[, 1])
  combined_dfs <- list()
  
  counter <- 0
  for (name in unique_names) {
    df_subset <- df_list[grepl(name, names(df_list))]
    status <- grepl("Pre", names(df_subset))
    if (length(df_subset) == 1) {
      counter <- counter + 1
      print(counter)
      print(name)
      next
    }
    for (i in 1:length(df_subset)) {
      df_subset[[i]] <- df_subset[[i]] %>% mutate(status = status[i])
    }
    combined_dfs[[name]] <- do.call(rbind, df_subset)
  }
  
  return(combined_dfs)
}

combined_randomized_list <- combine_dataframes(randomized_list)
combined_remaining_list <- combine_dataframes(remaining_list)

assess_disparity <- function(randomized_list) {
  n <- length(randomized_list)
  neg_to_pos <- numeric(n)
  pos_to_neg <- numeric(n)
  sum_pre <- numeric(n)
  sum_post <- numeric(n)
  for (i in 1:n) {
    tmp_list <- randomized_list[[i]]
    pre <- tmp_list %>% filter(status == "TRUE")
    post <- tmp_list %>% filter(status == "FALSE")
    distances <- as.matrix(pdist(pre[, 5:7], post[, 5:7]))
    indices <- which(distances < 5, arr.ind = TRUE)
    
    neg_to_pos_counter <- 0
    pos_to_neg_counter <- 0
    for (j in 1:nrow(indices)) {
      pre_cvs <- pre[indices[j, 1], 3]
      post_cvs <- post[indices[j, 2], 3]
      if (pre_cvs == "Excluded") {
        pre_cvs <- "Negative"
      }
      if (post_cvs == "Excluded") {
        post_cvs <- "Negative"
      }
      if (pre_cvs != post_cvs) {
        if (pre_cvs == "Positive")
          pos_to_neg_counter <- pos_to_neg_counter + 1
        if (pre_cvs == "Negative")
          neg_to_pos_counter <- neg_to_pos_counter + 1
      }
    }
    
    neg_to_pos[i] <- neg_to_pos_counter
    pos_to_neg[i] <- pos_to_neg_counter
    sum_pre[i] <- sum(pre$cvs_status == "Positive")
    sum_post[i] <- sum(post$cvs_status == "Positive")
  }
  
  return(as.data.frame(cbind(neg_to_pos, pos_to_neg, sum_pre, sum_post)))
}

combined_list <- append(combined_randomized_list, combined_remaining_list)
disparity <- assess_disparity(combined_list)

tmp <- do.call()

tmp <- c(paste(key_id_pair[, 2], key_id_pair[, 3]) %>% str_trim(), remaining_sheets)
