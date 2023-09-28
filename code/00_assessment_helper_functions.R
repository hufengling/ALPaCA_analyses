concatenate_pred_csvs <- function(path, split_id, cv_ids) {
  csv_list <- lapply(list.files(file.path(path, paste0("split_", split_id)),
                                full.names = TRUE), read.csv)
  all_df <- do.call(rbind, csv_list)
  all_df <- all_df %>% 
    mutate(subject_id = sub(".*/", "", subject)) %>% 
    left_join(cv_ids, by = "subject_id") %>% 
    select(-X.y, -subject_id) %>% 
    rename(X = X.x)
  
  train <- all_df %>% filter(cv_index != split_id)
  test <- all_df %>% filter(cv_index == split_id)
  
  return(list(train = train,
              test = test))
}

preprocess_preds <- function(preds_df, competitor_preds) {
  preds_df <- preds_df %>%
    rename(lesion_id = X, lesion_pred = X0, prl_pred = X1, cvs_pred = X2,
           lesion_var = X3, prl_var = X4, cvs_var = X5,
           lesion_true = X6, prl_true = X7, cvs_true = X8,
           lesion_missing = X9, prl_missing = X10, cvs_missing = X11) %>% 
    mutate(subject = str_sub(subject, -6),
           lesion_id = lesion_id + 1) %>% 
    mutate(lesion_true = ifelse(lesion_missing == 0, NA, lesion_true),
           prl_true = ifelse(prl_missing == 0, NA, prl_true),
           cvs_true = ifelse(cvs_missing == 0, NA, cvs_true))
  
  preds_df <- preds_df %>% 
    left_join(competitor_preds, by = c("subject", "lesion_id"))
  return(preds_df)
}

check_total_lesion_number <- function(train_pred, test_pred, competitor_preds) {
  nrow(train_pred) + nrow(test_pred) == nrow(competitor_preds)
}

mlapply <- function(func, list1, list2) {
  tmp_list <- vector("list", length(list1))
  func <- match.fun(func)
  for (i in 1:length(list1)) {
    tmp_list[[i]] <- func(list1[[i]], list2[[i]])
  }
  
  return(tmp_list)
}

get_threshold <- function(roc_obj) {
  youd_ind <- which.max(roc_obj$sensitivities + roc_obj$specificities)
  youdens <- roc_obj$thresholds[youd_ind]
  print(paste0("Optimal: Sens = ", roc_obj$sensitivities[youd_ind], 
               " Spec = ", roc_obj$specificities[youd_ind]))
  spec_ind <- which.max(roc_obj$sensitivities + 3 * roc_obj$specificities)
  spec <- roc_obj$thresholds[spec_ind]
  print(paste0("Spec: Sens = ", roc_obj$sensitivities[spec_ind], 
               " Spec = ", roc_obj$specificities[spec_ind]))
  sens_ind <- which.max(3 * roc_obj$sensitivities + roc_obj$specificities)
  sens <- roc_obj$thresholds[sens_ind]
  print(paste0("Sens: Sens = ", roc_obj$sensitivities[sens_ind], 
               " Spec = ", roc_obj$specificities[sens_ind]))
  return(c(youdens, spec, sens))
}

get_spec_sens <- function(roc_obj, threshold) {
  threshold <- as.numeric(threshold)
  return(c(mean(roc_obj$controls < threshold),
           mean(roc_obj$cases >= threshold)))
}

get_confusion <- function(roc_obj, threshold, ...) {
  threshold <- as.numeric(threshold)
  reference <- as.factor(as.numeric(unname(roc_obj$response) == 1))
  prediction <- as.factor(as.numeric(roc_obj$predictor > threshold))
  return(caret::confusionMatrix(prediction, reference, positive = "1", ...))
}

get_disparate_preds <- function(preds_df, train_roc_obj) {
  lesion_thresh <- as.numeric(coords(train_roc_obj$les_roc, "best")[1])
  prl_thresh <- as.numeric(coords(train_roc_obj$prl_roc, "best")[1])
  cvs_thresh <- as.numeric(coords(train_roc_obj$cvs_roc, "best")[1])
  prl_disparity <- (preds_df$lesion_pred < lesion_thresh) & (preds_df$prl_pred > prl_thresh)
  cvs_disparity <- (preds_df$lesion_pred < lesion_thresh) & (preds_df$cvs_pred > cvs_thresh)
  
  vec <- c(sum(prl_disparity), 
           sum(!is.na(preds_df$lesion_true[prl_disparity])),
           mean(preds_df$lesion_true[prl_disparity], na.rm = TRUE),
           sum(!is.na(preds_df$prl_true[prl_disparity])),
           mean(preds_df$prl_true[prl_disparity], na.rm = TRUE),
           sum(cvs_disparity),
           sum(!is.na(preds_df$lesion_true[cvs_disparity])),
           mean(preds_df$lesion_true[cvs_disparity], na.rm = TRUE),
           sum(!is.na(preds_df$cvs_true[cvs_disparity])),
           mean(preds_df$cvs_true[cvs_disparity], na.rm = TRUE))
  vec
}

change_disparate_preds <-  function(preds_df, train_roc_obj) {
  lesion_thresh <- as.numeric(coords(train_roc_obj$les_roc, "best")[1])
  prl_thresh <- as.numeric(coords(train_roc_obj$prl_roc, "best")[1])
  cvs_thresh <- as.numeric(coords(train_roc_obj$cvs_roc, "best")[1])
  prl_disparity <- (preds_df$lesion_pred < lesion_thresh) & (preds_df$prl_pred > prl_thresh)
  cvs_disparity <- (preds_df$lesion_pred < lesion_thresh) & (preds_df$cvs_pred > cvs_thresh)
  
  preds_df$prl_pred[prl_disparity] <- 0
  preds_df$cvs_pred[cvs_disparity] <- 0
  return(preds_df)
}

get_train_roc <- function(preds_df) {
  les_train_roc <- roc(lesion_true ~ lesion_pred, data = preds_df)
  prl_train_roc <- roc(prl_true ~ prl_pred, data = preds_df)
  cvs_train_roc <- roc(cvs_true ~ cvs_pred, data = preds_df)
  
  mimosa_train_roc <- roc(lesion_true ~ mimosa_pred, data = preds_df)
  aprl_train_roc <- roc(prl_true ~ aprl_pred, data = preds_df)
  acvs_train_roc <- roc(cvs_true ~ acvs_pred, data = preds_df)
  
  
  return(list(les_roc = les_train_roc,
              les_conf = get_confusion(les_train_roc, coords(les_train_roc, "best")[1]),
              prl_roc = prl_train_roc,
              prl_conf = get_confusion(prl_train_roc, coords(prl_train_roc, "best")[1]),
              cvs_roc = cvs_train_roc,
              cvs_conf = get_confusion(cvs_train_roc, coords(cvs_train_roc, "best")[1]),
              mimosa_roc = mimosa_train_roc,
              mimosa_conf_opt = get_confusion(mimosa_train_roc, coords(mimosa_train_roc, "best")[1]),
              aprl_roc = aprl_train_roc,
              aprl_conf_opt = get_confusion(aprl_train_roc, coords(aprl_train_roc, "best")[1]),
              acvs_roc = acvs_train_roc,
              acvs_conf_opt = get_confusion(acvs_train_roc, coords(acvs_train_roc, "best")[1]),
              plot(les_train_roc, print.auc = TRUE),
              plot(mimosa_train_roc, print.auc = TRUE),
              plot(prl_train_roc, print.auc = TRUE),
              plot(aprl_train_roc, print.auc = TRUE),
              plot(cvs_train_roc, print.auc = TRUE),
              plot(acvs_train_roc, print.auc = TRUE)
  ))
}

get_test_roc <- function(preds_df, train_roc_obj) {
  if (all(c(0, 1) %in% preds_df$lesion_true)) {
    les_test_roc <- roc(lesion_true ~ lesion_pred, data = preds_df)
    les_plot = plot(les_test_roc, print.auc = TRUE)
    les_conf = get_confusion(les_test_roc, coords(train_roc_obj$les_roc, "best")[1])
    
    mimosa_test_roc <- roc(lesion_true ~ mimosa_pred, data = preds_df)
    mimosa_plot = plot(mimosa_test_roc, print.auc = TRUE)
    mimosa_conf_opt = get_confusion(mimosa_test_roc, coords(train_roc_obj$mimosa_roc, "best")[1])
    mimosa_conf_02 = get_confusion(mimosa_test_roc, 0.2)
  } else {
    les_test_roc = NULL
    les_plot = NULL
    les_conf = NULL
    
    mimosa_test_roc = NULL
    mimosa_plot = NULL
    mimosa_conf_opt = NULL
    mimosa_conf_02 = NULL
  }
  
  if (all(c(0, 1) %in% preds_df$prl_true)) {
    prl_test_roc <- roc(prl_true ~ prl_pred, data = preds_df)
    prl_plot = plot(prl_test_roc, print.auc = TRUE)
    prl_conf = get_confusion(prl_test_roc, coords(train_roc_obj$prl_roc, "best")[1])
    
    aprl_test_roc <- roc(prl_true ~ aprl_pred, data = preds_df)
    aprl_plot = plot(aprl_test_roc, print.auc = TRUE)
    aprl_conf_opt = get_confusion(aprl_test_roc, coords(train_roc_obj$aprl_roc, "best")[1])
    aprl_conf_02 = get_confusion(aprl_test_roc, 0.5)
  } else {
    prl_test_roc = NULL
    prl_plot = NULL
    prl_conf = NULL
    
    aprl_test_roc = NULL
    aprl_plot = NULL
    aprl_conf_opt = NULL
    aprl_conf_02 = NULL
  }
  
  if (all(c(0, 1) %in% preds_df$cvs_true)) {
    cvs_test_roc <- roc(cvs_true ~ cvs_pred, data = preds_df)
    cvs_plot = plot(cvs_test_roc, print.auc = TRUE)
    cvs_conf = get_confusion(cvs_test_roc, coords(train_roc_obj$cvs_roc, "best")[1])
    
    acvs_test_roc <- roc(cvs_true ~ acvs_pred, data = preds_df)
    acvs_plot = plot(acvs_test_roc, print.auc = TRUE)
    acvs_conf_opt = get_confusion(acvs_test_roc, coords(train_roc_obj$acvs_roc, "best")[1])
    acvs_conf_02 = get_confusion(acvs_test_roc, 0.2)
  } else {
    cvs_test_roc = NULL
    cvs_plot = NULL
    cvs_conf = NULL
    
    acvs_test_roc = NULL
    acvs_plot = NULL
    acvs_conf_opt = NULL
    acvs_conf_02 = NULL
  }
  
  output <- list(les_roc = les_test_roc,
                 les_plot = les_plot,
                 les_conf = les_conf,
                 prl_roc = prl_test_roc,
                 prl_plot = prl_plot,
                 prl_conf = prl_conf,
                 cvs_roc = cvs_test_roc,
                 cvs_plot = cvs_plot,
                 cvs_conf = cvs_conf,
                 mimosa_roc = mimosa_test_roc,
                 mimosa_plot = mimosa_plot,
                 mimosa_conf_opt = mimosa_conf_opt,
                 mimosa_conf_02 = mimosa_conf_02,
                 aprl_roc = aprl_test_roc,
                 aprl_plot = aprl_plot,
                 aprl_conf_opt = aprl_conf_opt,
                 aprl_conf_02 = aprl_conf_02,
                 acvs_roc = acvs_test_roc,
                 acvs_plot = acvs_plot,
                 acvs_conf_opt = acvs_conf_opt,
                 acvs_conf_02 = acvs_conf_02)
  return(output)
}

extract_roc_acc <- function(roc_output) {
  output <- c(roc_output$les_roc$auc, roc_output$les_conf$overall["Accuracy"],
              roc_output$prl_roc$auc, roc_output$prl_conf$overall["Accuracy"],
              roc_output$cvs_roc$auc, roc_output$cvs_conf$overall["Accuracy"],
              roc_output$mimosa_roc$auc, 
              roc_output$mimosa_conf_opt$overall["Accuracy"],
              roc_output$mimosa_conf_02$overall["Accuracy"],
              roc_output$aprl_roc$auc, 
              roc_output$aprl_conf_opt$overall["Accuracy"],
              roc_output$aprl_conf_02$overall["Accuracy"],
              roc_output$acvs_roc$auc, 
              roc_output$acvs_conf_opt$overall["Accuracy"],
              roc_output$acvs_conf_02$overall["Accuracy"])
  full_output <- rep(NA, 15)
  full_output[!sapply(roc_output[c(1, 3, 4, 6, 7, 8, 10, 11, 13)], 
                      is.null)] <- output
  names(full_output) <- c("les_roc", "les_acc",
                          "prl_roc", "prl_acc",
                          "cvs_roc", "cvs_acc",
                          "mim_roc", "mim_acc_opt", "mim_acc_02",
                          "aprl_roc", "aprl_acc_opt", "aprl_acc_02",
                          "acvs_roc", "acvs_acc_opt", "acvs_acc_02")
  return(full_output)
}

get_prediction_labels <- function(preds_df, train_roc_obj) {
  lesion_thresh <- as.numeric(coords(train_roc_obj$les_roc, "best")[1])
  prl_thresh <- as.numeric(coords(train_roc_obj$prl_roc, "best")[1])
  cvs_thresh <- as.numeric(coords(train_roc_obj$cvs_roc, "best")[1])
  
  mimosa_thresh <- as.numeric(coords(train_roc_obj$mimosa_roc, "best")[1])
  aprl_thresh <- as.numeric(coords(train_roc_obj$aprl_roc, "best")[1])
  acvs_thresh <- as.numeric(coords(train_roc_obj$acvs_roc, "best")[1])
  
  preds_yn <- preds_df %>% 
    mutate(lesion_yn = lesion_pred >= lesion_thresh,
           prl_yn = prl_pred >= prl_thresh,
           cvs_yn = cvs_pred >= cvs_thresh,
           mimosa_yn = mimosa_pred >= mimosa_thresh,
           aprl_yn = aprl_pred >= aprl_thresh,
           acvs_yn = acvs_pred >= acvs_thresh) %>% 
    select(subject, lesion_id, lesion_true, prl_true, cvs_true,
           lesion_yn, prl_yn, cvs_yn,
           mimosa_yn, aprl_yn, acvs_yn)
  preds_yn[preds_yn == TRUE] <- 1
  preds_yn[preds_yn == FALSE] <- 0
  preds_yn
  # preds_df %>% 
  #   mutate(pred_lesion = case_when(
  #     lesion_true == 1 & lesion_pred < lesion_thresh ~ "FN",
  #     lesion_true == 1 & lesion_pred >= lesion_thresh ~ "TP",
  #     lesion_true == 0 & lesion_pred < lesion_thresh ~ "TN",
  #     lesion_true == 0 & lesion_pred >= lesion_thresh ~ "FP",
  #     is.na(lesion_true) & lesion_pred < lesion_thresh ~ "UN",
  #     is.na(lesion_true) & lesion_pred >= lesion_thresh ~ "UP"
  #   )) %>% 
  #   mutate(pred_prl = case_when(
  #     prl_true == 1 & prl_pred < prl_thresh ~ "FN",
  #     prl_true == 1 & prl_pred >= prl_thresh ~ "TP",
  #     prl_true == 0 & prl_pred < prl_thresh ~ "TN",
  #     prl_true == 0 & prl_pred >= prl_thresh ~ "FP",
  #     is.na(prl_true) & prl_pred < prl_thresh ~ "UN",
  #     is.na(prl_true) & prl_pred >= prl_thresh ~ "UP"
  #   )) %>% 
  #   mutate(pred_cvs = case_when(
  #     cvs_true == 1 & cvs_pred < cvs_thresh ~ "FN",
  #     cvs_true == 1 & cvs_pred >= cvs_thresh ~ "TP",
  #     cvs_true == 0 & cvs_pred < cvs_thresh ~ "TN",
  #     cvs_true == 0 & cvs_pred >= cvs_thresh ~ "FP",
  #     is.na(cvs_true) & cvs_pred < cvs_thresh ~ "UN",
  #     is.na(cvs_true) & cvs_pred >= cvs_thresh ~ "UP"
  #   )) %>% 
  #   mutate(lesion_01 = lesion_pred >= lesion_thresh) %>% 
  #   mutate(prl_01 = prl_pred >= prl_thresh) %>% 
  #   mutate(cvs_01 = cvs_pred >= cvs_thresh)
}