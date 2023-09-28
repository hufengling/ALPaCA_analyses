files <- read.csv("/home/fengling/Documents/prl/file_names.txt")
for (i in 1:length(files)) {
  tmp <- gsub("/home/fengling/Documents/prl/data/processed", 
              "/project/CRC_documents/APRL/Zheng_data", 
              files[i, 1])
  print(tmp)
  file.copy(from = tmp, to = files[i, 1])
}