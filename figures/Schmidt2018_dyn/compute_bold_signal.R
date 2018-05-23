library('neuRosim')
args <- commandArgs(trailingOnly=TRUE)
print(args)

x <- read.table(args[1])
d <- data.matrix(x)

T <- 100
it <- 0.001

out <- balloon(d, T, it)
write.table(out, args[2])