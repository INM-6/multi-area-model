library('aod')
args <- commandArgs(trailingOnly=TRUE)
source(paste(args,'multiarea_model/data_multiarea/bbAlt.R', sep=""))
f <- file(paste(args,'multiarea_model/data_multiarea/raw_data/RData_prepared_logdensities.txt', sep=""),'r')
x <- read.table(f)
close(f)


dens <- data.matrix(x)[,7]

m2.bb <- betabin(cbind(S, I) ~ dens , ~ 1, data = x, "probit", control = list(maxit = 100000))
h2.bb <- c(coef(m2.bb))

print(h2.bb)
