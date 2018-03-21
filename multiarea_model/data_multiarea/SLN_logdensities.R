library('aod')
source(paste(Sys.getenv('HOME'),'/model-june/data_multi_area/bbAlt.R', sep=""))
f <- file(paste(Sys.getenv('HOME'),'/model-june/data_multi_area/raw_data/RData_prepared_logdensities.txt', sep=""),'r')
x <- read.table(f)
close(f)


dens <- data.matrix(x)[,7]

m2.bb <- betabin(cbind(S, I) ~ dens , ~ 1, data = x, "probit", control = list(maxit = 100000))
h2.bb <- c(coef(m2.bb))

print(h2.bb)
