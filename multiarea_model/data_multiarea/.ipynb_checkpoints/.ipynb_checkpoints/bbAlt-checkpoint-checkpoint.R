# Script provided by Kenneth Knoblauch
# This code is based on functions from the aod package of R written by
# Matthieu Lesnoff and Renaud Lancelot
# (https://cran.r-project.org/web/packages/aod/index.html), published
# under the GPL-3 license
# (https://cran.r-project.org/web/licenses/GPL-3).

betabin <- function (formula, random, data = NULL, link = c("logit", "probit", "cloglog"), 
    phi.ini = NULL, warnings = FALSE, na.action = na.omit, fixpar = list(), 
    hessian = TRUE, control = list(maxit = 2000), ...) 
{
    CALL <- mf <- match.call(expand.dots = FALSE)
    tr <- function(string) gsub("^[[:space:]]+|[[:space:]]+$", 
        "", string)
    link <- match.arg(link)
    if (length(formula) != 3) 
        stop(paste(tr(deparse(formula)), collapse = " "), "is not a valid formula.")
    else if (substring(deparse(formula)[1], 1, 5) != "cbind") 
        stop(paste(tr(deparse(formula)), collapse = ""), " is not a valid formula.\n", 
            "The response must be a matrix of the form cbind(success, failure)")
    if (length(random) == 3) {
        form <- deparse(random)
        warning("The formula for phi (", form, ") contains a response which is ignored.")
        random <- random[-2]
    }
    explain <- as.character(attr(terms(random), "variables"))[-1]
    if (length(explain) > 1) {
        warning("The formula for phi contains several explanatory variables (", 
            paste(explain, collapse = ", "), ").\n", "Only the first one (", 
            explain[1], ") was considered.")
        explain <- explain[1]
    }
    gf3 <- if (length(explain) == 1) 
        paste(as.character(formula[3]), explain, sep = " + ")
    else as.character(formula[3])
    gf <- formula(paste(formula[2], "~", gf3))
    if (missing(data)) 
        data <- environment(gf)
    mb <- match(c("formula", "data", "na.action"), names(mf), 
        0)
    mfb <- mf[c(1, mb)]
    mfb$drop.unused.levels <- TRUE
    mfb[[1]] <- as.name("model.frame")
    names(mfb)[2] <- "formula"
    mfb <- eval(mfb, parent.frame())
    mt <- attr(mfb, "terms")
    modmatrix.b <- if (!is.empty.model(mt)) 
        model.matrix(mt, mfb)
    else matrix(, NROW(Y), 0)
    Y <- model.response(mfb, "numeric")
    weights <- model.weights(mfb)
    if (!is.null(weights) && any(weights < 0)) 
        stop("Negative wts not allowed")
    n <- rowSums(Y)
    y <- Y[, 1]
    if (any(n == 0)) 
        warning("The data set contains at least one line with weight = 0.\n")
    mr <- match(c("random", "data", "na.action"), names(mf), 
        0)
    mr <- mf[c(1, mr)]
    mr$drop.unused.levels <- TRUE
    mr[[1]] <- as.name("model.frame")
    names(mr)[2] <- "formula"
    mr <- eval(mr, parent.frame())
    if (length(explain) == 0) 
        modmatrix.phi <- model.matrix(object = ~1, data = mr)
    else {
        express <- paste("model.matrix(object = ~ -1 + ", explain, 
            ", data = mr", ", contrasts = list(", explain, " = 'contr.treatment'))", 
            sep = "")
        if (is.ordered(data[, match(explain, table = names(mr))])) 
            warning(explain, " is an ordered factor.\n", "Treatment contrast was used to build model matrix for phi.")
        modmatrix.phi <- eval(parse(text = express))
    }
    fam <- eval(parse(text = paste("binomial(link =", link, ")")))
    fm <- glm(formula = formula, family = fam, data = data, na.action = na.action)
    b <- coef(fm)
    if (any(is.na(b))) {
        print(nab <- b[is.na(b)])
        stop("Initial values for the fixed effects contain at least one missing value.")
    }
    nb.b <- ncol(modmatrix.b)
    nb.phi <- ncol(modmatrix.phi)
    if (!is.null(phi.ini) && !(phi.ini < 1 & phi.ini > 0)) 
        stop("phi.ini was set to ", phi.ini, ".\nphi.ini should verify 0 < phi.ini < 1")
    else if (is.null(phi.ini)) 
        phi.ini <- rep(0.1, nb.phi)
    param.ini <- c(b, phi.ini)
    if (!is.null(unlist(fixpar))) 
        param.ini[fixpar[[1]]] <- fixpar[[2]]
    minuslogL <- function(param) {
        if (!is.null(unlist(fixpar))) 
            param[fixpar[[1]]] <- fixpar[[2]]
        b <- param[1:nb.b]
        eta <- as.vector(modmatrix.b %*% b)
        p <- invlink(eta, type = link)
        phi <- as.vector(modmatrix.phi %*% param[(nb.b + 1):(nb.b + 
            nb.phi)])
        cnd <- phi == 0
        f1 <- dbinom(x = y[cnd], size = n[cnd], prob = p[cnd], 
            log = TRUE)
        n2 <- n[!cnd]
        y2 <- y[!cnd]
        p2 <- p[!cnd]
        phi2 <- phi[!cnd]
        f2 <- lchoose(n2, y2) + lbeta(p2 * (1 - phi2)/phi2 + 
            y2, (1 - p2) * (1 - phi2)/phi2 + n2 - y2) - lbeta(p2 * 
            (1 - phi2)/phi2, (1 - p2) * (1 - phi2)/phi2)
        fn <- sum(c(f1, f2))
        if (!is.finite(fn)) 
            fn <- -1e+20
        -fn
    }
    withWarnings <- function(expr) {
        myWarnings <- NULL
        wHandler <- function(w) {
            myWarnings <<- c(myWarnings, list(w))
            invokeRestart("muffleWarning")
        }
        val <- withCallingHandlers(expr, warning = wHandler)
        list(value = val, warnings = myWarnings)
    }
    reswarn <- withWarnings(optim(par = param.ini, fn = minuslogL, 
        hessian = hessian, control = control, ...))
    res <- reswarn$value
    if (warnings) {
        if (length(reswarn$warnings) > 0) {
            v <- unlist(lapply(reswarn$warnings, as.character))
            tv <- data.frame(message = v, freq = rep(1, length(v)))
            cat("Warnings during likelihood maximisation:\n")
            print(aggregate(tv[, "freq", drop = FALSE], list(warning = tv$message), 
                sum))
        }
    }
    param <- res$par
    namb <- colnames(modmatrix.b)
    namphi <- paste("phi", colnames(modmatrix.phi), sep = ".")
    nam <- c(namb, namphi)
    names(param) <- nam
    if (!is.null(unlist(fixpar))) 
        param[fixpar[[1]]] <- fixpar[[2]]
    H <- H.singular <- Hr.singular <- NA
    varparam <- matrix(NA)
    is.singular <- function(X) qr(X)$rank < nrow(as.matrix(X))
    if (hessian) {
        H <- res$hessian
        if (is.null(unlist(fixpar))) {
            H.singular <- is.singular(H)
            if (!H.singular) 
                varparam <- qr.solve(H)
            else warning("The hessian matrix was singular.\n")
        }
        else {
            idparam <- 1:(nb.b + nb.phi)
            idestim <- idparam[-fixpar[[1]]]
            Hr <- as.matrix(H[-fixpar[[1]], -fixpar[[1]]])
            H.singular <- is.singular(Hr)
            if (!H.singular) {
                Vr <- solve(Hr)
                dimnames(Vr) <- list(idestim, idestim)
                varparam <- matrix(rep(NA, NROW(H) * NCOL(H)), 
                  ncol = NCOL(H))
                varparam[idestim, idestim] <- Vr
            }
        }
    }
    else varparam <- matrix(NA)
    if (any(!is.na(varparam))) 
        dimnames(varparam) <- list(nam, nam)
    nbpar <- if (is.null(unlist(fixpar))) 
        sum(!is.na(param))
    else sum(!is.na(param[-fixpar[[1]]]))
    logL.max <- sum(dbinom(x = y, size = n, prob = y/n, log = TRUE))
    logL <- -res$value
    dev <- -2 * (logL - logL.max)
    df.residual <- sum(n > 0) - nbpar
    iterations <- res$counts[1]
    code <- res$convergence
    msg <- if (!is.null(res$message)) 
        res$message
    else character(0)
    if (code != 0) 
        warning("\nPossible convergence problem. Optimization process code: ", 
            code, " (see ?optim).\n")
    new(Class = "glimML", CALL = CALL, link = link, method = "BB", 
        data = data, formula = formula, random = random, param = param, 
        varparam = varparam, fixed.param = param[seq(along = namb)], 
        random.param = param[-seq(along = namb)], logL = logL, 
        logL.max = logL.max, dev = dev, df.residual = df.residual, 
        nbpar = nbpar, iterations = iterations, code = code, 
        msg = msg, singular.hessian = as.numeric(H.singular), 
        param.ini = param.ini, na.action = na.action)
}

invlink <- function (x, type = c("cloglog", "log", "logit", "probit")) 
{
    switch(type, logit = plogis(x), probit = pnorm(x), log = exp(x), cloglog = 1 - 
        exp(-exp(x)))
}

link <- function (x, type = c("cloglog", "log", "logit", "probit")) 
{
    switch(type, logit = qlogis(x), probit = qnorm(x), 
    log = log(x), cloglog = log(-log(1 - x)))
}


pr <- function (object, ...) 
{
    .local <- function (object, newdata = NULL, type = c("response", 
        "link"), se.fit = FALSE, ...) 
    {
        type <- match.arg(type)
        mf <- object@CALL
        b <- coef(object)
        f <- object@formula[-2]
        data <- object@data
        offset <- NULL
        if (is.null(newdata)) {
            mb <- match(c("formula", "data", "na.action"), names(mf), 
                0)
            mfb <- mf[c(1, mb)]
            mfb$drop.unused.levels <- TRUE
            mfb[[1]] <- as.name("model.frame")
            names(mfb)[2] <- "formula"
            mfb <- eval(mfb, parent.frame())
            mt <- attr(mfb, "terms")
            Y <- model.response(mfb, "numeric")
            X <- if (!is.empty.model(mt)) 
                model.matrix(mt, mfb, contrasts)
            else matrix(, NROW(Y), 0)
            offset <- model.offset(mfb)
        }
        else {
            mfb <- model.frame(f, newdata)
            offset <- model.offset(mfb)
            X <- model.matrix(object = f, data = newdata)
        }
        eta <- as.vector(X %*% b)
        eta <- if (is.null(offset)) 
            eta
        else eta + offset
        varparam <- object@varparam
        varb <- as.matrix(varparam[seq(length(b)), seq(length(b))])
        vareta <- X %*% varb %*% t(X)
        if (type == "response") {
            p <- invlink(eta, type = object@link)
            J <- switch(object@link, logit = diag(p * (1 - p), 
                nrow = length(p)), probit = diag(dnorm( qnorm(p) ), 
                nrow = length(p)), cloglog = diag(-(1 - p) * 
                log(1 - p), nrow = length(p)), log = diag(p, 
                nrow = length(p)))
            varp <- J %*% vareta %*% J
            se.p <- sqrt(diag(varp))
        }
        se.eta <- sqrt(diag(vareta))
        if (!se.fit) 
            res <- switch(type, response = p, link = eta)
        else res <- switch(type, response = list(fit = p, se.fit = se.p), 
            link = list(fit = eta, se.fit = se.eta))
        res
    }
    .local(object, ...)
}

setMethod(predict, "glimML", pr)
setMethod(fitted, "glimML", function (object, ...) {
    mf <- object@CALL
    mb <- match(c("formula", "data", "na.action"), names(mf), 
        0)
    mfb <- mf[c(1, mb)]
    mfb$drop.unused.levels <- TRUE
    mfb[[1]] <- as.name("model.frame")
    names(mfb)[2] <- "formula"
    mfb <- eval(mfb, parent.frame())
    mt <- attr(mfb, "terms")
    Y <- model.response(mfb, "numeric")
    X <- if (!is.empty.model(mt)) 
        model.matrix(mt, mfb, contrasts)
    else matrix(, NROW(Y), 0)
    offset <- model.offset(mfb)
    b <- coef(object)
    eta <- as.vector(X %*% b)
    eta <- if (is.null(offset)) 
        eta
    else eta + offset
    invlink(eta, type = object@link)
}
)
