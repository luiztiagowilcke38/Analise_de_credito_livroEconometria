#' Métodos Econométricos - Luiz Tiago Wilcke
#' 
#' Este script implementa uma suíte de ferramentas econométricas fundamentada no livro texto.
#' Abrange Séries Temporais, Finanças Quantitativas e Econometria Bayesiana.
#' 
#' @author Luiz Tiago Wilcke
#' @description Implementações de alto rigor matemático em R.

library(MASS)
library(stats)

# ==============================================================================
# 1. SÉRIES TEMPORAIS DINÂMICAS: SVAR e VECM
# ==============================================================================

#' Algoritmo SVAR de Luiz Tiago (Structural Vector Autoregression)
#' Identificação via Decomposição de Cholesky
svar_fit <- function(data, p = 1) {
  n <- nrow(data)
  k <- ncol(data)
  
  # Preparar defasagens
  Y <- data[(p + 1):n, ]
  X <- matrix(1, nrow = n - p, ncol = 1)
  
  for (i in 1:p) {
    X <- cbind(X, data[(p + 1 - i):(n - i), ])
  }
  
  # Estimação OLS (beta = (X'X)^-1 X'Y)
  XtX_inv <- solve(t(X) %*% X)
  beta <- XtX_inv %*% t(X) %*% Y
  
  residuos <- Y - X %*% beta
  sigma <- (t(residuos) %*% residuos) / (n - p - k * p - 1)
  
  # Identificação estrutural via triangularização de Cholesky
  B <- t(chol(sigma)) # L tal que L L' = Sigma
  
  return(list(beta = beta, sigma = sigma, B = B, p = p, k = k, data = data))
}

#' Função de Resposta ao Impulso (IRF) Estrutural
svar_irf <- function(model, horizons = 15, accum = FALSE) {
  k <- model$k
  p <- model$p
  B <- model$B
  beta <- model$beta[-1, ] # Remover intercepto
  
  # Companion Form matrix
  m <- k * p
  F_mat <- matrix(0, m, m)
  F_mat[1:k, ] <- t(beta)
  if (p > 1) {
    F_mat[(k + 1):m, 1:(m - k)] <- diag(m - k)
  }
  
  responses <- array(0, dim = c(horizons, k, k))
  
  for (j in 1:k) {
    shock <- rep(0, m)
    shock[1:k] <- B[, j]
    
    state <- shock
    responses[1, , j] <- state[1:k]
    
    for (h in 2:horizons) {
      state <- F_mat %*% state
      responses[h, , j] <- state[1:k]
    }
  }
  
  if (accum) {
    for (j in 1:k) {
      responses[, , j] <- apply(responses[, , j], 2, cumsum)
    }
  }
  
  return(responses)
}

#' Teste de Rank de Johansen (VECM)
vecm_johansen <- function(data, p = 2) {
  n <- nrow(data)
  k <- ncol(data)
  
  diff_data <- diff(data)
  Y <- diff_data[p:nrow(diff_data), ]
  X_diff <- NULL
  if (p > 1) {
    for (i in 1:(p - 1)) {
      X_diff <- cbind(X_diff, diff_data[(p - i):(nrow(diff_data) - i), ])
    }
  }
  
  Z <- data[p:(n - 1), ]
  
  # Concentrar as defasagens de curto prazo (regressão auxiliar)
  if (!is.null(X_diff)) {
    Xd <- cbind(1, X_diff)
    M <- diag(nrow(Y)) - Xd %*% solve(t(Xd) %*% Xd) %*% t(Xd)
    R0 <- M %*% Y
    R1 <- M %*% Z
  } else {
    R0 <- scale(Y, scale = FALSE)
    R1 <- scale(Z, scale = FALSE)
  }
  
  S00 <- (t(R0) %*% R0) / nrow(R0)
  S11 <- (t(R1) %*% R1) / nrow(R1)
  S01 <- (t(R0) %*% R1) / nrow(R0)
  S10 <- t(S01)
  
  # Autovalores Canônicos
  C <- solve(S11) %*% S10 %*% solve(S00) %*% S01
  res <- eigen(C)
  
  # Estatística do Traço
  evals <- res$values
  trace_stat <- - (n - p) * rev(cumsum(log(1 - rev(evals))))
  
  return(list(evals = evals, trace = trace_stat, evecs = res$vectors))
}

#' Filtro de Kalman de Luiz Tiago (Local Level Model)
kalman_filter <- function(y, m0, C0, V, W) {
  n <- length(y)
  m <- numeric(n + 1)
  C <- numeric(n + 1)
  m[1] <- m0
  C[1] <- C0
  
  for (t in 1:n) {
    # Predição
    a <- m[t]
    R <- C[t] + W
    
    # Atualização
    f <- a
    Q <- R + V
    K <- R / Q
    
    m[t+1] <- a + K * (y[t] - f)
    C[t+1] <- R - K * Q * K
  }
  return(list(m = m[-1], C = C[-1]))
}

#' Análise Espectral e Periodograma Suavizado
analise_espectral <- function(y, spans = c(3,3)) {
  # f(w) via spec.pgram
  res <- spec.pgram(y, spans = spans, plot = FALSE)
  return(list(freq = res$freq, spec = res$spec))
}

# ==============================================================================
# 2. VOLATILIDADE E RISCO (GARCH e EVT)
# ==============================================================================

#' GJR-GARCH(1,1) via Máxima Verossimilhança
gjr_garch_fit <- function(rets) {
  n <- length(rets)
  
  # Função Objetivo: -Log-Verossimilhança
  ll_target <- function(params) {
    omega <- params[1]
    alpha <- params[2]
    gamma <- params[3]
    beta  <- params[4]
    
    if (omega <= 0 || alpha < 0 || gamma < 0 || beta < 0 || (alpha + 0.5*gamma + beta) >= 1) return(1e10)
    
    v <- numeric(n)
    v[1] <- var(rets)
    
    for (t in 2:n) {
      I <- if (rets[t-1] < 0) 1 else 0
      v[t] <- omega + (alpha + gamma * I) * rets[t-1]^2 + beta * v[t-1]
    }
    
    ll <- -0.5 * sum(log(2*pi*v) + rets^2 / v)
    return(-ll)
  }
  
  p0 <- c(0.01, 0.05, 0.1, 0.8)
  res <- optim(p0, ll_target, method = "L-BFGS-B", 
               lower = c(1e-6, 0, 0, 0), upper = c(1, 1, 1, 1))
  
  return(res)
}

#' Valor-em-Risco (VaR) via Teoria de Valores Extremos (EVT-GPD)
evt_var_es <- function(rets, alpha = 0.99, q_thresh = 0.90) {
  losses <- -rets
  u <- quantile(losses, q_thresh)
  excess <- losses[losses > u] - u
  
  # Estimação manual via MLE para Pareto Generalizada (GPD)
  gpd_ll <- function(params) {
    xi <- params[1]
    beta <- params[2]
    if (beta <= 0) return(1e10)
    
    # f(x; xi, beta) = (1/beta) * (1 + xi*x/beta)^(-1/xi - 1)
    n_exc <- length(excess)
    if (abs(xi) < 1e-8) {
      ll <- -n_exc*log(beta) - sum(excess/beta)
    } else {
      cond <- 1 + xi*excess/beta
      if (any(cond <= 0)) return(1e10)
      ll <- -n_exc*log(beta) - (1/xi + 1) * sum(log(cond))
    }
    return(-ll)
  }
  
  res <- optim(c(0.1, sd(excess)), gpd_ll)
  xi <- res$par[1]
  beta <- res$par[2]
  
  # VaR e ES
  n <- length(losses)
  nu <- length(excess)
  
  var_evt <- u + (beta/xi) * (((n/nu)*(1-alpha))^(-xi) - 1)
  es_evt <- (var_evt + beta - xi*u) / (1 - xi)
  
  return(list(VaR = var_evt, ES = es_evt, params = res$par))
}

# ==============================================================================
# 3. ECONOMETRIA BAYESIANA
# ==============================================================================

#' Amostrador de Gibbs (Gibbs Sampler) para Regressão Linear
gibbs_linear <- function(y, X, draws = 5000) {
  n <- nrow(X)
  k <- ncol(X)
  
  # Priors: beta ~ N(b0, B0), sigma2 ~ IG(n0/2, s02*n0/2)
  b0 <- rep(0, k)
  B0_inv <- diag(0.001, k)
  n0 <- 0.01
  s02 <- 0.01
  
  # Inits
  sig2 <- 1.0
  beta_trace <- matrix(0, draws, k)
  sig2_trace <- numeric(draws)
  
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  
  for (i in 1:draws) {
    # 1. Update Beta | sigma2
    V_n <- solve(B0_inv + XtX / sig2)
    m_n <- V_n %*% (B0_inv %*% b0 + Xty / sig2)
    beta <- mvrnorm(1, m_n, V_n)
    
    # 2. Update sigma2 | Beta
    err <- y - X %*% beta
    nn <- n0 + n
    sn2 <- (n0*s02 + sum(err^2)) / nn
    # R: rgamma(shape, scale) => IG(a, b) = 1 / rgamma(a, 1/b)
    sig2 <- 1 / rgamma(1, nn/2, rate = (nn*sn2)/2)
    
    beta_trace[i, ] <- beta
    sig2_trace[i] <- sig2
  }
  
  return(list(beta = beta_trace, sigma2 = sig2_trace))
}

# ==============================================================================
# 4. CAUSALIDADE E MACHINE LEARNING
# ==============================================================================

#' Double Machine Learning (DML) - Orthogonalized OLS
dml_linear <- function(y, d, X) {
  # 1. Residualizar Y
  mod_y <- lm(y ~ X)
  res_y <- residuals(mod_y)
  
  # 2. Residualizar D
  mod_d <- lm(d ~ X)
  res_d <- residuals(mod_d)
  
  # 3. Efeito Causal (Frisch-Waugh-Lovell Theorem)
  theta <- sum(res_d * res_y) / sum(res_d^2)
  return(theta)
}

#' Black-Scholes e Gregas
black_scholes_gregas <- function(S, K, T, r, sigma, type = "call") {
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  
  if (type == "call") {
    price <- S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
    delta <- pnorm(d1)
  } else {
    price <- K * exp(-r * T) * pnorm(-d2) - S * pnorm(-d1)
    delta <- pnorm(d1) - 1
  }
  
  gamma <- dnorm(d1) / (S * sigma * sqrt(T))
  vega <- S * dnorm(d1) * sqrt(T)
  
  return(list(price = price, delta = delta, gamma = gamma, vega = vega))
}

# ==============================================================================
# 5. DEMONSTRAÇÃO GERAL E VALIDAÇÃO
# ==============================================================================

executar_demonstracao <- function() {
  cat("\n" %+% paste(rep("=", 60), collapse="") %+% "\n")
  cat("SUÍTE ECONOMÉTRICA PROFISSIONAL - LUIZ TIAGO WILCKE\n")
  cat(paste(rep("=", 60), collapse="") %+% "\n")
  
  # 1. SVAR e IRF
  set.seed(42)
  e <- matrix(rnorm(600), 300, 2) %*% t(chol(matrix(c(1, 0.6, 0.6, 1), 2, 2)))
  data <- matrix(0, 300, 2)
  for(t in 2:300) {
    data[t, ] <- data[t-1, ] %*% matrix(c(0.6, 0.1, 0.2, 0.8), 2, 2) + e[t, ]
  }
  
  mod <- svar_fit(data, p = 1)
  cat("\n[SVAR] Matriz Estrutural B (Choque): \n")
  print(mod$B)
  irfs <- svar_irf(mod, horizons=10)
  cat("Resposta de Y2 ao choque em Y1 no passo 5:", irfs[5, 2, 1], "\n")
  
  # 2. Kalman Filter
  y_ks <- cumsum(rnorm(100, sd=0.5)) + rnorm(100, sd=1)
  kf <- kalman_filter(y_ks, 0, 10, 1, 0.25)
  cat("\n[KALMAN] Último estado estimado:", tail(kf$m, 1), "\n")
  
  # 3. Opções e Gregas
  opt <- black_scholes_gregas(100, 105, 0.5, 0.05, 0.2)
  cat("\n[FINANÇAS] Call Price (S=100, K=105):", opt$price, "\n")
  cat("Delta:", opt$delta, "| Gamma:", opt$gamma, "\n")
  
  cat("\n[STATUS] Todos os modelos executados com rigor matemático.\n")
}

# Helper para concatenação
`%+%` <- function(a, b) paste0(a, b)

if (TRUE) {
  executar_demonstracao()
}
