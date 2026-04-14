#' ---
#' Título: BVAR (Bayesian VAR) com Minnesota Prior
#' Autor: Luiz Tiago Wilcke
#' Fonte: Extraído e adaptado do Capítulo 20 - Econometria Bayesiana
#' Descrição: Estimação de Vetores Autorregressivos Bayesianos para macroeconomia.
#' Aplicação: Previsão de PIB, Inflação e Taxa de Juros sob incertezas sociais.
#' ---

# Carregar pacotes
if (!require(BVAR)) install.packages("BVAR")
library(BVAR)

# 1. Simulação de Dados Macroeconômicos (Cenário Brasil)
# y1: PIB (Crescimento), y2: IPCA (Inflação), y3: SELIC (Juros)
set.seed(123)
n_obs <- 120 # 10 anos de dados mensais
t <- 1:n_obs

pib <- 2 + 0.05 * t + rnorm(n_obs, 0, 0.5)
inflacao <- 4 + 0.02 * t + rnorm(n_obs, 0, 1)
selic <- 10 - 0.01 * t + rnorm(n_obs, 0, 2)

dados_macro <- cbind(pib, inflacao, selic)
colnames(dados_macro) <- c("PIB", "IPCA", "SELIC")

# 2. Configuração da Minnesota Prior
# O parâmetro 'lambda' controla o encolhimento (shrinkage)
# Valores menores de lambda significam prioris mais fortes em direção ao passeio aleatório
espec_prior <- bv_priors(
  hyper = "auto", 
  mn = bv_mn(lambda = bv_lambda(mode = 0.2, sd = 0.4), alpha = 2)
)

# 3. Estimação do Modelo BVAR
# Usamos 2 defasagens (lags) para capturar a dinâmica temporal
ajuste_bvar <- bvar(dados_macro, lags = 2, n_draw = 10000, n_burn = 5000, 
                    priors = espec_prior)

# 4. Previsão (Forecast) para os próximos 12 meses
previsoes <- predict(ajuste_bvar, horizon = 12)

# 5. Visualização das Previsões com Intervalos de Credibilidade
plot(previsoes, main = "Previsão Macroeconômica Bayesiana (BVAR)",
     xlab = "Meses à Frente", ylab = "Valores Estimados")

cat("Modelo BVAR estimado com sucesso. Autor: Luiz Tiago Wilcke.\n")
cat("A abordagem Bayesiana permite incorporar incerteza estrutural na política monetária.\n")
