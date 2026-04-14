#' ---
#' Título: Modelo de Saltos de Merton (Jump-Diffusion)
#' Autor: Luiz Tiago Wilcke
#' Fonte: Extraído e adaptado do Capítulo 19 - Finanças Quantitativas
#' Descrição: Implementação avançada da dinâmica de preços com saltos de Poisson.
#' Aplicação: Mercado Financeiro (Black Monday, Pandemia) e Fenômenos Sociais (Shocks).
#' ---

# Limpar ambiente
rm(list = ls())

# Funcao para simular o processo de Merton Jump-Diffusion
# dS/S = (mu - lambda * k)dt + sigma * dW + dJ
simular_saltos_merton <- function(S0 = 100, mu = 0.05, sigma = 0.2, 
                                 lambda = 0.75, mu_j = -0.05, sigma_j = 0.1, 
                                 T_total = 1, n_passos = 252) {
  
  # Parametros iniciais
  dt <- T_total / n_passos
  t_vetor <- seq(0, T_total, length.out = n_passos + 1)
  
  # Compensador de drift (k = E[exp(Y) - 1])
  k <- exp(mu_j + 0.5 * sigma_j^2) - 1
  drift_ajustado <- (mu - lambda * k) * dt
  
  # Caminho do preco
  precos <- numeric(n_passos + 1)
  precos[1] <- S0
  
  # Gerar componentes estocasticos
  choques_difusao <- rnorm(n_passos, mean = 0, sd = 1)
  
  # Processo de Poisson (Numero de saltos em cada dt)
  # Usamos aproximação para dt pequeno: Prob(Salto) ~ lambda * dt
  num_saltos <- rpois(n_passos, lambda * dt)
  
  for (i in 1:n_passos) {
    # Componente de Difusao (Movimento Browniano Geometrico)
    difusao <- sigma * sqrt(dt) * choques_difusao[i]
    
    # Componente de Salto (Se houver saltos, somamos os log-incrementos)
    salto_total <- 0
    if (num_saltos[i] > 0) {
      salto_total <- sum(rnorm(num_saltos[i], mean = mu_j, sd = sigma_j))
    }
    
    # Atualizacao do preco via Euler-Maruyama no espaco logaritmico
    log_retorno <- drift_ajustado - 0.5 * sigma^2 * dt + difusao + salto_total
    precos[i+1] <- precos[i] * exp(log_retorno)
  }
  
  return(data.frame(tempo = t_vetor, preco = precos))
}

# Aplicacao ao Mercado Financeiro e Sociedade
set.seed(42)
resultado_simulacao <- simular_saltos_merton(S0 = 1000, mu = 0.1, sigma = 0.25, 
                                              lambda = 2, mu_j = -0.15, sigma_j = 0.05)

# Visualizacao Estetica
plot(resultado_simulacao$tempo, resultado_simulacao$preco, type = "l", col = "darkblue",
     main = "Simulação de Salto de Merton: Crises e Choques Estruturais",
     xlab = "Tempo (Anos)", ylab = "Nível do Ativo / Indicador Social",
     sub = "Autor: Luiz Tiago Wilcke | Fonte: Cap. 19 do Livro")
grid()

# Adicionar linha de tendencia sem saltos para comparacao
abline(h = 1000 * exp(0.1), col = "red", lty = 2)
legend("topleft", legend = c("Preço com Saltos", "Tendência Esperada"), 
       col = c("darkblue", "red"), lty = 1:2)

cat("Simulação concluída com sucesso. O modelo captura a volatilidade extrema e saltos de mercado.\n")
