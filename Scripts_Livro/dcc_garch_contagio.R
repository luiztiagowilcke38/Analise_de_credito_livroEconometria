#' ---
#' Título: Modelagem de Contágio via DCC-GARCH
#' Autor: Luiz Tiago Wilcke
#' Fonte: Extraído e adaptado do Capítulo 19 - Finanças Quantitativas
#' Descrição: Modelo de Correlação Condicional Dinâmica (DCC).
#' Aplicação: Análise de contágio entre mercados (IBOVESPA vs SP500) e crises sociais.
#' ---

# Carregar bibliotecas necessarias
if (!require(rmgarch)) install.packages("rmgarch")
if (!require(quantmod)) install.packages("quantmod")

library(rmgarch)
library(quantmod)

# 1. Coleta de Dados Reais: IBOVESPA (^BVSP) e S&P 500 (^GSPC)
# Representando a interconexao dos mercados globais e o contagio financeiro
simbolos <- c("^BVSP", "^GSPC")
getSymbols(simbolos, from = "2019-01-01", to = Sys.Date())

# Calcular retornos logaritmicos
retornos_ibov <- dailyReturn(Cl(BVSP), type = "log")
retornos_sp500 <- dailyReturn(Cl(GSPC), type = "log")

# Unir dados e remover NAs
dados_conjuntos <- merge(retornos_ibov, retornos_sp500, all = FALSE)
colnames(dados_conjuntos) <- c("IBOV", "SP500")
dados_limpos <- na.omit(dados_conjuntos)

# 2. Especificação do Modelo GARCH Univariado (sGARCH(1,1) com t-Student)
# Capturamos a volatilidade individual de cada mercado
espec_univariada <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)

# Replica para os dois ativos
espec_multipla <- multispec(replicate(2, espec_univariada))

# 3. Especificação do DCC(1,1)
# O modelo DCC permite que a correlacao entre os mercados mude ao longo do tempo
espec_dcc <- dccspec(
  uspec = espec_multipla, 
  dccOrder = c(1, 1), 
  distribution = "mvt" # t-Student multivariada para caudas pesadas conjuntas
)

# 4. Estimacao do Modelo
ajuste_dcc <- dccfit(espec_dcc, data = as.matrix(dados_limpos))

# 5. Visualizacao do Contagio (Correlacao Dinamica)
correlacoes_dinamicas <- rcor(ajuste_dcc) # Retorna array [n, n, T]
rho_ibov_sp500 <- correlacoes_dinamicas[1, 2, ]

# Plotar o grafico de contagio
plot(as.zoo(dados_limpos$IBOV), type = "n", 
     main = "Contágio Dinâmico: Correlação IBOVESPA vs S&P 500",
     ylab = "Correlação (Rho)", xlab = "Tempo")
lines(as.Date(index(dados_limpos)), rho_ibov_sp500, col = "darkgreen", lwd = 2)
abline(h = mean(rho_ibov_sp500), col = "red", lty = 2)
grid()

legend("bottomright", legend = c("Correlação Dinâmica (DCC)", "Média Histórica"), 
       col = c("darkgreen", "red"), lty = c(1, 2), bty = "n")

cat("Análise de contágio finalizada. Autor: Luiz Tiago Wilcke.\n")
cat("O modelo mostra como a interdependência aumenta drasticamente em períodos de crise.\n")
