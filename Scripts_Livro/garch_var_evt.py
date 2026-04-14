"""
Título: Risco Sistêmico com GARCH-EVT (Extreme Value Theory)
Autor: Luiz Tiago Wilcke
Fonte: Extraído e adaptado do Capítulo 19 - Finanças Quantitativas
Descrição: Estimação de VaR e Expected Shortfall (ES) via GPD (Generalized Pareto).
Aplicação: Gestão de riscos extremos em mercados e redes sociais.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model # Instalar via `pip install arch` se necessário

def calcular_risco_extremo_garch_evt(retornos):
    """
    Combina volatilidade condicional (GARCH) com modelagem de cauda (EVT).
    """
    # 1. Estimar GARCH(1,1) para extrair os resíduos padronizados (Z_t)
    modelo = arch_model(retornos * 100, vol='Garch', p=1, q=1, dist='t')
    res_fit = modelo.fit(disp='off')
    
    # Resíduos padronizados
    z_t = res_fit.std_resid.dropna()
    vol_condicional = res_fit.conditional_volatility[-1] / 100
    mu_estimado = res_fit.params['mu'] / 100
    
    # 2. Aplicar EVT (GPD) na cauda esquerda dos resíduos
    perdas = -z_t
    limiar_u = np.quantile(perdas, 0.90) # 90% percentil
    excedencias = perdas[perdas > limiar_u] - limiar_u
    
    # Estimação dos parâmetros da Pareto Generalizada (xi e beta)
    xi, _, beta = stats.genpareto.fit(excedencias, floc=0)
    
    # 3. Calcular VaR e ES para o nível de confiança (ex: 99%)
    confianca = 0.99
    n_total = len(perdas)
    n_u = len(excedencias)
    
    # VaR para os resíduos (z_alpha)
    var_z = limiar_u + (beta / xi) * ((((n_total * (1 - confianca)) / n_u) ** -xi) - 1)
    
    # ES para os resíduos (es_z)
    es_z = (var_z + beta - xi * limiar_u) / (1 - xi)
    
    # Converter para retornos (despadronizar)
    var_final = -(mu_estimado + vol_condicional * var_z)
    es_final = -(mu_estimado + vol_condicional * es_z)
    
    return var_final, es_final

if __name__ == "__main__":
    # Simulação de retornos com caudas grossas (t-student)
    np.random.seed(123)
    retornos_simulados = np.random.standard_t(df=4, size=1000) * 0.02
    
    var_99, es_99 = calcular_risco_extremo_garch_evt(retornos_simulados)
    
    print("-" * 50)
    print("MÉTRICAS DE RISCO AVANÇADAS (GARCH-EVT)")
    print(f"Autor: Luiz Tiago Wilcke")
    print(f"Fonte: Cap. 19 do Livro")
    print("-" * 50)
    print(f"VaR (99%): {var_99*100:.4f}%")
    print(f"Expected Shortfall (99%): {es_99*100:.4f}%")
    print("-" * 50)
    print("Nota: O modelo captura o 'Fat Tail' que o VaR Gaussiano subestima.")
