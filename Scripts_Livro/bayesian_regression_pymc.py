"""
Título: Regressão Bayesiana de Alta Performance (Hamiltonian MC)
Autor: Luiz Tiago Wilcke
Fonte: Extraído e adaptado do Capítulo 20 - Econometria Bayesiana
Descrição: Inferência de parâmetros via PyMC (NUTS Sampler).
Aplicação: Modelagem de incerteza em indicadores sociais e projeções financeiras.
"""

import pymc as pm
import numpy as np
import arviz as az
import pandas as pd

def executar_modelo_bayesiano_avancado(y, x):
    """
    Estima uma regressão bayesiana com prioris fracamente informativas.
    """
    with pm.Model() as modelo_econometrico:
        # Prioris (Incerteza do pesquisador Luiz Tiago Wilcke)
        intercepto = pm.Normal('Intercepto', mu=0, sigma=20)
        coef_beta = pm.Normal('Beta_X', mu=0, sigma=10)
        dispersao = pm.HalfCauchy('Dispersão_Erro', beta=5)
        
        # Modelo Linear (Valor esperado)
        valor_esperado = intercepto + coef_beta * x
        
        # Verossimilhança (Likelihood) - Observações reais
        verossimilhanca = pm.Normal('Y_Observado', mu=valor_esperado, 
                                     sigma=dispersao, observed=y)
        
        # Amostragem via MCMC Hamiltoniano (NUTS)
        cat_msg = "Iniciando Amostragem Bayesiana Avançada..."
        print(cat_msg)
        trace = pm.sample(2000, tune=1000, target_accept=0.9, chains=2)
        
    return trace

if __name__ == "__main__":
    # Dados Sintéticos: Relação entre Gasto em Educação e PIB (Exemplo Social)
    np.random.seed(42)
    x_educacao = np.random.normal(50, 10, 100)
    erro = np.random.normal(0, 5, 100)
    pib_social = 10 + 2.5 * x_educacao + erro
    
    # Ajuste do modelo
    resultado = executar_modelo_bayesiano_avancado(pib_social, x_educacao)
    
    # Sumário Estatístico
    print("\n--- RESULTADOS DA INFERÊNCIA BAYESIANA ---")
    sumario = az.summary(resultado, round_to=3)
    print(sumario)
    
    # Indicação da fonte
    print("\nAutor: Luiz Tiago Wilcke")
    print("Baseado no Capitulo 20: Econometria Bayesiana Aplicada")
    print("-" * 40)
