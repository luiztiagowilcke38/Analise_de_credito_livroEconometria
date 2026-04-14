"""
Título: Simulação do Modelo de Saltos de Merton (Jump-Diffusion) em Python
Autor: Luiz Tiago Wilcke
Fonte: Extraído e adaptado do Capítulo 19 - Finanças Quantitativas
Descrição: Implementação via Monte Carlo para capturar 'Crashes' e Choques Sociais.
Aplicação: Finanças Quantitativas e Análise de Risco Sistêmico.
"""

import numpy as np
import matplotlib.pyplot as plt

def simular_saltos_merton_python(s0=1000, mu=0.1, sigma=0.2, lamb=1.5, 
                                 mu_j=-0.1, sigma_j=0.05, t_final=1.0, n_passos=252):
    """
    Simula trajetórias de preços usando o modelo de Merton Jump-Diffusion.
    """
    dt = t_final / n_passos
    tempos = np.linspace(0, t_final, n_passos + 1)
    
    # Compensador do drift para manter a martingale se necessário (k = E[exp(Y)-1])
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_ajustado = (mu - lamb * k) * dt
    
    precos = np.zeros(n_passos + 1)
    precos[0] = s0
    
    # Gerar choques de difusão (Movimento Browniano)
    z = np.random.standard_normal(n_passos)
    
    # Gerar número de saltos em cada intervalo dt (Processo de Poisson)
    n_saltos = np.random.poisson(lamb * dt, n_passos)
    
    for i in range(n_passos):
        # Difusão padrão
        difusao = sigma * np.sqrt(dt) * z[i]
        
        # Saltos (soma de log-normais se n_saltos > 0)
        componente_salto = 0
        if n_saltos[i] > 0:
            componente_salto = np.sum(np.random.normal(mu_j, sigma_j, n_saltos[i]))
            
        # Evolução do preço (Euler-Maruyama logarítmico)
        incremento_log = drift_ajustado - 0.5 * (sigma**2) * dt + difusao + componente_salto
        precos[i+1] = precos[i] * np.exp(incremento_log)
        
    return tempos, precos

if __name__ == "__main__":
    # Parametrização para um cenário de crise de mercado (Shocks negativos frequentes)
    np.random.seed(9)
    tempo, preco_final = simular_saltos_merton_python(
        s0=100, mu=0.08, sigma=0.15, lamb=3.0, mu_j=-0.05, sigma_j=0.02
    )
    
    # Visualização Profissional
    plt.figure(figsize=(10, 6))
    plt.plot(tempo, preco_final, label='Modelo de Merton (Saltos)', color='navy', lw=1.5)
    plt.title('Simulação de Crises e Saltos Estruturais - Autor: Luiz Tiago Wilcke', fontsize=12)
    plt.xlabel('Tempo (Anos)', fontsize=10)
    plt.ylabel('Preço / Índice Socieconômico', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Indicação da fonte conforme solicitado
    plt.figtext(0.5, 0.01, "Fonte: Livro 'Métodos Econométricos', Capítulo 19: Finanças Quantitativas", 
                ha="center", fontsize=8, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.show()
    print("Script Python de Saltos de Merton finalizado.")
