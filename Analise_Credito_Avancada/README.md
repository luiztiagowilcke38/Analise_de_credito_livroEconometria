# Motor de Análise de Crédito Avançado

Este ambiente interativo contém a implementação estrita de uma arquitetura preditiva e causal de crédito formulada com base no conjunto analítico estabelecido no livro da tese. Tradicionalmente, o mercado de crédito foca apenas em *Regressões Logísticas* ou ML "Caixa Preta". Aqui estendemos esta visão aplicando algoritmos econométricos computacionais de ponta.

## 1. O Problema Fundamental do Crédito

Instituições financeiras lidam com duas frentes fundamentais:
1. **Previsão Precisa (Prediction)**: Quão provável é que este cliente pare de pagar? (Probability of Default - PD).
2. **Inferência Causal de Intervenção (Causality)**: Se eu oferecer a ele uma extensão de limite (Tratamento) ou uma pausa no pagamento, isso vai causar um aumento ou redução no default final? 

Responder a segunda questão olhando dados observacionais é um erro gravíssimo de **Confounding Bias** (Fator Oculto de Confusão). Clientes que recebem melhores ofertas geralmente já são os melhores clientes. Um MQO Ingênuo subestimaria ou inverteria o efeito.

## 2. Modelagem Estatística Avançada e Equações

O código computacional engloba 3 metodologias analíticas robustas abordadas na obra:

### A. Previsão Direta com Ensemble (XGBoost)
O default $Y_i \in \{0,1\}$ é modelado como um processo não-linear de interação de covariáveis de score e renda empírica ($X_i$). Diferente do Logit padrão, as árvores impulsionadas solucionam a função de otimização de forma recursiva:

$$
\hat{Y}_i = \sum_{k=1}^K f_k(X_i), \quad f_k \in \mathcal{F}
$$

A função objetivo a ser otimizada (minimização do Brier Score ou Logloss com regulação) é dada por:

$$
\mathcal{O} = \sum_{i=1}^n L\left( Y_i, \hat{Y}_i \right) + \sum_{k=1}^K \Omega(f_k)
$$

Onde $\Omega$ estabiliza complexidade e sobreajuste (Regularização L1/L2).

### B. Interpretabilidade Causal Regulatória (Cálculo SHAP)
No setor bancário, modelos de recusa precisam de "Explainability". Ao invés do obsoleto índice Gini, recorremos à teoria dos jogos de Shapley:

$$
\phi_j(v) = \sum_{S \subseteq \{x_1, \dots, x_p\} \setminus \{x_j\}} \frac{|S|!(p-|S|-1)!}{p!} \left[ v(S \cup \{x_j\}) - v(S) \right]
$$

Isso decompõe perfeitamente o impacto isolado que a ausência ou presença do score e renda provocam na probabilidade algorítmica final.

### C. Abordagem Causal pelo Double Machine Learning (DML)
Para descobrir o *Efeito Mínimo Assintótico* da intervenção (Treatment $D_i$) controlando os fatores de confusão (Confounders $X_i$), o modelo utiliza o **Neyman Orthogonalization**:

**Passo 1 (Modelo de Resultado $M_Y$):** Residualizamos o resultado $Y_i$:

$$
\tilde{Y}_i = Y_i - \hat{\mathbb{E}}[Y_i | X_i]
$$

**Passo 2 (Modelo de Propensão $M_D$):** Residualizamos o tratamento $D_i$:

$$
\tilde{D}_i = D_i - \hat{\mathbb{E}}[D_i | X_i]
$$

**Passo 3 (Estimativa Parcial):** O efeito purificado $\hat{\theta}$ emerge da regressão OLS Frisch-Waugh-Lovell nos resíduos oriundos de florestas aleatórias com Cross-Fitting:

$$
\hat{\theta} = \left( \sum_{i=1}^n \tilde{D}_i^2 \right)^{-1} \sum_{i=1}^n \tilde{D}_i \tilde{Y}_i
$$

## 3. Instruções de Verificação

Execute o arquivo:
```bash
python3 motor_credito_avancado.py
```
O console testificará o Efeito Ingênuo deturpado comparado ao verdadeiro Causal Estimator revelado pelo mecanismo ortogonal, em sincronia integral com as teorias lecionadas nos capítulos avaçandos.
