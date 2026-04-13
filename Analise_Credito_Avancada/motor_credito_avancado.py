import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class CreditRiskEngine:
    """
    Motor de Análise de Crédito Avançado estruturado com os preceitos do livro:
    - Previsão de Risco usando Ensemble Estocástico (XGBoost)
    - Interpretabilidade Algorítmica (SHAP Values)
    - Inferência Causal via Double Machine Learning (DML)
    """
    def __init__(self, n_samples=25000):
        self.n_samples = n_samples
        self.df = None
        self.xgb_model = None

    def generate_complex_synthetic_data(self):
        """ Geração de um dataset multi-dimensional com interações não lineares severas e confundimento causal. """
        np.random.seed(42)
        n = self.n_samples
        
        # Variáveis Observáveis (Covariáveis Financeiras e Demográficas)
        idade = np.round(np.random.normal(40, 12, n))
        salario = np.random.lognormal(mean=9, sigma=0.8, size=n)
        anos_emprego = np.round(np.random.exponential(scale=5, size=n))
        score_bureau = np.random.normal(600, 100, n)
        
        # Variável Latente (Risco Comportamental Invisível)
        risco_oculto = np.random.normal(0, 1, n)
        
        # Correlações Ocultas (Confounding)
        score_bureau = score_bureau - (risco_oculto * 50) 
        
        # Intervenção (Tratamento D: Oferecer Período de Carência/Extensão de Limite)
        # O tratamento NÃO é aleatório. Clientes com bom bureau e salário recebem mais, gerando Viés de Seleção.
        prob_tratamento = 1 / (1 + np.exp(-(-2 + 0.0001*salario + 0.005*score_bureau)))
        tratamento = np.random.binomial(1, prob_tratamento)
        
        # Efeito Causal Verdadeiro da Intervenção no Default: Reduz a chance de Default?
        # Suponha que o verdadeiro Efeito Causal seja -0.05 (Reduz a probabilidade de default em 5%)
        true_causal_effect = -0.05
        
        # Variável de Resposta (Inadimplência - Default Y)
        # É uma função altamente não-linear das covariáveis + Tratamento + Choque
        logit_Y = (-1.5 
                   - 0.03 * idade 
                   - 0.00005 * salario 
                   - 0.1 * anos_emprego
                   - 0.004 * score_bureau
                   + 1.5 * (score_bureau < 500) # Relação Não-Linear Intensa
                   + 0.5 * risco_oculto
                   + tratamento * true_causal_effect)
        
        prob_default = 1 / (1 + np.exp(-logit_Y))
        default = np.random.binomial(1, prob_default)
        
        self.df = pd.DataFrame({
            'idade': idade,
            'salario': salario,
            'anos_emprego': anos_emprego,
            'score_bureau': score_bureau,
            'tratamento_limite': tratamento,
            'default': default
        })
        
        print(f"Dataset Gerado: {self.df.shape}. Taxa de Inadimplência global: {self.df['default'].mean():.2%}")
        
    def train_probability_of_default(self):
        """ Etapa 1: Previsão Pura (Machine Learning) com XGBoost para o PD (Probability of Default) """
        print("\n--- Treinamento Preditivo: XGBoost ---")
        X = self.df.drop(columns=['default'])
        y = self.df['default']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            objective='binary:logistic',
            eval_metric='auc'
        )
        self.xgb_model.fit(X_train, y_train)
        
        y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        print(f"Desempenho Out-of-Sample -> AUC-ROC: {auc:.4f} | Brier Score: {brier:.4f}")
        return X_train
        
    def interpretability_shap(self, X_sample):
        """ Etapa 2: Explicabilidade Transparente para Crédito (Evitar Caixa Preta) """
        print("\n--- Explicabilidade: Valores SHAP ---")
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculando a importância global
        feature_importance = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importância Média Absoluta': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='Importância Média Absoluta', ascending=False)
        
        print(feature_importance.to_string(index=False))

    def causal_inference_dml(self):
        """ Etapa 3: Double Machine Learning (DML) para estimar o impacto causal da Política de Crédito """
        print("\n--- Inferência Causal Analítica: Double Machine Learning (DML) ---")
        print("Objetivo: Qual o Efeito PURO da `tratamento_limite` sobre o `default`?")
        
        Y = self.df['default'].values
        D = self.df['tratamento_limite'].values 
        X = self.df.drop(columns=['default', 'tratamento_limite']).values
        
        # Abordagem Naive (Regressão Simples Viesada pelas correlações ocultas)
        naive_model = sm.OLS(Y, sm.add_constant(D)).fit()
        print(f"Efeito OLS Ingênuo: {naive_model.params[1]:.4f} (Violou o Back-Door Criterion!)")
        
        # Algoritmo Double Machine Learning com Cross-Fitting
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        res_Y = np.zeros(len(Y))
        res_D = np.zeros(len(D))
        
        for train_idx, test_idx in kf.split(X):
            # Modelos Nuisance (Bosques Aleatórios para Alta Dimensionalidade)
            model_Y = RandomForestClassifier(n_estimators=100, max_depth=5)
            model_D = RandomForestClassifier(n_estimators=100, max_depth=5)
            
            # Predict E[Y|X] e Residualize
            model_Y.fit(X[train_idx], Y[train_idx])
            res_Y[test_idx] = Y[test_idx] - model_Y.predict_proba(X[test_idx])[:, 1]
            
            # Predict E[D|X] e Residualize
            model_D.fit(X[train_idx], D[train_idx])
            res_D[test_idx] = D[test_idx] - model_D.predict_proba(X[test_idx])[:, 1]
            
        # Terceira Fase: Ortogonalização de Neyman (Regressão dos Resíduos)
        dml_model = sm.OLS(res_Y, res_D).fit()
        print(f"Efeito Causal Isolado via DML: {dml_model.params[0]:.4f}")
        print("Interpretação: Controlando não-linearmente pelo viés de seleção estatístico, o tratamento causou de fato este decréscimo assintótico estrutural na inflação do risco.")

if __name__ == "__main__":
    engine = CreditRiskEngine()
    engine.generate_complex_synthetic_data()
    X_train = engine.train_probability_of_default()
    engine.interpretability_shap(X_train.sample(1000, random_state=42))
    engine.causal_inference_dml()
