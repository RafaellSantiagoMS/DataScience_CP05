# Checkpoint 5 - Preço de apartamentos em São Paulo

Data Science & Statistical Computing - FIAP 2026

Grupo: Enzo Augusto (RM562249), Gustavo Neres (RM561785), Rafaell Santiago (RM563486), Sebastian Iriarte (RM563619)

Continuação do CP04. Usamos a mesma base e o mesmo problema (estimar o preço de venda de apartamentos em São Paulo), agora comparando Random Forest, XGBoost e LightGBM com validação cruzada, Grid Search e Optuna.

## Dados

São Paulo Real Estate - Sale/Rent - April 2019 (Kaggle): https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019. O notebook lê o CSV direto por URL. Depois do tratamento ficaram 6.240 anúncios de venda.

## Resultado

Os três modelos usaram o mesmo pipeline, os mesmos 5 folds de validação cruzada e o RMSE como métrica principal. O teste (20% da base) só foi usado no final.

| Configuração | RMSE na validação cruzada |
|---|---|
| LightGBM (Optuna) | R$ 221,1 mil |
| LightGBM (Grid Search) | R$ 223,6 mil |
| XGBoost (Optuna) | R$ 227,8 mil |
| Random Forest (Optuna) | R$ 239,4 mil |

Modelo final: LightGBM ajustado pelo Optuna. No teste: RMSE de R$ 197,9 mil, MAE de R$ 88,0 mil e R² de 0,93. A tabela completa com as nove configurações está na seção 6 do notebook.

## Arquivos

- `Checkpoint05_RF_XGBoost_LightGBM.ipynb`: notebook com os exercícios 1 a 7
- `app.py`: aplicação Streamlit
- `requirements.txt`: dependências
- `base_tratada.csv`: base depois do tratamento (gerada pelo notebook)
- `modelo/modelo_final.joblib` e `modelo/info.json`: modelo final e informações usadas pelo app (gerados pelo notebook)

## Como rodar

```
pip install -r requirements.txt
streamlit run app.py
```

O arquivo do modelo precisa das mesmas versões de scikit-learn, xgboost e lightgbm usadas para gerá-lo. Se o notebook for rodado de novo em outro ambiente (como o Colab), ajuste as versões no requirements.txt para as desse ambiente.

## Limitações

Os preços são de anúncios de 2019, e não de vendas efetivas. O erro médio é de cerca de 14% do valor do imóvel, então o modelo serve como referência de preço, e não como avaliação de um imóvel específico.

## Links

- GitHub: https://github.com/RafaellSantiagoMS/DataScience_CP05
- Aplicação Streamlit: https://datasciencecp05-tnjy9hyxbr6y7vqhhibo4p.streamlit.app/
