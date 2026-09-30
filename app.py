"""
Checkpoint 5 - previsão do preço de apartamentos em São Paulo
Grupo: Enzo Augusto (RM562249), Gustavo Neres (RM561785),
       Rafaell Santiago (RM563486), Sebastian Iriarte (RM563619)

Executar com: streamlit run app.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Preço de apartamentos em SP", layout="wide")


@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo/modelo_final.joblib")


@st.cache_data
def carregar_info():
    with open("modelo/info.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def carregar_base():
    return pd.read_csv("base_tratada.csv")


# mesma função usada no notebook
def limpar_imovel(df):
    df = df.copy()
    df["condominio"] = df["condominio"].replace(0, np.nan)
    fora_sp = ~(df["latitude"].between(-24, -23.3) & df["longitude"].between(-47, -46.3))
    df.loc[fora_sp, ["latitude", "longitude"]] = np.nan
    return df


modelo = carregar_modelo()
info = carregar_info()
base = carregar_base()

st.title("Quanto vale um apartamento em São Paulo?")
st.write(
    "Estimativa do preço de venda anunciado de apartamentos em São Paulo. "
    "Dados: São Paulo Real Estate - Sale/Rent - April 2019 (Kaggle). "
    f"Modelo: {info['modelo']}."
)

c1, c2, c3 = st.columns(3)
c1.metric("RMSE no teste", f"R$ {info['rmse_teste']:,.0f}")
c2.metric("MAE no teste", f"R$ {info['mae_teste']:,.0f}")
c3.metric("R² no teste", f"{info['r2_teste']:.3f}")

if "entrada" not in st.session_state:
    st.session_state.entrada = None
if st.button("Usar o anúncio do teste de paridade"):
    st.session_state.entrada = info["paridade_entrada"]


def padrao(campo, valor):
    e = st.session_state.entrada
    if e is None or e.get(campo) is None:
        return valor
    return e[campo]


st.subheader("Dados do apartamento")
col1, col2, col3 = st.columns(3)
with col1:
    distrito = st.selectbox("Distrito", info["distritos"], index=info["distritos"].index(padrao("distrito", "Moema")))
    area = st.number_input("Área (m²)", 20, 1000, int(padrao("area_m2", 70)))
    quartos = st.number_input("Quartos", 1, 10, int(padrao("quartos", 2)))
    suites = st.number_input("Suítes", 0, 10, int(padrao("suites", 1)))
with col2:
    banheiros = st.number_input("Banheiros", 1, 10, int(padrao("banheiros", 2)))
    vagas = st.number_input("Vagas", 0, 10, int(padrao("vagas", 1)))
    condominio = st.number_input("Condomínio (R$/mês, 0 se não souber)", 0, 20000, int(padrao("condominio", 0)))
with col3:
    elevador = st.checkbox("Elevador", bool(padrao("elevador", 1)))
    piscina = st.checkbox("Piscina", bool(padrao("piscina", 0)))
    mobiliado = st.checkbox("Mobiliado", bool(padrao("mobiliado", 0)))
    novo = st.checkbox("Imóvel novo", bool(padrao("novo", 0)))

# localização: por padrão usamos o centro típico do distrito (mediana das coordenadas da base)
centro = base.groupby("distrito")[["latitude", "longitude"]].median()
tem_coord = st.session_state.entrada is not None and st.session_state.entrada.get("latitude") is not None
with st.expander("Localização exata (opcional)"):
    st.write("Se não souber, deixe desmarcado: o app usa o centro típico do distrito escolhido.")
    usar_coord = st.checkbox("Informar latitude e longitude", value=tem_coord)
    if usar_coord:
        latitude = st.number_input("Latitude", -24.0, -23.3, float(padrao("latitude", centro.loc[distrito, "latitude"])), format="%.6f")
        longitude = st.number_input("Longitude", -47.0, -46.3, float(padrao("longitude", centro.loc[distrito, "longitude"])), format="%.6f")
if not usar_coord:
    latitude, longitude = centro.loc[distrito, "latitude"], centro.loc[distrito, "longitude"]

entrada = pd.DataFrame([{
    "area_m2": area, "quartos": quartos, "banheiros": banheiros, "suites": suites, "vagas": vagas,
    "condominio": condominio, "elevador": int(elevador), "mobiliado": int(mobiliado),
    "piscina": int(piscina), "novo": int(novo), "latitude": latitude, "longitude": longitude,
    "distrito": distrito,
}]).astype({"condominio": float, "latitude": float, "longitude": float})

if suites > quartos:
    st.warning("O número de suítes é maior que o de quartos.")
for campo, nome in [("area_m2", "Área"), ("quartos", "Quartos"), ("banheiros", "Banheiros"), ("vagas", "Vagas")]:
    minimo, maximo = info["faixas"][campo]
    if not minimo <= entrada[campo].iloc[0] <= maximo:
        st.warning(f"{nome} fora da faixa vista no treino ({minimo:g} a {maximo:g}); a previsão pode ser pouco confiável.")

previsao = modelo.predict(limpar_imovel(entrada))[0]
st.subheader(f"Preço estimado: R$ {previsao:,.2f}")
st.caption(f"Em dados novos, o modelo erra em média cerca de R$ {info['mae_teste']:,.0f}.")

if st.session_state.entrada is not None:
    st.info(f"Teste de paridade - notebook: R$ {info['paridade_previsao_notebook']:,.2f} | app: R$ {previsao:,.2f}")

with st.expander("Ver amostra da base"):
    st.dataframe(base.sample(10, random_state=1))
    st.dataframe(base.describe())
