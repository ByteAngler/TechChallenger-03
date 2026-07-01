# Predicao de Evasao de Estudantes

Projeto desenvolvido para a prova substitutiva da Fase 3 de Machine Learning Engineering.

O objetivo e criar um modelo de Machine Learning capaz de prever se um estudante tem risco de evasao a partir de dados academicos, financeiros e cadastrais.

## Visao geral

A base utilizada foi `StudentsPrepared.xlsx`, localizada na pasta `data/`.

Cada linha representa um estudante. A coluna original `Target` indica a situacao do aluno:

- `Desistente`
- `Graduado`
- `Matriculado`

Para resolver o problema como classificacao binaria, a coluna foi transformada em:

| Situacao original | Classe usada no modelo |
|---|---:|
| Desistente | 1 |
| Graduado | 0 |
| Matriculado | 0 |

Assim, o modelo aprende a identificar a classe `1`, que representa estudantes com evasao.

## O que foi feito

O projeto foi organizado em etapas simples:

1. Leitura da planilha com os dados dos estudantes.
2. Criacao da variavel alvo binaria `Evasao`.
3. Tratamento de colunas numericas com valores inconsistentes.
4. Separacao das variaveis por tipo: categoricas, numericas e booleanas.
5. Treinamento de um modelo de classificacao.
6. Avaliacao com conjunto de teste e validacao cruzada.
7. Salvamento do modelo treinado.
8. Criacao de uma aplicacao Streamlit para usar o modelo.

## Tratamento dos dados

Durante a exploracao da base, foram encontrados valores muito altos nas colunas:

- `UnidadesCurriculares1SemestreGrau`
- `UnidadesCurriculares2SemestreGrau`

Essas colunas representam notas ou graus academicos. Alguns valores apareciam em escala impossivel, chegando a ordem de `10^16`.

Como os valores corrigidos retornavam para uma escala academica plausivel de `0` a `20`, foi aplicada uma correcao de magnitude: os valores maiores que 20 foram divididos por 10 repetidamente ate retornarem ao intervalo esperado.

Tambem foram aplicados tratamentos diferentes conforme o tipo de variavel:

| Tipo de variavel | Tratamento |
|---|---|
| Categoricas | OneHotEncoder |
| Numericas continuas | StandardScaler |
| Booleanas 0/1 | Mantidas sem transformacao |

As colunas booleanas mantidas como 0/1 foram:

- `NecessidadesEspeciais`
- `Devedor`
- `MensalidadesEmDia`
- `Bolsista`
- `International`

## Modelo utilizado

O modelo escolhido foi uma Random Forest, usando um pipeline do scikit-learn.

Esse pipeline inclui tanto o pre-processamento quanto o modelo. Isso significa que, ao chamar `predict`, os dados passam automaticamente pelas transformacoes necessarias antes da predicao.

Principais configuracoes do modelo:

- `n_estimators=300`
- `max_depth=8`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `class_weight="balanced"`
- `random_state=42`

A opcao `class_weight="balanced"` foi usada porque existem mais estudantes sem evasao do que estudantes desistentes.

## Resultados

O modelo final foi avaliado em um conjunto de teste separado, com 20% da base.

| Metrica | Resultado |
|---|---:|
| Accuracy | 0.8701 |
| Precision | 0.7845 |
| Recall | 0.8204 |
| F1-score | 0.8021 |
| ROC AUC | 0.9265 |

Matriz de confusao:

```text
[[537, 64],
 [ 51, 233]]
```

Tambem foi realizada validacao cruzada com 5 divisoes:

| Metrica | Treino | Validacao |
|---|---:|---:|
| Accuracy | 0.8738 | 0.8483 |
| Precision | 0.7861 | 0.7496 |
| Recall | 0.8344 | 0.7933 |
| F1-score | 0.8095 | 0.7706 |
| ROC AUC | 0.9385 | 0.9079 |

## Conclusao da analise

O modelo apresentou bom desempenho para identificar estudantes com risco de evasao.

O ROC AUC de `0.9265` no teste indica boa capacidade de separacao entre estudantes evadidos e nao evadidos. O recall de `0.8204` tambem e importante, pois mostra que o modelo consegue encontrar boa parte dos casos de evasao.

Nao ha sinal de underfitting, pois o modelo apresenta desempenho consistente e superior ao esperado para um modelo que nao capturasse os padroes dos dados.

Tambem nao ha sinal forte de overfitting. Existe uma diferenca entre treino e validacao, mas ela e moderada, e o desempenho em validacao continua bom.

## Estrutura do projeto

```text
.
|-- app/
|   `-- streamlit_app.py
|-- data/
|   `-- StudentsPrepared.xlsx
|-- models/
|   `-- model.joblib
|-- notebooks/
|   `-- explore.ipynb
|-- reports/
|   `-- metrics.json
|-- src/
|   |-- data_processing.py
|   |-- modeling.py
|   `-- train.py
|-- README.md
`-- requirements.txt
```

## Como rodar o projeto

Crie e ative um ambiente virtual. Depois instale as dependencias:

```bash
pip install -r requirements.txt
```

Para treinar o modelo novamente:

```bash
python -m src.train
```

Esse comando gera:

- `models/model.joblib`
- `reports/metrics.json`

Para rodar a aplicacao Streamlit:

```bash
python -m streamlit run app/streamlit_app.py
```

Depois, acesse o endereco mostrado no terminal.

## Aplicacao Streamlit

A aplicacao permite preencher os dados de um estudante e obter:

- probabilidade de evasao;
- classificacao final: com risco de evasao ou sem indicacao de evasao.

O app usa o mesmo pipeline treinado no projeto, garantindo que os dados preenchidos passem pelo mesmo pre-processamento utilizado durante o treino.
