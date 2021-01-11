---
title: Teste de Normalidade com Python
author: Natália Amorim
date: 2021-01-11 14:38:00 +0800
categories: [Data Science, Teste de Hipóteses]
tags: [Data Science]
toc: false
---

Muitos métodos estatísticos que utilizamos para análise de dados e tomada de decisões assumem que nossos dados seguem a distribuição normal.

Porém, é sempre bom fazer uma verificação desta suposição, uma vez que podemos realizar previsões que não se aproximam nem um pouco da realidade quando utilizamos métodos que assumem normalidade a dados que não "são normais".

Este artigo se divide em duas partes: A primeira dá uma breve explicação sobre a distribuição normal e a segunda consiste em um exemplo prático de teste de normalidade de um conjunto de dados. Fique à vontade para ler o que lhe for mais conveniente.

# O que é a Distribuição Normal?

Imagine que vamos coletar dados sobre o peso de recém-nascidos de uma cidade durante um certo período.

Quando nossa planilha tiver um número bem grande de recém-nascidos com seus respectivos pesos, poderemos plotar um histograma de frequências destes valores.

Vamos supor que plotamos o histograma de frequências e observamos um comportamento como este:

![](/assets/img/posts/testeNormal/DistNormal.png)
*Fonte: http://leg.ufpr.br/~shimakur/CE701/node36.html*

Vamos interpretar algumas coisas sobre estes dados:

- Existe uma grande quantidade de recém-nascidos que têm peso próximo a 3000 g.

- A média dos pesos de recém-nascidos coincide (ou está muito próxima) com o peso da maioria dos recém-nascidos.

- A frequência de recém-nascidos com peso que se afastam do valor médio é menor quanto mais a seu peso se afasta da média.

Quando isso acontece, dizemos que temos um atributo (peso de um conjunto de recém-nascidos) que segue a Distribuição Normal.

Você, com certeza, já ouviu muito sobre a distribuição normal e deve estar se perguntando o motivo dela ser tão popular. A razão é simples: <b>As medidas produzidas aleatoriamente em diversos processos seguem a distribuição normal.</b>

Agora que você já tem uma noção preliminar do que é a distribuição normal, podemos defini-la de uma maneira mais formal:

> "A distribuição normal de probabilidade é uma distribuição de probabilidade contínua que é simétrica em relação à média e mesocurtica e assíntota em relação ao eixo das abcissas em ambas as direções." (CASTANHEIRA, 2012)

Com isso, nós podemos fazer uma afirmação:

- A probabilidade de ocorrência de indivíduos com alturas pesos próximos ao peso médio é maior do que os demais.

# Exemplo Prático: Teste de Normalidade com Python

Em nossos projetos, podemos verificar se a distribuição dos dados com os quais trabalhamos é normal. Com python este processo é bem simples.

Mas primeiro, certifique-se de que você tenha os requisitos necessários para fazer este exemplo prático:

- Python 3.x
- Pandas, Numpy e Scipy
- [Faça download dos dados aqui](https://storage.googleapis.com/kaggle-data-sets/13720/18513/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201218%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201218T191508Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=28e517efbd8b6b8eb6179832e0388befd0d09fcec624d93b9f9ae914c1f4624316dc3102de8e7e60bfaa327c97d0a50e4903c8ddac27d1e489ad5514a970fdbbafb0e179178afceb8c5b93ea854a1d48a0d3b530887f4e6a8b152a24e8363e012c86da9497a533970f93b2d1d9fb4718f5361908bd9c2ce840f4a67ef66cd2b4cef352540588151734901ef046254fb81fce5deb46e18df330d4f32543babd9840e79e4f83a558fc363881aea6c6e6b71330ca8f0bf16a7e5cb911102d0ac8639dd8cbad250f65b8dd7cd97743c3a65e07ee639152f2f36b77f49ee020adffef37efdff79b23655dcc6420ccdafead0d40714b5fbd1f56f56ac46238ac7f9b4b)



- Lembre-se de descompactar os dados para fazer este exemplo prático!

Vamos supor que trabalhamos em uma companhia de seguros de vida e precisamos, por  algum motivo, realizar alguns procedimentos utilizando este conjunto de dados. Estamos interessados em saber se a distribuição dos dados na coluna "bmi" é normal. Ou seja, queremos saber se o índice de massa corporal dos pacientes segue distribuição normal.

Para isso, vamos utilizar este algoritmo:

```python
dados = pd.read_csv('insurance.csv')

#Verificando as primeiras 5 linhas do conjunto de dados
print(dados.head())
print(dados.shape)

print(dados.dtypes)

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(dados['bmi'], color="dodgerblue", label="Compact",)

plt.show()

alpha = 0.05
k2, p = normaltest(dados['bmi'])

#Hipotese nula: Os dados seguem distribuição normal
#Hipotese alternativa: Os dados não seguem distribuição normal

if p < alpha:
    print("A Hipótese Nula pode ser rejeitada")
else:
    print("A hipótese nula não pode ser rejeitada")
```

## Lendo o arquivo e conhecendo os dados

```python
dados = pd.read_csv('insurance.csv')

#Verificando as primeiras 5 linhas do conjunto de dados
print(dados.head())
print(dados.shape)

print(dados.dtypes)
```

As primeiras linhas do código acima começam com alguns comandos que são muito úteis para conhecermos os dados.

Com o comando ```print(dados.head())``` podemos visualizar as cinco primeiras linhas de cada coluna existente em nosso dataset. Este comando resulta na seguinte exibição no seu terminal:

```
   age     sex     bmi  children smoker     region      charges
0   19  female  27.900         0    yes  southwest  16884.92400
1   18    male  33.770         1     no  southeast   1725.55230
2   28    male  33.000         3     no  southeast   4449.46200
3   33    male  22.705         0     no  northwest  21984.47061
4   32    male  28.880         0     no  northwest   3866.85520
```

Os comandos ```print(dados.shape)``` e ```print(dados.dtypes)``` nos ajudam a verificar quantas linhas e colunas o dataset possui e também qual o tipo de dado armazenado em cada coluna. Se você observar no terminal, esses comandos resultaram em:

```
(1338, 7)
age           int64
sex          object
bmi         float64
children      int64
smoker       object
region       object
charges     float64
dtype: object
```

Com isso nós sabemos que o dataset tem 1338 linhas e 7 colunas. Cada linha representa um cliente e cada coluna tem uma informação sobre este cliente.

A coluna que nos interessa, "bmi" (índice de massa corporal), tem seus dados representados em float64.

## Plotando o Histograma de Frequências

Neste trecho de código nós plotamos um histograma de frequências dos valores de bmi. Utilizamos a matplotlib e a seaborn para esta tarefa.

```python
plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(dados['bmi'], color="dodgerblue", label="Compact",)

plt.show()
```

E o resultado que obtivemos foi:

![](/assets/img/posts/testeNormal/bmi.png)

Olhando para esta figura, nós podemos pensar que os valores de índices de massa corporal dos clientes de nossa empresa de seguros seguem distribuição normal, porém, para validar esta afirmação, precisamos ainda realizar o teste de normalidade.

## Fazendo o Teste de Normalidade

```python
alpha = 0.05
k2, p = normaltest(dados['bmi'])

#Hipotese nula: Os dados seguem distribuição normal
#Hipotese alternativa: Os dados não seguem distribuição normal

if p < alpha:
    print("A Hipótese Nula pode ser rejeitada")
else:
    print("A hipótese nula não pode ser rejeitada")
```

E é aqui que acontece o que estávamos ansiosos para ver!

Utilizamos o comando ```normaltest(dados['bmi'])``` para verificar a normalidade da nossa amostra. Neste teste calculamos o p-valor e o k2. Além disso, definimos um valor alpha que serve como um threshold para rejeitar ou não a hipótese nula.

Vamos falar um pouco mais de cada um destes valores:

- <b>p-valor:</b> Pode também ser citado como valor-p. É a probabilidade de se obter um efeito tão extremo quanto o que está ocorrendo em nossos dados, assumindo que a hipótese nula é verdadeira. (Qual a probabilidade da distribuição que observamos naquele histograma ocorrer? Este é o p-valor).

- <b>alpha:</b> É o nível de significância, isto é, a probabilidade de rejeitarmos a hipótese nula quando ela é verdadeira. (Neste caso, a probabilidade de concluirmos que os dados não seguem a distribuição normal, quando na verdade seguem).

- <b>k2:</b> Este valor é, na verdade, a soma de dois termos elevados ao quadrado: s² + k². Sendo s o valor z obtido através do teste de assimetria (skewtest) e k é o valor da estatística z obtido pelo teste de curtose.

Rejeitamos a hipótese nula quando o valor-p for menor do que o nível de significância do nosso teste. Em outras palavras, estamos dizendo que se a probabilidade de ocorrência de valores como o que temos em nosso dataset (na coluna "bmi") é maior do que a probabilidade de cometermos o erro de rejeitar a hipótese nula quando ela é verdadeira, podemos assumir que H0 é verdadeira. Este teste é conhecido como o <b>Teste de Normalidade de Kolmogorov-Smirnov</b>.

No caso do nosso conjunto de dados, podemos ver no terminal que o resultado foi:

```
A Hipótese Nula pode ser rejeitada
```

Ou seja, o valor-p foi menor do que nosso nível de significância, logo, a probabilidade de obtermos dados como estes é muito pequena. Assim, podemos concluir que os valores de índice de massa corporal dos clientes da empresa de seguros na qual trabalhamos não seguem distribuição normal.