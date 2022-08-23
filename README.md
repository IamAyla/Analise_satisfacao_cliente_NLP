# Análise de satisfação dos clientes com PNL

**Problema de negócio**: entendendo os clientes...

* É necessário compreender as avaliações dos clientes para o sucesso da empresa, além disso é preciso avaliar o sentimento do cliente em relação à marca.

* A análise dos comentários nos ajudara a discernir as preferências dos clientes. Esses insights podem ser utilizados para melhorar o serviço e a experiência do cliente.

**Fonte de dados**: Este conjunto de dados é público dos pedidos realizados na Olist Store. São dados comerciais reais, e foram anonimizados e as referências às empresas e parceiros foram substítuidas pelos nomes das grandes casas de Game of Thrones.

[Acesse aqui a base de dados no Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**Pré-processamento**: Realizaremos algumas modificações iniciais na EDA e no conjunto de dados. O conjunto de dados de revisão tem uma quantidade significativa de NAn no texto e títulos das revisões, portanto, eliminaremos estes valores ausentes.

Seguindo as tarefas da PNL, implementaremos algumas etapas de pré-processamento necessárias que envolvem: 

* Transfromar os dados das revisões removendo stopwords, usando o módulo de expressão regular para aceitar apenas letras, tokenizando o texto e tornando todas as palavras em minúscula para consistência. Nesse caso, teriamos que remover palavras irrelevantes em português.

Concluimos que 58% dos clientes não deixaram comentários e apenas 11,7% se preocuparam em dar títulos aos seus comentários.

**Visualização dos dados**: após o pré-processamento de nossos dados, é hora de visualizar nosso texto de revisão usando Wordclouds que consiste em uma representação visual dos dados do texto envolvidos e mostra a importância das palavras pelo tamanho da fonte.

1. Dos unigramas e trigramas podemos afirmar que a maioria dos clientes ficou satisfeita com o serviço da entrega e com a qualidade dos produtos
2. Entretanto, existem os que não ficaram satisfeitos com os serviços prestados e vamos nos aprofundar nisto.

No gráficos acima após remoção os valores NaN, cerca de 10.000 pessoas deram avaliações 1 estrela, enquanto um pouco mais de 20.000 pessoas deram avaliações de 5 estrelas, o que representa:

1. Cerca de 36% dos revisores de 5 deram comentários, enquanto que 79% dos revisores de 1 estrela comentaram. Um cliente é mais propenso a fazer comentários quando insatisfeito
2. Para entender os clientes insatisfeitos construiremos um modelo de análise de sentimento.

## Máquina Preditiva de Análise de Sentimentos

Este é um caso de aprendizado supervisionado, portanto, criaremos uma nova coluna representando a pontuação do sentimento (1 ou 0). 1 para palavras positivas e 0 para palavras negativas. Excluindo a pontuação de revisão de 3 pontos que representa neutra e incluindo 1 e 2 como palavras negativas e 4 e 5 na revisão de palavras positivas.

Temos  71% de emoções positivas e 29% de emoções negativasnas variáveis teste.
Utilizamos o algoritmo de regressão linear com acurácia de 92%.

**Insights**: Poderíamos pedir pelo menos 3 ou 4 plavras para os clientesvisando se aprofundar quanto a satisfação ou não do cliente com o problema real escolhido.

