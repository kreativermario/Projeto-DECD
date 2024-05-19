# Projeto DECD: E-Redes
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kreativermario/Projeto-DECD/blob/development/index.ipynb)


----

## Conteúdo

1. Data understanding
   1. [Tratamento da coleção - Consumos faturados](./notebooks/tratamento_3_consumos_faturados_por_municipio_10_anos.ipynb)
   2. [Tratamento da coleção - Caracterizacao CPEs](./notebooks/tratamento_20_caracterizacao_pes_contrato_ativo.ipynb)
   3. [Tratamento da coleção - Diagrama de carga](./notebooks/tratamento_22_diagrama_de_carga_por_instalacao.ipynb)
   4. [Junção dos datasets da E-Redes](./notebooks/tratamento_juncao.ipynb)
   5. [Tratamento da coleção - Pordata Densidade Populacional](./notebooks/tratamento_pordata_populacional.ipynb)
   6. [Tratamento da coleção - Pordata Industrias](./notebooks/tratamento_pordata_industrias.ipynb)
   7. [Visualização e análise do dataset merged](./notebooks/data_understanding.ipynb)
2. Preparação de dados
   1. [Preparação dataset totalmente categórico](./notebooks/data_preparation_categoric.ipynb)
      - Foi decidido que não era ideal para o nosso target de análise
   2. [Preparação dataset totalmente numérico](./notebooks/data_preparation_numeric.ipynb)
      - Foi decidido utilizar um dataset totalmente númerico
      - Na segunda parte, adicionou-se dados de densidade populacional devido às distribuições assimétrica porque existem muitos concelhos com pouca densidade populacional.
3. Aprendizagem não supervisionada
   - Foram criados dois subsets na prepação de dados de modo a podermos analisar os dois tipos de tensões de energia. 
   1. [Clustering Baixa Tensão](./notebooks/clustering_low_tensions.ipynb)
   2. [Clustering Alta Tensão](./notebooks/clustering_high_tension.ipynb)
4. Aprendizagem supervisionada
   1. [Regressão Baixa Tensão](./notebooks/supervised_learning_low_tension.ipynb)
   2. [Classificação Alta Tensão](./notebooks/supervised_learning_high_tension.ipynb)

## Recursos Adicionais

- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [The Python Tutorial](https://docs.python.org/3/tutorial/index.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python Pandas Tutorial: A Complete Introduction for Beginners](https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/)
- [Pandas Exercises, Practice, Solution](https://www.w3resource.com/python-exercises/pandas/index.php)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html) 