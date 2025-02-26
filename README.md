# House Prices - Advanced Regression Techniques (Kaggle Competition) 
[🇩🇪 Deutsche Version](#german-version)

This repository contains the solution and analysis for the Kaggle competition **["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**. The objective of this competition is to predict the final sale price of homes based on a variety of features, such as square footage, number of rooms, location. See the summary report [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/reports/summary_report.md).

![](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/reports/images/sale_prices.png)

My main goal was to improve [AutoGluon](https://autogluon.mxnet.io/)’s prediction score through data engineering and configuration optimization.

## Key Findings:
- Improved AutoGluon's baseline prediction score by **10.1%** (Table 1).
- Achieved a **top 3% ranking** among nearly 4,200 competitors in the Kaggle competition.

### Table 1: Approaches Employed to Improve Model Performance

| Data                        | AutoGluon Presets, Time (min) | Score (RMSE) | Score Improvement (%) |
|-----------------------------|-------------------------------|--------------|-----------------------|
| Original data               | Default                       | 0.12748      | -                     |
| Original data               | Good, 30                      | 0.12036      | 5.6                   |
| - Feature 'Id'              | Good, 30                      | 0.11632      | 8.8                   |
| + NaN Treatment             | Good, 30                      | 0.11539      | 9.5                   |
| + Median Price (25 Closest) | Good, 30                      | 0.11459      | 10.1                  |

---

## Project Overview

Below is a succinct description of the steps developed in this project.

## [Adding Latitude and Longitude to the Dataset](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/01_ah_merging_datas.ipynb)  
- Combined latitude and longitude information from [R's tidymodels](https://www.tmwr.org/ames) dataset with Kaggle's dataset. While this did not improve the test dataset score (Table 2), it enabled the calculation of the median sale price of the 25 closest houses.
- Produced visualizations (see [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_1.ipynb)).

## [Exploratory Data Analysis and Data Engineering](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/02_ah_EDA.ipynb)  
At this stage, I went through several procedures:  
1. **Sanity check**
- Inspected for duplicate entries, standardized text data, and analyzed missing values.
- Dropped the 'Id' feature, improving the test dataset score by **3.2%** (Table 1).
- Tested dropping categorical features that lacked a second representative category in the training dataset, but this approach did not improve the score on the test dataset (Table 2).

2. **Missing values treatment**
- Implemented a custom approach for handling missing values, resulting in a **0.7%** improvement over AutoGluon's default method (Table 1).
  
3. **Categorical features**
- Tested combining underrepresented categories and grouping neighborhoods, but neither approach improved the score (Table 2).

4. **Numeric features**
- Experimented with delta features, seasonal features, and area-related features, but none improved the score (Table 2).

5. **Numeric vs. non-numeric features**
- Explored an alternative encoding method for ordinal features (see [here](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_2.ipynb)), but this did not improve the score (Table 2).
  
6. **Outliers**
- Evaluated the impact of removing extreme outliers, but this did not improve the score (Table 2).
  
7. **New feature based on location**
- Introduced a feature representing the median sale price of the 25 closest houses, with the number of neighboring houses determined through experimentation (values tested: 5, 25, 50, 75, and 100). This feature yielded a **0.6%** improvement (Table 1).

8. **New feature based on economic data**
- Tested two economic indexes related to housing prices, but these features did not improve model performance (Table 2).

9. **Interaction between original features**
- Experimented with creating interaction features by combining pairs of the top numeric/ordinal features (e.g., 'GrLivArea' * 'OverallQual') based on AutoGluon’s feature importance ranking. I tested combinations of the top 3, 5, and 10 features, but this approach did not improve the test dataset score (Table 2). A caveat is that AutoGluon's feature importance ranking is intended for use on the test dataset; however, since I do not have the SalePrice for the test dataset, I applied it to the training dataset instead. 

## [Machine Learning](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/03_ah_MODEL.ipynb)  
- I used **AutoGluon** to perform the machine learning analyses.  
- The analyses were conducted using the `presets='good_quality'` and `time_limit=30 min` (the "good_quality" preset typically completes in under 30 minutes). The `medium_quality`, `high_quality`, and `best_quality` presets did not improve the score on the test dataset (see Table 2).  
- Neither the **AutoGluon pseudolabel model refit** nor **model distillation** improved the score on the test dataset (Table 2).  

### Table 2: Proposed Treatments That Did Not Improve Scores

| **Treatments**                                                                 |
|--------------------------------------------------------------------------------|
| Removing 'Street', 'Utilities', 'RoofMatl', 'Condition2', 'PoolQC'             |
| Longitude and Latitude                                                         |
| Combining Underrepresented Classes                                             |
| Grouping Neighborhoods into Larger Areas                                       |
| Defining Seasons of the Year                                                   |
| Delta Features for Construction, Remodeling, and Selling Years                |
| Total Area, Finished Area, High-Quality Area, and Total Bathrooms              |
| Ordinal Encoding                                                               |
| Removing Extreme Outliers                                                      |
| Case-Shiller U.S. National Home Price Index                                    |
| All-Transactions House Price Index for Ames, IA                                |
| Interaction Features Among Top Features                                        |
| AutoGluon Presets: `medium_quality`, `high_quality`, `best_quality`            |
| AutoGluon Pseudolabel Model Refit and Model Distillation                       |

---

## Tools and Technologies:
- **Libraries**: AutoGluon, Featuretools, Matplotlib, Numpy, Pandas, Seaborn, Scikit-Learn, 

---

## Project organization
```
├── .gitignore                              <- Files and directories to be ignored by Git
|
├── LICENSE                                 <- License type 
|
├── README.md                               <- Main README explaining the project
|
├── data                                    <- Project data files
|   ├── ames_dataset.csv                    <- Original dataset from R's tidymodels
|   ├── ATNHPIUS11180Q.csv                  <- Data for Ames' House Price Index
|   ├── clean_data.csv                      <- Dataset prepared for machine learning analysis
|   ├── CSUSHPINSA.csv                      <- Data for the USA's Case-Shiller Index
|   ├── original_plus_lat_lon.csv           <- Kaggle dataset with added longitude and latitude from tidymodels
|   ├── test.csv                            <- Original test dataset from Kaggle
|   ├── train.csv                           <- Original train dataset from Kaggle
|
├── environments                            <- Requirements files to reproduce the analysis environments
|   ├── autogluon_environment.yml           <- Environment for running '03_ah_MODEL.ipynb' 
|   ├── eda_environment.yml                 <- Environment for running '01_ah_merging_datas.ipynb', '02_ah_EDA.ipynb', and 'ah_appendix_2.ipynb' 
|   ├── maps_environment.yml                <- Environment for running 'ah_appendix_1.ipynb'
|
├── models                                  <- Trained and serialized models, model predictions, or model summaries
|
├── notebooks                               <- Jupyter notebooks
|   ├── 01_ah_merging_datas.ipynb           <- Adding latitude and longitude information to Kaggle's dataset
|   ├── 02_ah_EDA.ipynb                     <- Exploratory data analysis
|   ├── 03_ah_MODEL.ipynb                   <- Machine learning approach
|   ├── ah_appendix_1.ipynb                 <- Creating maps
|   ├── ah_appendix_2.ipynb                 <- Exploring an alternative dataset transformation
|   └── src                                 <- Source code used in this project
|       ├── __init__.py                     <- Makes this a Python module
|       ├── auxiliaries.py                  <- Scripts to compute nearest houses and median sale prices
|       ├── config.py                       <- Basic project configuration
|       ├── eda.py                          <- Scripts for exploratory data analysis and visualizations
|
├── references                              <- Data dictionaries, manuals, and other explanatory materials
|   ├── data_description.txt                <- Description of the dataset as presented on Kaggle
|
├── reports                                 <- Generated analyses in HTML, PDF, LaTeX, etc., and results
|   ├── best_kaggle_prediction.csv          <- Best prediction from Kaggle competition
|   ├── summary_report.md                   <- Summary report
│   └── images                              <- Images used in the project
|      ├── sale_prices_train_test_plot.png  <- Map showing houses with and without prices 
|      ├── sale_prices.png                  <- Map displaying house prices
```

## Contributing
All contributions are welcome!

### Issues
Submit issues for:
- Recommendations or improvements
- Additional analyses or models
- Feature enhancements
- Bug reports

### Pull Requests
- Open an issue before starting work.
- Fork the repository and clone it.
- Create a branch and commit your changes.
- Push your changes and open a pull request for review.


# German Version

# Hauspreise - Fortgeschrittene Regressionsmethoden (Kaggle-Wettbewerb)

Dieses Repository enthält die Lösung und Analyse für den Kaggle-Wettbewerb **["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**. Das Ziel dieses Wettbewerbs ist es, den endgültigen Verkaufspreis von Häusern basierend auf verschiedenen Merkmalen wie Quadratmeterzahl, Anzahl der Zimmer und Lage vorherzusagen. Siehe den zusammenfassenden Bericht [hier](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/reports/summary_report.md).

![](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/reports/images/sale_prices.png)

Mein Hauptziel war es, die Vorhersagegenauigkeit von [AutoGluon](https://autogluon.mxnet.io/) durch Datenengineering und Konfigurationsoptimierung zu verbessern.

## Wichtige Erkenntnisse:
- Verbesserung der Baseline-Vorhersagegenauigkeit von AutoGluon um **10,1 %** (Tabelle 1).
- Erreichen eines **Top-3%-Rangs** unter fast 4.200 Teilnehmern im Kaggle-Wettbewerb.

### Tabelle 1: Angewandte Ansätze zur Verbesserung der Modellleistung

| Daten                        | AutoGluon-Voreinstellungen, Zeit (min) | Score (RMSE) | Verbesserung des Scores (%) |
|-----------------------------|----------------------------------------|--------------|-----------------------------|
| Originaldaten               | Standard                               | 0,12748      | -                           |
| Originaldaten               | Gut, 30                                | 0,12036      | 5,6                         |
| - Merkmal 'Id'              | Gut, 30                                | 0,11632      | 8,8                         |
| + NaN-Behandlung            | Gut, 30                                | 0,11539      | 9,5                         |
| + Medianpreis (25 nächste)  | Gut, 30                                | 0,11459      | 10,1                        |

---

## Projektübersicht

Nachfolgend finden Sie eine kurze Beschreibung der in diesem Projekt entwickelten Schritte.

## [Hinzufügen von Breiten- und Längengraden zum Datensatz](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/01_ah_merging_datas.ipynb)  
- Kombination von Breiten- und Längengradinformationen aus dem [R tidymodels](https://www.tmwr.org/ames)-Datensatz mit dem Kaggle-Datensatz. Obwohl dies die Testdatensatz-Bewertung nicht verbesserte (Tabelle 2), ermöglichte es die Berechnung des medianen Verkaufspreises der 25 nächsten Häuser.
- Erstellung von Visualisierungen (siehe [hier](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_1.ipynb)).

## [Explorative Datenanalyse und Datenengineering](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/02_ah_EDA.ipynb)  
In dieser Phase durchlief ich mehrere Verfahren:  
1. **Plausibilitätsprüfung**
- Überprüfung auf doppelte Einträge, Standardisierung von Textdaten und Analyse fehlender Werte.
- Entfernung des Merkmals 'Id', was die Testdatensatz-Bewertung um **3,2 %** verbesserte (Tabelle 1).
- Test der Entfernung kategorischer Merkmale, die im Trainingsdatensatz keine zweite repräsentative Kategorie aufwiesen, was jedoch die Testdatensatz-Bewertung nicht verbesserte (Tabelle 2).

2. **Behandlung fehlender Werte**
- Implementierung eines benutzerdefinierten Ansatzes zur Behandlung fehlender Werte, was zu einer **0,7 %**-Verbesserung gegenüber der Standardmethode von AutoGluon führte (Tabelle 1).
  
3. **Kategorische Merkmale**
- Test der Kombination unterrepräsentierter Kategorien und Gruppierung von Nachbarschaften, was jedoch die Bewertung nicht verbesserte (Tabelle 2).

4. **Numerische Merkmale**
- Experimente mit Delta-Merkmalen, saisonalen Merkmalen und flächenbezogenen Merkmalen, die jedoch die Bewertung nicht verbesserten (Tabelle 2).

5. **Numerische vs. nicht-numerische Merkmale**
- Erforschung einer alternativen Kodierungsmethode für ordinale Merkmale (siehe [hier](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/ah_appendix_2.ipynb)), was jedoch die Bewertung nicht verbesserte (Tabelle 2).
  
6. **Ausreißer**
- Bewertung der Auswirkungen der Entfernung extremer Ausreißer, was jedoch die Bewertung nicht verbesserte (Tabelle 2).
  
7. **Neues Merkmal basierend auf der Lage**
- Einführung eines Merkmals, das den medianen Verkaufspreis der 25 nächsten Häuser darstellt, wobei die Anzahl der Nachbarhäuser durch Experimente bestimmt wurde (getestete Werte: 5, 25, 50, 75 und 100). Dieses Merkmal führte zu einer **0,6 %**-Verbesserung (Tabelle 1).

8. **Neues Merkmal basierend auf Wirtschaftsdaten**
- Test zweier Wirtschaftsindizes im Zusammenhang mit Hauspreisen, die jedoch die Modellleistung nicht verbesserten (Tabelle 2).

9. **Interaktion zwischen ursprünglichen Merkmalen**
- Experimente mit der Erstellung von Interaktionsmerkmalen durch Kombination von Paaren der wichtigsten numerischen/ordinalen Merkmale (z.B. 'GrLivArea' * 'OverallQual') basierend auf der Merkmalswichtigkeitsrangliste von AutoGluon. Ich testete Kombinationen der Top 3, 5 und 10 Merkmale, aber dieser Ansatz verbesserte die Testdatensatz-Bewertung nicht (Tabelle 2). Ein Vorbehalt ist, dass die Merkmalswichtigkeitsrangliste von AutoGluon für die Verwendung auf dem Testdatensatz gedacht ist; da ich jedoch den SalePrice für den Testdatensatz nicht habe, wendete ich sie stattdessen auf den Trainingsdatensatz an.

## [Maschinelles Lernen](https://github.com/alexhubbe/house_prices_kaggle_competition/blob/master/notebooks/03_ah_MODEL.ipynb)  
- Ich verwendete **AutoGluon**, um die maschinellen Lernanalysen durchzuführen.  
- Die Analysen wurden mit den Voreinstellungen `presets='good_quality'` und `time_limit=30 min` durchgeführt (die "good_quality"-Voreinstellung ist in der Regel in weniger als 30 Minuten abgeschlossen). Die Voreinstellungen `medium_quality`, `high_quality` und `best_quality` verbesserten die Bewertung des Testdatensatzes nicht (siehe Tabelle 2).  
- Weder das **AutoGluon Pseudolabel Model Refit** noch die **Modelldestillation** verbesserten die Bewertung des Testdatensatzes (Tabelle 2).  

### Tabelle 2: Vorgeschlagene Behandlungen, die die Bewertungen nicht verbesserten

| **Behandlungen**                                                                 |
|--------------------------------------------------------------------------------|
| Entfernung von 'Street', 'Utilities', 'RoofMatl', 'Condition2', 'PoolQC'       |
| Längen- und Breitengrad                                                       |
| Kombination unterrepräsentierter Klassen                                       |
| Gruppierung von Nachbarschaften in größere Gebiete                             |
| Definition der Jahreszeiten                                                   |
| Delta-Merkmale für Bau-, Umbau- und Verkaufsjahre                             |
| Gesamtfläche, fertiggestellte Fläche, hochwertige Fläche und Gesamtzahl der Badezimmer |
| Ordinale Kodierung                                                            |
| Entfernung extremer Ausreißer                                                  |
| Case-Shiller U.S. National Home Price Index                                    |
| All-Transactions House Price Index für Ames, IA                                |
| Interaktionsmerkmale unter den Top-Merkmalen                                   |
| AutoGluon-Voreinstellungen: `medium_quality`, `high_quality`, `best_quality`   |
| AutoGluon Pseudolabel Model Refit und Modelldestillation                       |

---

## Tools und Technologien:
- **Bibliotheken**: AutoGluon, Featuretools, Matplotlib, Numpy, Pandas, Seaborn, Scikit-Learn, 

---

## Projektorganisation
```
├── .gitignore                              <- Dateien und Verzeichnisse, die von Git ignoriert werden sollen
|
├── LICENSE                                 <- Lizenztyp 
|
├── README.md                               <- Haupt-README, die das Projekt erklärt
|
├── data                                    <- Projektdateien
|   ├── ames_dataset.csv                    <- Originaldatensatz von R's tidymodels
|   ├── ATNHPIUS11180Q.csv                  <- Daten für den Hauspreisindex von Ames
|   ├── clean_data.csv                      <- Für die maschinelle Lernanalyse vorbereiteter Datensatz
|   ├── CSUSHPINSA.csv                      <- Daten für den Case-Shiller-Index der USA
|   ├── original_plus_lat_lon.csv           <- Kaggle-Datensatz mit hinzugefügten Längen- und Breitengraden von tidymodels
|   ├── test.csv                            <- Originaler Testdatensatz von Kaggle
|   ├── train.csv                           <- Originaler Trainingsdatensatz von Kaggle
|
├── environments                            <- Anforderungsdateien zur Reproduktion der Analyseumgebungen
|   ├── autogluon_environment.yml           <- Umgebung für die Ausführung von '03_ah_MODEL.ipynb' 
|   ├── eda_environment.yml                 <- Umgebung für die Ausführung von '01_ah_merging_datas.ipynb', '02_ah_EDA.ipynb' und 'ah_appendix_2.ipynb' 
|   ├── maps_environment.yml                <- Umgebung für die Ausführung von 'ah_appendix_1.ipynb'
|
├── models                                  <- Trainierte und serialisierte Modelle, Modellvorhersagen oder Modellzusammenfassungen
|
├── notebooks                               <- Jupyter-Notebooks
|   ├── 01_ah_merging_datas.ipynb           <- Hinzufügen von Längen- und Breitengradinformationen zum Kaggle-Datensatz
|   ├── 02_ah_EDA.ipynb                     <- Explorative Datenanalyse
|   ├── 03_ah_MODEL.ipynb                   <- Ansatz des maschinellen Lernens
|   ├── ah_appendix_1.ipynb                 <- Erstellung von Karten
|   ├── ah_appendix_2.ipynb                 <- Erforschung einer alternativen Datensatztransformation
|   └── src                                 <- In diesem Projekt verwendeter Quellcode
|       ├── __init__.py                     <- Macht dies zu einem Python-Modul
|       ├── auxiliaries.py                  <- Skripte zur Berechnung der nächsten Häuser und medianen Verkaufspreise
|       ├── config.py                       <- Grundlegende Projektkonfiguration
|       ├── eda.py                          <- Skripte für explorative Datenanalyse und Visualisierungen
|
├── references                              <- Datenwörterbücher, Handbücher und andere erklärende Materialien
|   ├── data_description.txt                <- Beschreibung des Datensatzes, wie auf Kaggle präsentiert
|
├── reports                                 <- Generierte Analysen in HTML, PDF, LaTeX usw. und Ergebnisse
|   ├── best_kaggle_prediction.csv          <- Beste Vorhersage aus dem Kaggle-Wettbewerb
|   ├── summary_report.md                   <- Zusammenfassenden Bericht
│   └── images                              <- Im Projekt verwendete Bilder
|      ├── sale_prices_train_test_plot.png  <- Karte mit Häusern mit und ohne Preise 
|      ├── sale_prices.png                  <- Karte mit Hauspreisen
```

## Mitwirkung
Alle Beiträge sind willkommen!

### Probleme
Reichen Sie Probleme ein für:
- Empfehlungen oder Verbesserungen
- Zusätzliche Analysen oder Modelle
- Funktionserweiterungen
- Fehlermeldungen

### Pull Requests
- Öffnen Sie ein Problem, bevor Sie mit der Arbeit beginnen.
- Forken Sie das Repository und klonen Sie es.
- Erstellen Sie einen Branch und committen Sie Ihre Änderungen.
- Pushen Sie Ihre Änderungen und öffnen Sie einen Pull Request zur Überprüfung.
