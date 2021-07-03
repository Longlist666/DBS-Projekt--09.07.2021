# DBS-Projekt--09.07.2021

Programm erstellt aus den 4 Datensätzen eine Datenbank mit Hilfe von SQLight. 
Es kann mit der Anwendung ein 2D oder 3D Chart erzeugt werde. 

Für den 2D Chart muss der Befehl -2d und zwei Parameter eingegeben werden (year, co2_emission, population_total, gdp oder life_expectancy).
Für den 3D Chart muss der Befehl -3d und zwei Parameter eingegeben werden (co2_emission, population_total, gdp oder life_expectancy). 

Bei dem 3D Chart, sind die Jahre schon als Parameter standardmäßig festgelegt. 

Beispiel: 

db_projekt.py -2d year life_expectancy

db_projekt.py -3d gdp population_total

Link zu Datensatz: 

https://www.kaggle.com/brendan45774/countries-life-expectancy
