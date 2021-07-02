import sqlite3, csv, os, argparse, glob
import pandas as pd
import numpy as np
from argparse import RawTextHelpFormatter
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# Checks, if DBS exists and saves the path 
# Path is later used to ensure vital data is't delated if the DB is already construct
path = glob.glob("DBS_Projekt_Datenbank.db")


parser = argparse.ArgumentParser(description="-----Database-Projekt-----", formatter_class=RawTextHelpFormatter)
group = parser.add_mutually_exclusive_group()
group.add_argument("-2d", type=str, 
                    help="Generates a static 2d-plot in respect to the given column-names.\nOrder: x y\nInput-choices: year co2_emission gdp life_expectancy population_total",  
                    nargs=2, choices=["year", "co2_emission", "gdp", "life_expectancy", "population_total"],
                    metavar="COLUMN")
group.add_argument("-3d", type=str, 
                    help="Generates a dynamic 3d-plot in respect to the given years and column-names.\nOrder: x=year y z\nInput-choices: co2_emission gdp life_expectancy population_total", 
                    nargs=2, choices=["co2_emission", "gdp", "life_expectancy", "population_total"], 
                    metavar="COLUMN")

args = vars(parser.parse_args())
if not any(args.values()):
    parser.error("No arguments provided.")
args = parser.parse_args()
 
class StatTestsPlot:
    #Plots a simple 2D graph for two given arrays of the same size
    def plotter_2d(self, x, y, xlabel, ylabel, txt = None, description = None):
        for a in y:
            plt.plot(x, a)
        if leg is not None:
            plt.legend(leg, loc='upper right')
        if txt is not None:
            pass
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    #Plots a connected 2D or 3D scatterplot, depending if a array for the z-axis is present
    def mixedplot(self, x, y, xlabel, ylabel, marks = None, z = None, zlabel = None):
        fig = plt.figure()
        m = len(marks)//6                                                               #Partitions x/y/z array in their respective subsets
        colorB = ['b', 'r', 'c', 'm', 'y', 'g']
        hashmap = {'b':"Blue", 'r':"Red", 'c':"Cyan", 'm':"Magenta", 'y':"Yellow", 'g':"Green"}
        colors = ['b']*m+['r']*m+['c']*m+['m']*m+['y']*m+['g']*m
        if (z is not None):                                                             #3D connected scatterplot
            ax = fig.add_subplot(projection='3d')
            for i, color in enumerate(colorB):
                print("%s : %s" % (hashmap[color], marks[i*m][0]))                      #Just printing the countries in relation to their colours due to missing legend on 3d-plot
                ax.scatter(y[i],x[i],z[i], c=colorB[i])                             
                ax.plot(y[i].flatten(),x[i].flatten(),z[i].flatten(), c=colorB[i])
            ax.set_zlabel(zlabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            for i, color in enumerate(colorB):                                          #2D connected scatterplot
                plt.scatter(y[i], x[i], c=color, label=marks[i*m][0])
                plt.plot(y[i], x[i], c=color)
                plt.legend(loc='upper right')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
        plt.show()
        
conn = sqlite3.connect("DBS_Projekt_Datenbank.db")
cur = conn.cursor()


# Creates DB with SQLight
cur.execute('''CREATE TABLE IF NOT EXISTS co2_emission
               (country_name        text,
                Code                text, 
                year                real, 
                co2_emission        real,
                PRIMARY KEY(country_name, year))''')

cur.execute('''CREATE TABLE IF NOT EXISTS life_expectancy
               (country_name        text,
                year                real, 
                life_expectancy     real,
                PRIMARY KEY(country_name, year))''')

cur.execute('''CREATE TABLE IF NOT EXISTS gdp
               (country_name        text,
                year                real, 
                gdp                 real, 
                PRIMARY KEY(country_name, year))''')

cur.execute('''CREATE TABLE IF NOT EXISTS population_total
               (country_name        text,
                year                real, 
                population_total    real,
                PRIMARY KEY(country_name, year))''')

#Just saving an extremely long SQL-Substatement to eliminate certain data in our database (countries not examined)
del_specification = " WHERE (country_name  != 'Brazil') AND (country_name  != 'China') AND (country_name  != 'Germany') AND (country_name  != 'India') AND (country_name  != 'Japan') AND (country_name  != 'United States')"

# Add data in csv files to SQL tables 
if not path:
    array, array_save = np.array([]), np.array(["Country Name", "year", "gdp"])

    with open('co2_emission.csv', 'r') as file:
        for row in file:
            cur.execute("INSERT INTO co2_emission Values (?,?,?,?)", row.split(","))

    with open('life_expectancy.csv', 'r') as file:
        for row in file:
            cur.execute("INSERT INTO life_expectancy Values (?,?,?)", row.split(","))
        
    with open('population_total.csv', 'r') as file:
        for row in file:
            if len(row.split(",")) == 3:
                cur.execute("INSERT INTO population_total Values (?,?,?)", row.split(","))


    array = np.array([])
    array_save = np.array(["Country Name", "year", "gdp"])

    # Data is in another format(compared with the other files), need to flip column with rows -> so we can add the data in the same fromat to the SQL table as the other tables 
    with open('gdp.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header_row = next(reader)
        for index, column_header in enumerate(reader):
            country_name = column_header[0]

            if (country_name == "Brazil" or country_name == "China" or country_name == "Germany" or country_name == "India" or country_name == "Japan" or country_name == "United States"):
                for header, gdp in zip (header_row[4:], column_header[4:]):
                    array_save = np.vstack((array_save,np.array([country_name, header, gdp])))
    df_gdp = pd.DataFrame(array_save)
    for row in array_save[0:]:
        cur.execute("INSERT INTO gdp(country_name, Year,gdp) Values (?,?,?)", row)

#Calculates the convex hull of our dataset of "year" and excludes every entrie outside of said intervall.
#It's needed because it is essential that our used dataset is complete, otherwise we can't create a 
#contiguous plot. 

#(Please don't add, move or changefollowing marked code blocks, due to the changed cursor)
#-------------------------------------------------------------------------------

cur.row_factory = lambda cursor, x:x[0]     #Changes returned cursor-tables from tuples to arrays
f, g = lambda x:x[::2], lambda x:x[1::2]    #function to get only even/odd indexed values of a given array

#selects minimum/maximum year of every table, compares them and saves the supremum/infimum in tmp
tmp = cur.execute("SELECT year FROM life_expectancy WHERE rowid = 2 OR rowid = (SELECT max(rowid) FROM life_expectancy)").fetchall()
tmp += cur.execute("SELECT year FROM co2_emission WHERE rowid = 2 OR rowid = (SELECT max(rowid) FROM co2_emission)").fetchall()
tmp += cur.execute("SELECT year FROM gdp WHERE rowid = 2 OR rowid = (SELECT max(rowid)-1 FROM gdp)").fetchall()
cur.execute("SELECT year FROM population_total WHERE rowid = 2 OR rowid = (SELECT max(rowid) FROM population_total)")
tmp = [int(x) for x in tmp+cur.fetchall()]
tmp = (max(f(tmp)), min(g(tmp)))
cur.row_factory = None
#------------------------------------------------------------------------------

#Excludes every entry where the year is not in the calculated convex hull and not examined countries
if not path:
    cur.execute("DELETE FROM co2_emission" + del_specification + " OR year <%s OR year >%s" % (tmp[0],tmp[1]))
    cur.execute("DELETE FROM gdp" + del_specification + " OR year <%s OR year >%s" % (tmp[0],tmp[1]))
    cur.execute("DELETE FROM life_expectancy" + del_specification + " OR year <%s OR year >%s" % (tmp[0],tmp[1]))
    cur.execute("DELETE FROM population_total" + del_specification + " OR year <%s OR year >%s" % (tmp[0],tmp[1]))

                                        
if not path:
    # Reorder (by Country and Year) and update population_total table  
    df_population_total_sort = pd.read_sql_query("SELECT * from population_total", conn)
    df_population_total_sort = df_population_total_sort.sort_values(by=['country_name', 'year'])
    df_population_total_sort.to_sql('population_total', conn, if_exists='replace', index = False)

    # Replace missing values (indiceted with '' in csv file) with 0 -> for the ability to display gdp later with other data   
    df_gdp = pd.read_sql_query("SELECT * from gdp", conn)
    df_gdp.loc[(df_gdp.gdp == '', 'gdp')] = 0
    df_gdp.to_sql('gdp', conn, if_exists='replace', index = False)


data = []

#Selects tables based on given argparse arguments and transforms them to fit in our plotting function
#-------------------------------------------------------------------------------
cur.row_factory = lambda cursor, x:x[0]
if(list(vars(args).items())[1][1] is None):
    a = list(vars(args).values())[0]
    if ("year" not in a):
        for x in a:
            data += [np.array(cur.execute("SELECT %s FROM %s" % (x, x)).fetchall()).astype(float)]
        marks = np.array(cur.execute("SELECT country_name FROM %s" % (a[0])).fetchall())    
        StatTestsPlot().mixedplot(np.split(data[0],6), np.split(data[1],6), a[0], a[1], marks)
    else:
        cur.row_factory = None
        if(a[0]!="year"):
            b = a[0]
            a[0], a[1] = a[1], a[0]
        else:
            b = a[1]
        data = np.array(cur.execute("SELECT %s, year FROM %s" % (b, b)).fetchall()).astype(float)
        marks = np.array(cur.execute("SELECT country_name FROM %s" % (b)).fetchall())
        data = np.hsplit(data,2)
        StatTestsPlot().mixedplot(np.split(data[0],6), np.split(data[1],6), a[0], a[1], marks)
else:
    a = list(vars(args).values())[1] 
    cur.row_factory = None
    data_a = np.array(cur.execute("SELECT %s, year FROM %s" % (a[0], a[0])).fetchall()).astype(float)
    data_b = np.array(cur.execute("SELECT %s FROM %s" % (a[1], a[1])).fetchall()).astype(float)
    marks = np.array(cur.execute("SELECT country_name FROM %s" % (a[0])).fetchall())
    data_a = np.hsplit(data_a,2)
    StatTestsPlot().mixedplot(np.split(data_a[0],6), np.split(data_a[1],6), "years", a[0], marks, np.split(data_b, 6), a[1])
cur.row_factory = None
#-------------------------------------------------------------------------------
conn.commit()
conn.close()