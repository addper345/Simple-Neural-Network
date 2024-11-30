import csv
import numpy as np

def getTrainingData(BMI):
    with open("500_Person_Gender_Height_Weight_Index.csv", 'r') as file:
        reader = csv.reader(file)
        array_2d = []
        for row in reader:
            if(int(row[3]) == BMI):
                array_2d.append(np.array(row[:3], dtype=float))
    return array_2d[1:]

    




