import csv

def getTraingingData():
    with open('500_Person_Gender_Height_Weight_Index.csv', 'r') as file:
        reader = csv.reader(file)
        array_2d = []
        for row in reader:
            array_2d.append(row[:3])
        print(array_2d)

    return array_2d









