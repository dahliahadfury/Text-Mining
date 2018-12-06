from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import math
import csv
import numpy as np
import string
from collections import OrderedDict
exclude = set(string.punctuation)
import re

#author : yurdha fadhila
 
#stopword
file_sw=open('tala.txt','r')
stopword= file_sw.read()
array_sw = stopword.split()

#stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#prepocessing
def readMyFile(filecsv):
    data=[]
    with open(filecsv, mode='r', encoding='utf8') as csvFile:
        csvReader = csv.DictReader(csvFile, delimiter=';')
        for row in csvReader:
            data.append(row)
    return data

def getTeks(data):
    dataTeks=[]
    for row in data:
        dataTeks.append(row["Teks"])
    return dataTeks

def lexicalAnalysis(data):
    sentence=[]
    words=[]
    token=[]

    #case folding
    for row in data:
        sentence.append(row.lower())
    
    #get word per sentence
    for i in range(len(sentence)):
        sentence[i]= sentence[i].split()

    #get all words
    for row in sentence:
        for row2 in row:
            temp = row2.split().pop()
            words.append(temp)

    #deleting link
    for n, row in enumerate(words):
        if re.match(r"\w+(?:\.me)(?:.*)", str(row)) or re.match(r"\w+(?:\.com)", str(row)) or re.match(r"(?:(https?|s?ftp):\/\/)(?:.*)?", str(row)) or re.match(r"(?:www\.)(?:.*)?", str(row)) :
            del words[n]
    
    #deleting punctuation
    for i in range(len(words)):
        temp = re.sub(r'[^\w\s]',' ',str(words[i]))
        words[i] = temp

    #deleting number
    for i in range(len(words)):
        temp =re.sub(r'\d',' ',str(words[i]))
        words[i] = temp
    
    #clean the space
    for row in words:
        temp = row.split()
        token += temp
    
    for i in range(len(token)):
        for n, row in enumerate(token):
            if len(row) < 2 or row == "xxx" or row == "gb" or row == "ff" or row == "rp" or row == "bb" or row == "dll" or row == "ii" or row == "rb" or row == "mb" :
                del token[n]
    
    return token

def stopwordRemoval(data):
    #filtered by sastrawi
    for i in range(len(data)):
        for n, row in enumerate(data):
            if row in array_sw:
                del data[n]
    return data

def stemming(data):
    sentence = ' '.join(data)
    output   = stemmer.stem(sentence)
    hasil_stem = output.split()
    array_stem = []
    for row in hasil_stem:
        array_stem.append(row)
    return array_stem

def getTerm(data):
    term=[]
    for row in data:
        if row not in term:
            term.append(row)
    return term

def data12():
    #DATA LATIH DEBANYAK 12 SMS
    print("DATA LATIH DEBANYAK 12 SMS")
    datalatih12 = readMyFile('datalatih12.csv')
    teks = getTeks(datalatih12)
    #print(file)
    token=lexicalAnalysis(teks)
    # print(token)
    token=stopwordRemoval(token)
    # print(token)
    # print("hasil stemming")
    token= stemming(token)
    # print(token)
    print("term")
    term = getTerm(token)
    print(term)

def data18():
    #DATA LATIH DEBANYAK 18 SMS
    print("DATA LATIH DEBANYAK 18 SMS")
    datalatih18 = readMyFile('datalatih18.csv')
    teks = getTeks(datalatih18)
    #print(file)
    token=lexicalAnalysis(teks)
    # print(token)
    token=stopwordRemoval(token)
    # print(token)
    # print("hasil stemming")
    token= stemming(token)
    # print(token)
    print("term")
    term = getTerm(token)
    print(term)

def data24():
    #DATA LATIH DEBANYAK 24 SMS
    print("DATA LATIH DEBANYAK 24 SMS")
    datalatih24 = readMyFile('datalatih24.csv')
    teks = getTeks(datalatih24)
    #print(file)
    token=lexicalAnalysis(teks)
    # print(token)
    token=stopwordRemoval(token)
    # print(token)
    # print("hasil stemming")
    token= stemming(token)
    # print(token)
    print("term")
    term = getTerm(token)
    print(term)

#term weighting
def getTermInDoc(dataTeks):
    termInDoc=[]
    temp=[]
    temp4=' '
    for i in range(len(dataTeks)):
        temp5 = []
        temp5.append(dataTeks[i]["Teks"])
        temp5.append(temp4)
        temp.append(temp5)

    temp6=[]
    for row in temp:
        temp3 = stemming(stopwordRemoval(lexicalAnalysis(row)))
        temp6.append(temp3)
    return temp6

def rawWeight(dataTeks, term):
    jum=[]
    for k in range(0,len(dataTeks)):
        jum.append([])
        for l in term:
            if l not in dataTeks[k]:
                jum[k].append(0)
            elif l in dataTeks[k]:
                x = dataTeks[k].count(l)
                jum[k].append(x)    
    return jum

#Likelihood atau Conditional probability 
def getKelasRawData(data, raw):
    result=[]
    for i in range(len(data)):
        temp = OrderedDict()
        temp["raw"] = raw[i]
        temp["kelas"] = data[i]["label"]
        result.append(temp.copy())
    return result

def getTotalTermInKelas(data):
    result = {}
    for row in data:
        if row["kelas"] in result:
            result[row["kelas"]] += sum(row["raw"])
        else:
            result[row["kelas"]] = sum(row["raw"])
    return result

def getRawPerKelas(data):
    result = {}
    for row in data:
        if row["kelas"] not in result:
            result[row["kelas"]] = []
        result[row["kelas"]].append(row["raw"])
    return  result

#fase training
def Likelihood(data, totalTermPerKelas ,totalTerm):
    totalPerTermPerKelas={}
    hasilLikelihood={}
    for row in data:
        if row in totalPerTermPerKelas:
            totalPerTermPerKelas[row] = [sum(x) for x in zip(*data[row])]
        else:
            totalPerTermPerKelas[row] = [sum(x) for x in zip(*data[row])]
    
    for row in totalPerTermPerKelas: #3 index
        if row in hasilLikelihood:
            temp=[]
            for row2 in totalPerTermPerKelas[row]:
                temp2 = (row2+1)/(totalTermPerKelas[row] + totalTerm)
                temp.append(temp2)
            hasilLikelihood[row] = temp
        else:
            temp=[]
            for row2 in totalPerTermPerKelas[row]:
                temp2 = (row2+1)/(totalTermPerKelas[row] + totalTerm)
                temp.append(temp2)
            hasilLikelihood[row] = temp
    return hasilLikelihood

#fase testing
def prior(dataTraining):
    result={}
    for row in dataTraining:
        if row["label"] in result:
            result[row["label"]] +=1
        else:
            result[row["label"]] =1

    for row in result:
        result[row] = result[row] / len(dataTraining) 

    return result        

def findMatchTerm(termInDocTesting, term):
    indexTerm=[]
    for i in range(len(termInDocTesting)):
        # print(termInDocTesting[i])
        indexTerm.append([])
        for row2 in termInDocTesting[i]:
            if row2 in term:
                indexTerm[i].append(term.index(row2))
            else:
                indexTerm[i].append("null")
                    
    return indexTerm

def posterior(indexTerm, hslLikelihood, hslprior):
    result = {}
    indexTerms =[]

    for row in indexTerm:
        if row not in indexTerms and row is not "null" :
            indexTerms.append(row)
    
    print(indexTerms)

    for row in hslprior:
        if row in result:
            for index in indexTerms:
                temp = 1
                temp *= hslLikelihood[row][int(float(index))]
            result[row] = temp
        else:
            for index in indexTerms:
                temp = 1
                temp *= hslLikelihood[row][int(float(index))]
            result[row] = temp
    
    print(result)
    
    temp3 = []
    for row in result:
        temp3.append(result[row])

    temp4 = max(temp3)
    print(temp4)

    label = ''

    for x, row in result.items():
        if row == temp4:
            label = x


    return label




print("DATA LATIH DEBANYAK 24 SMS")
datalatih24 = readMyFile('datalatih12.csv')
teks = getTeks(datalatih24)
token=lexicalAnalysis(teks)
# print(token)
token=stopwordRemoval(token)
# print(token)
print("hasil stemming")
token= stemming(token)
# print(token)
print("term")
term = getTerm(token)
print(term)
totalTerm = len(term)
# print(len(term))
#BARU
termInDoc = getTermInDoc(datalatih24)
print("term in doc")
print(termInDoc)
print("raw")
raw = rawWeight(termInDoc,term)
print(raw)
print("raw dengan kelasnya")
raw = getKelasRawData(datalatih24, raw)
print(raw)
totalTermPerKelas = getTotalTermInKelas(raw)
print("total term per kelas")
print(totalTermPerKelas)
print("Raw dengan kelas yang sama")
rawPerKelas = getRawPerKelas(raw)
print(rawPerKelas)
print("likelihood")
hasilLikelihood = Likelihood(rawPerKelas, totalTermPerKelas ,totalTerm)
print(hasilLikelihood)

#DATA UJI SEBANYAK 6 SMS
print("DATA UJI SEBANYAK 6 SMS")
datauji = readMyFile('dataset2.csv')
termInDocTesting = getTermInDoc(datauji)
print(termInDocTesting)
print("PRIOR")
hasilPrior = prior(datalatih24)
print(hasilPrior)
print("FIND MATCH TERM")
indexMacthTerm = findMatchTerm(termInDocTesting, term)
print(indexMacthTerm)
hasilPosterior = []
for i in range(len(indexMacthTerm)):
    temp = OrderedDict()
    temp["nomor sms"] = i
    temp["label"] = posterior(indexMacthTerm[i], hasilLikelihood, hasilPrior)
    hasilPosterior.append(temp)

print("hasil klasifikasi")
print(hasilPosterior)
print(list(hasilPosterior[0].keys()))
for data in hasilPosterior:
	print(list(data.values()))
