from django.shortcuts import render
from django.http import HttpResponse
from .ml_model import prediction_model


def home(request):
    return render(request, 'C:/Users/diego.mayer/Documents/'
                  'developer/Projects/titanic/app/templates/'
                  'app/index.html')

def result(request):
    PassengerId = 1310
    Pclass = int(request.GET["age"])
    Sex = str(request.GET["sex"])
    Age = int(request.GET["age"])
    SibSp = int(request.GET["sibsp"])
    Parch = int(request.GET["parch"])
    Ticket = str(request.GET["ticket"])
    Fare = float(request.GET["fare"])
    Cabin = str(request.GET["cabin"])
    Embarked = str(request.GET["embarked"])
    Title = str(request.GET["title"])
    First_name = str(request.GET["first_name"])
    Last_name = str(request.GET["last_name"])
    Name = Last_name+', '+Title+' '+First_name   

    
        # Corrigir no  html
    X = [[PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]]
    prediction = prediction_model(X)
    
    return render(request, 'C:/Users/diego.mayer/Documents/'
                  'developer/Projects/titanic/app/templates/'
                  'app/result.html', {'prediction' : prediction,
                                     'name': Name})
