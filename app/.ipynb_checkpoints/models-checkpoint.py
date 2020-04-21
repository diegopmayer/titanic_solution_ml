from django.db import models
from django.conf import settings
from django.utils import timezone
import pandas as pd

class Passenger(models.Model):
    PassengerId = 1311
    first_name = models.CharField(max_length=200) # use to show the result
    last_name = models.CharField(max_length=200) # get surname
    list_title = [
            ('Mr','Mr.'),
            ('Master','Master'),
            ('Mrs','Mrs.'),
            ('Miss','Miss.')
            ]
    title = models.CharField(max_length=10,
                             choices=list_title) # get suspention list
    Name = str(last_name)+' '+str(title)+' '+str(first_name)
    list_Pclass = [
        (1, 1),
        (2, 2),
        (3, 3)
        ]
    Pclass = models.IntegerField(choices=list_Pclass) # selected in a list
    list_Sex = [
        ('Female', 'female'),
        ('Male', 'male')
    ]
    Sex = models.CharField(max_length=6, choices=list_Sex)
    Age = models.IntegerField()
    SibSp = models.IntegerField() # Sum of Simbling and Spouses
    Parch = models.IntegerField() # Sum of Simbling and Spouses
    Ticket = models.CharField(max_length=10) #name or number of ticket
    Fare = models.DecimalField(max_digits=6, decimal_places=2) # must upper than 1
    Cabin = models.CharField(max_length=3) # must start with a Letter and 2 others numbers
    list_Embark = [
        ('Cherbourg','C'),
        ('Queenstown','Q'),
        ('Southampton','S')
        ]
    Embarked = models.CharField(max_length=11, choices=list_Embark) # City from embarked pessenger
    
    # Data like test.csv, you will export in the same structure
    # PassengerId	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
    # Get the value and export to csv test.csv
    
    test_csv = pd.DataFrame([PassengerId, Pclass, Name, 
                             Sex, Age, SibSp, Parch, Ticket, 
                             Fare, Cabin, Embarked])
    test_csv.to_csv('test.csv', index=False)
    
    def __str__(self):
        return self.first_name

