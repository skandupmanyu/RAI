"""
Columns constants
"""
# original columns
INCTOT = 'INCTOT' #
AGE = 'AGE'
YEAR = 'YEAR'
MET2013 ='MET2013'  #2013 US Census metropolitan area
RACE = 'RACE'
HISPAN = 'HISPAN'
RENT = 'RENT'
MORTGAGE = 'MORTGAGE'
SERIAL  = 'SERIAL'   #track individuals over time
CBSERIAL = 'CBSERIAL' #not used
NUMPREC = 'NUMPREC' #not used
SUBSAMP = 'SUBSAMP' #not used
HHWT = 'HHWT' #not used
HHTYPE = 'HHTYPE'  #similar to married status/with dependents
EDUC = 'EDUC'  #educated
MARST = 'MARST'  #Marital status

COLUMNS = ['INCTOT', 'AGE', 'YEAR', 'MET2013', 'RACE', 'HISPAN', 'RENT', 'MORTGAGE',
           'SERIAL', 'CBSERIAL', 'NUMPREC', 'SUBSAMP', 'HHWT', 'HHTYPE', 'EDUC', 'MARST']

NUMERICAL_VAR = ['RENT', 'MORTGAGE', 'AGE', 'INCTOT']

CATEGORICAL_VAR = ['EDUC', 'MARST', 'HHTYPE']