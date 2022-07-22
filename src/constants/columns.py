"""
Columns constants
"""
# original columns
INCTOT = 'inctot' #
AGE = 'age'
YEAR = 'year'
MET2013 ='met2013'  #2013 US Census metropolitan area
RACE = 'race'
HISPAN = 'hispan'
RENT = 'rent'
MORTGAGE = 'mortgage'
# SERIAL  = 'serial'   #track individuals over time
# CBSERIAL = 'cbserial' #not used
# NUMPREC = 'numprec' #not used
# SUBSAMP = 'subsamp' #not used
# HHWT = 'hhwt' #not used
HHTYPE = 'hhtype'  #similar to married status/with dependents
EDUC = 'educ'  #educated
MARST = 'marst'  #Marital status

COLUMNS = ['HISPAN','INCTOT', 'AGE', 'YEAR', 'MET2013', 'RACE', 'RENT', 'MORTGAGE', 'HHTYPE', 'EDUC', 'MARST']
NUMERICAL_VAR = ['rent', 'mortgage', 'age', 'inctot']

CATEGORICAL_VAR = ['educ', 'marst', 'hhtype']

