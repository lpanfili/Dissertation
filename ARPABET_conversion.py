# 1 = voiced
voicing = {
	"AA":1,
	"AE":1,
	"AH":1,
	"AO":1,
	"AW":1,
	"AY":1,
	"B":1,
	"CH":0,
	"D":1,
	"DH":1,
	"EH":1,
	"ER":1,
	"EY":1,
	"F":0,
	"G":1,
	"HH":0,
	"IH":1,
	"IY":1,
	"JH":1,
	"K":0,
	"L":1,
	"M":1,
	"N":1,
	"NG":1,
	"OW":1,
	"OY":1,
	"P":0,
	"R":1,
	"S":0,
	"SH":0,
	"T":0,
	"TH":0,
	"UH":1,
	"UW":1,
	"V":1,
	"W":1,
	"Y":1,
	"Z":1,
	"ZH":1
}

# 1 = obstruent
manner = {
	"AA":0,
	"AE":0,
	"AH":0,
	"AO":0,
	"AW":0,
	"AY":0,
	"B":1,
	"CH":1,
	"D":1,
	"DH":1,
	"EH":0,
	"ER":0,
	"EY":0,
	"F":1,
	"G":1,
	"HH":1,
	"IH":0,
	"IY":0,
	"JH":1,
	"K":1,
	"L":0,
	"M":0,
	"N":0,
	"NG":0,
	"OW":0,
	"OY":0,
	"P":1,
	"R":0,
	"S":1,
	"SH":1,
	"T":1,
	"TH":1,
	"UH":0,
	"UW":0,
	"V":1,
	"W":0,
	"Y":0,
	"Z":1,
	"ZH":1
}

def isVoiced(phone):
	phone = removeStress(phone)
	return voicing[phone]

def isObstruent(phone):
	phone = removeStress(phone)
	return manner[phone]

def removeStress(phone):
	if ord(phone[-1]) >= 48 and ord(phone[-1]) <= 50:
		phone = phone[:-1]
	return phone