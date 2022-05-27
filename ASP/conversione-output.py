from dataclasses import dataclass
import os, re, json

import sys
from colorama import init
from sqlalchemy import null
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint
from pyfiglet import figlet_format

import argparse
parser = argparse.ArgumentParser(description = 'Process display arguments')
parser.add_argument('-o', '--optimal', nargs = '?', const = True, default = False)
args = parser.parse_args()

problem_instance = "SCHEDA-ALLENAMENTO-instance.lp"
problem_encoding = "SCHEDA-ALLENAMENTO-encoding.lp"

if args.optimal:
	# Runna il programma SENZA time-limit e cerca finché non trova l'ottimo
	print("Looking for optimal solution...")
	stream = os.popen("clingo "+ problem_instance + " " + problem_encoding + " --quiet=1")
else:
	print("Looking for a solution in 5 minutes...")
	stream = os.popen("clingo "+ problem_instance + " " + problem_encoding + " --time-limit=300 --quiet=1")

print("Running and solving... (it might take some time)")
output_string = stream.read()
output_string_T = output_string


# Print isOptimalSolution
index = output_string.find("Optimum    : ")
if output_string[index+13] == 'y':
	isOptimalSolution = True
else:
	isOptimalSolution = False

# print(output_string)
print("Plan found!")
pattern = re.compile(r'esegui\((.*?)\)')
x = pattern.findall(output_string)

# print(x)

# Get warmup time
warmup_time_pattern = re.compile(r'warmup_time\((.*?)\)')
y = warmup_time_pattern.findall(output_string_T)
warmup_time = int(y[0])

m, s = divmod(warmup_time, 60)
h, m = divmod(m, 60)


@dataclass
class Esercizio:
	nome_esercizio: any
	fattore_utilita_superiore: int
	fattore_utilita_core: int
	fattore_utilita_inferiore: int
	categoria_livello: any
	livello_ramo: int
	n_rep: int
	n_serie: int
	durata_rep: int
	durata_isometria: int
	pesi: int
	strumento: any
	tempo_riposo: int
	day: int

	def to_dict(self):
		return {
			"nome_esercizio": self.nome_esercizio,
			"fattore_utilita_superiore": self.fattore_utilita_superiore,
			"fattore_utilita_core": self.fattore_utilita_core,
			"fattore_utilita_inferiore": self.fattore_utilita_inferiore,
			"categoria_livello": self.categoria_livello,
			"livello_ramo": self.livello_ramo,
			"n_rep": self.n_rep,
			"n_serie": self.n_serie,
			"durata_rep": self.durata_rep,
			"durata_isometria": self.durata_isometria,
			"pesi": self.pesi,
			"strumento": self.strumento,
			"tempo_riposo": self.tempo_riposo,
			"day": self.day,
		}


lista_esercizi = []
for i in x:
	el = i.split(",")
	lista_esercizi.append(Esercizio(el[0][1:-1],el[1],el[2],el[3],el[4],el[5],el[6],el[7],el[8],el[9],el[10],el[11][1:-1],el[12],el[13]))

# print(lista_esercizi)



# conversione a json
results = [obj.to_dict() for obj in lista_esercizi]
# sort by days
sorted_results = sorted(results, key=lambda d: d['day'])
# print(results)
with open('workout-plan'+'.json', 'w') as outfile:
	print("Exporting to JSON...\n")
	json.dump(sorted_results, outfile)



# lettura json e display all'utente

with open('workout-plan.json') as f:
	print("Reading JSON...\n\n\n")
	data = json.load(f)

# print(type(data))

# print(data)

# for i in data:
# 	print(str(i) + '\n')

day = 0
days = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

print("Here's your weekly...\n==============================================================")

cprint(figlet_format('Workout plan', font='ogre'),
       'yellow', attrs=['bold'])

print("{:>51}".format("") + "... optimal!" if isOptimalSolution == True else "{:>48}".format("") + "... suboptimal!")

for i in data:
	new_day = int(i['day'])
	if day != new_day:
		print('\n'+str(days[new_day-1]))
		day = new_day
		print("---------------")
		print("{: >35}".format("Warm-up") + "{: >18}".format(m) + "m" + "{: >2}".format(s) + "s") # si potrebbe aggiungere anche l'atom di warmup time


	if i['durata_isometria'] == '0':
		durata_iso = "-"
	else:
		durata_iso = i['durata_isometria']+'s'
	print("{: >35}".format(i['nome_esercizio']) + "{: >3}".format(i['n_serie']) + "x" + "{:<2}".format(i['n_rep']) + "{:>5}".format(durata_iso) + "{: >10}".format(i['tempo_riposo'])+"s" + "{:>30}".format(i['strumento']))

# in qualche modo si può prendere l'output che ha prodotto anche se non ha finito trovando l'ottimo?
# così da limitare il tempo con --time-limit=60 e prendere l'ultimo answer set prodotto.