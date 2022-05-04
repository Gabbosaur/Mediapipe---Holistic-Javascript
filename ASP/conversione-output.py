from dataclasses import dataclass
import os, re, json

import sys
from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint
from pyfiglet import figlet_format

problem_instance = "SCHEDA-ALLENAMENTO-instance.lp"
problem_encoding = "SCHEDA-ALLENAMENTO-encoding.lp"
stream = os.popen("clingo "+ problem_instance + " " + problem_encoding + " --quiet=1")

print("Running and solving...")
output_string = stream.read()
# print(output_string)
print("Plan found!")
pattern = re.compile(r'esegui\((.*?)\)')
x = pattern.findall(output_string)

# print(x)

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


for i in data:
	new_day = int(i['day'])
	if day != new_day:
		print('\n'+str(days[new_day-1]))
		day = new_day
		print("---------------")
		print("{: >35}".format("Warm-up")) # si potrebbe aggiungere anche l'atom di warmup time


	print("{: >35}".format(i['nome_esercizio']) + "{: >3}".format(i['n_serie']) + "x" + "{:<2}".format(i['n_rep']) + "{: >7}".format(i['tempo_riposo'])+"s" + "{:>30}".format(i['strumento']))
	
