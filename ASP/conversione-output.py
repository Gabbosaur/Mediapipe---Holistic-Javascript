from dataclasses import dataclass
import os, re


problem_instance = "SCHEDA-ALLENAMENTO-instance.lp"
problem_encoding = "SCHEDA-ALLENAMENTO-encoding.lp"
stream = os.popen("clingo "+ problem_instance + " " + problem_encoding + " --quiet=1")

print("Running and solving...\n")
output_string = stream.read()
# print(output_string)

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

lista_esercizi = []
for i in x:
	el = i.split(",")
	lista_esercizi.append(Esercizio(el[0][1:-1],el[1],el[2],el[3],el[4],el[5],el[6],el[7],el[8],el[9],el[10],el[11][1:-1],el[12],el[13]))

print(lista_esercizi)

# conversione a json