% GENERA TUTTE LE COMBINAZIONI DI ESERCIZI PER I GIORNI DATI

d(1, 60). % giorno disponibile espresso in numero, minuti dell'allenamento disponibili.
d(3, 30).
% disponibilita(5, 90).

esercizio(pushup).
esercizio(squat).
% esercizio(dip).

% 0{es(NOME_ESERCIZIO, DAY) : d(DAY, _)} NUM_DAYS:- NUM_DAYS = #count{DAYS : d(DAYS,_)}, esercizio(NOME_ESERCIZIO).
% 1{es(NOME_ESERCIZIO, DAY) : d(DAY, _)}1 :- esercizio(NOME_ESERCIZIO).
{es(NOME_ESERCIZIO, DAY) : esercizio(NOME_ESERCIZIO), d(DAY, _)}.

#show es/2.