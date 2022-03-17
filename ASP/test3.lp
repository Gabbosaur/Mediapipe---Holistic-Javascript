% input utente
livello_attuale(e, 4). % con 1 livello di profondità, quindi proporremo esercizi del livello attuale e del livello precedente.

disponibilita(1, 60). % giorno disponibile espresso in numero, minuti dell'allenamento disponibili.
disponibilita(3, 30).
% disponibilita(5, 90).

{esercizio(pushup,1, b, 1)}.
% {esercizio(pushup,2, b, 1)}.
% {esercizio(pushup,3, b, 1)}.
{esercizio(squat,4, b, 2)}.
% {esercizio(pushup,5, b, 2)}.
% {esercizio(pushup,6, b, 2)}.
% {esercizio(pushup,7, b, 2)}.
% {esercizio(pushup,8, b, 3)}.
% {esercizio(pushup,9, b, 3)}.

% {esercizio(tuck_frontlever, 0, s, 4)}.
% {esercizio(skinthecat,1, s, 4)}.
% % {esercizio(pushup,10, e, 4)}.
% {esercizio(pushup,12, e, 4)}.
% % {esercizio(pushup,5, w, 4)}.
% {esercizio(pushup,6, w, 4)}.

1 {esercizi_settimana(NOME_ESERCIZIO, N_REP, CATEGORIA, LIVELLO_RAMO, DAY) : disponibilita(DAY, _)} NUM_DAYS :- NUM_DAYS = #count{DAYS : disponibilita(DAYS,_)}, esercizio(NOME_ESERCIZIO, N_REP, CATEGORIA, LIVELLO_RAMO).


% prendiamo solo gli answers dove abbiamo tutti i giorni disponibili
:- NUM_GIORNI_IN_ANSWER = #count{DAY : esercizi_settimana(_,_,_,_,DAY)}, NUM_DAYS = #count{DAYS : disponibilita(DAYS,_)}, NUM_GIORNI_IN_ANSWER != NUM_DAYS.

#show esercizi_settimana/5.


%% TODOS

% dopo aver fatto la lista degli esercizi e database

% conto dei minuti per ogni giorno che sono occupati da tutti gli esercizi che abbiamo scelto giorno per giorno.

% calcolare l'intensità se la sessione giornaliera è pesante.

% vedere se abbiamo dei giorni vicini con next, se sì, verificare che non abbiano un carico alto.