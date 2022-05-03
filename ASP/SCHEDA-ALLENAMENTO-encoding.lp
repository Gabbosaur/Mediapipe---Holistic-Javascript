% Generazione di answer sets per categoria base
{esegui(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE, CL, LR, REP+PR, SERIE+PS, DR, DI, PESI+PP, STRUMENTO, TR+PTR, DAY) :  esercizio(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE), variante_esercizio(_, NOME_ESERCIZIO, CL, LR, REP, SERIE, DR, DI, PESI, STRUMENTO, TR), input_performance_base(PR, PS, PTR, PP), disponibilita(DAY, _)} :- variante_esercizio(_, _, CL, _, _, _, _, _, _, _, _), CL = b.

% Generazione di answer sets per categoria skills
{esegui(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE, CL, LR, REP+PR, SERIE+PS, DR, DI, PESI+PP, STRUMENTO, TR+PTR, DAY) :  esercizio(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE), variante_esercizio(_, NOME_ESERCIZIO, CL, LR, REP, SERIE, DR, DI, PESI, STRUMENTO, TR), input_performance_skills(PR, PS, PTR, PP), disponibilita(DAY, _)} :- variante_esercizio(_, _, CL, _, _, _, _, _, _, _, _), CL = s.

% Generazione di answer sets per categoria endurance
{esegui(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE, CL, LR, REP+PR, SERIE+PS, DR, DI, PESI+PP, STRUMENTO, TR+PTR, DAY) :  esercizio(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE), variante_esercizio(_, NOME_ESERCIZIO, CL, LR, REP, SERIE, DR, DI, PESI, STRUMENTO, TR), input_performance_endurance(PR, PS, PTR, PP), disponibilita(DAY, _)} :- variante_esercizio(_, _, CL, _, _, _, _, _, _, _, _), CL = e.

% Generazione di answer sets per categoria weighted
{esegui(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE, CL, LR, REP+PR, SERIE+PS, DR, DI, PESI+PP, STRUMENTO, TR+PTR, DAY) :  esercizio(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE), variante_esercizio(_, NOME_ESERCIZIO, CL, LR, REP, SERIE, DR, DI, PESI, STRUMENTO, TR), input_performance_weighted(PR, PS, PTR, PP), disponibilita(DAY, _)} :- variante_esercizio(_, _, CL, _, _, _, _, _, _, _, _), CL = w.


% vincoli

% Rimuovere l'answer set che contiene più di una variante dello stesso esercizio nello stesso giorno.
:- #count{STRUMENTO : esegui(NOME_ESERCIZIO, _, _, _, _, _, _, _, _, _, _, STRUMENTO, _, DAY)} > 1, esercizio(NOME_ESERCIZIO,_,_,_), disponibilita(DAY, _).

% Filtro giorni
% Rimuove solo gli answer che non contengono tutti i giorni disponibili
:- NUM_GIORNI_IN_ANSWER = #count{DAY : esegui(_,_,_,_,_,_,_,_,_,_,_,_,_,DAY)}, NUM_DAYS = #count{DAY : disponibilita(DAY,_)}, NUM_GIORNI_IN_ANSWER != NUM_DAYS.


% Filtro livello
% Usato per stabilire una continuità tra il numero dei livelli
next_level(X,Z) :- esegui(_,_,_,_,_,X,_,_,_,_,_,_,_,_), X = Z-1, #false : X < Y, esegui(_,_,_,_,_,Y,_,_,_,_,_,_,_,_), Y < Z; esegui(_,_,_,_,_,Z,_,_,_,_,_,_,_,_), X < Z.

% % se livello attuale è diverso a 1 (diverso da b1), cancella gli answer che hanno 0 next_level.
% :- livello_attuale(_,Livello_ramo), Livello_ramo != 1, #count{X,Y : next_level(X,Y)} = 0.
% da cancellare probabilmente.

% % cancella gli answers che hanno 2 o più next_level(_,_).
:- #count{X,Y: next_level(X,Y)} > 1.

% % controlla se il livello attuale è compreso nel next_level, se no cancello.
:- livello_attuale(_,Livello_ramo), next_level(_,X), Livello_ramo != X.

% % prendo tutti gli atoms di esercizio dove ci possono essere al massimo 2 livelli. (Esempio se sono livello B3, avrò esercizi solo di B3 e B2).
:- #count{Ramo, Livello_ramo : esegui(_,_,_,_,Ramo,Livello_ramo,_,_,_,_,_,_,_,_)} > 2.
% nel caso S5, abbiamo answer set con esercizi B1 S4 S5 che non va bene, il vincolo serve per cancellare questo answer.

% tolgo tutti gli esercizi che hanno il livello_ramo > di livello attuale.
:- livello_attuale(_, Livello_attuale), esegui(_,_,_,_,_,Livello_ramo,_,_,_,_,_,_,_,_), Livello_ramo > Livello_attuale.

% filtro per lettera categoria esercizio.
:- #count{Categoria: esegui(_,_,_,_,Categoria,_,_,_,_,_,_,_,_,_), livello_attuale(Categoria, _)} = 0.

% se sono livello s w e, non devo vedere esercizi di altri rami. Per esempio se sono s5, non devo vedere e4 e w4.
:- livello_attuale(Categoria,_), esegui(_,_,_,_,Cat_es,_,_,_,_,_,_,_,_,_), Categoria != Cat_es, Cat_es != b.



% Filtro per strumenti a disposizione.
countStrumenti(CS, Strumento) :- CS = #count{NOME_ESERCIZIO, DAY : esegui(NOME_ESERCIZIO,_,_,_,_,_,_,_,_,_,_,Strumento,_,DAY)}, strumento(Strumento). % conta quante volte un esercizio utilizza quel determinato strumento.
countEs(CE) :- CE = #count{NOME_ESERCIZIO, Strumento, DAY : esegui(NOME_ESERCIZIO,_,_,_,_,_,_,_,_,_,_,Strumento,_,DAY)}. % contiamo gli esercizi con chiave NOME_ESERCIZIO, Strumento e DAY.
% compara il numero totale degli esercizi con la somma degli esercizi che vanno svolti con i nostri strumenti, e cancella gli answer set nel caso i risultati siano diversi.
:- TOTAL = #sum{CS : countStrumenti(CS, _)}, countEs(CE), TOTAL != CE.


% Calcolo tempo totale dell'allenamento per il determinato giorno.
tempoTotale(DAY, TOT) :- T = #sum{DURATA_REP*REP*SERIE + DURATA_ISO*SERIE + TR*SERIE : esegui(_,_,_,_,_,_,REP,SERIE,DURATA_REP,DURATA_ISO,_,_,TR,DAY)}, disponibilita(DAY, _), warmup_time(WT), TOT = T+WT.

% Prende l'answer set che si avvicina di più ai tempi disponibili dati in input dall'utente.
#minimize { |TEMPO_DISP-TOT|@1,DAY : tempoTotale(DAY, TOT), disponibilita(DAY, TEMPO_DISP) }.

% #minimize { VAR@2 : DELTA_SUM = #sum{D, DAY : deltaGiorno(DAY, D)}, N_GIORNI = #count{DAY : disponibilita(DAY, _)}, MEDIA = DELTA_SUM / N_GIORNI, NUM = #sum{(DELTA - MEDIA)**2, DAY : deltaGiorno(DAY, DELTA), mediaDelta(MEDIA)}, nGiorni(N_GIORNI), VAR = NUM / N_GIORNI}

% deltaGiorno(DAY, DELTA) :- DELTA = |TEMPO_DISP-TOT|, tempoTotale(DAY, TOT), disponibilita(DAY, TEMPO_DISP).
% mediaDelta(MEDIA) :- DELTA = #sum{D, DAY : deltaGiorno(DAY, D)}, N_GIORNI = #count{DAY : disponibilita(DAY, _)}, MEDIA = DELTA / N_GIORNI.
% nGiorni(N_GIORNI) :- N_GIORNI = #count{DAY : disponibilita(DAY, _)}.
% varianzaDelta(VAR) :- NUM = #sum{(DELTA - MEDIA)**2, DAY : deltaGiorno(DAY, DELTA), mediaDelta(MEDIA)}, nGiorni(N_GIORNI), VAR = NUM / N_GIORNI.


% Da usare fattori utilità

% mediaUtilita(MEDIA_FU_SUPERIORE) :- MEDIA_FU_SUPERIORE = ,MEDIA_FU_SUPERIORE < 6. % da fare


% vincolo varietà



%% TODOS

% FATTO dopo aver fatto la lista degli esercizi e database

% DA FARE conto dei minuti per ogni giorno che sono occupati da tutti gli esercizi che abbiamo scelto giorno per giorno.

% DA VALUTARE calcolare l'intensità se la sessione giornaliera è pesante.

% DA VALUTARE vedere se abbiamo dei giorni vicini con next_level, se sì, verificare che non abbiano un carico alto.


#show esegui/14.
% #show next_level/2.
% #show countStrumenti/2.
#show countEs/1.
% #show tempoTotale/2.
% #show deltaGiorno/2.
% #show mediaDelta/1.
% #show nGiorni/1.
% #show varianzaDelta/1.

% #show test/3.