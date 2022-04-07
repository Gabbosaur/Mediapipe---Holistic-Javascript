% unione tabelle (TODOS: unire anche il filtro-giorni e filtro-livello e forse altro(?))
{esegui(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE, CL, LR, REP+PR, SERIE+PS, DI, PESI+PP, STRUMENTO, TR+PTR, DAY) :  esercizio(NOME_ESERCIZIO, FU_SUPERIORE, FU_CORE, FU_INFERIORE), variante_esercizio(_, NOME_ESERCIZIO, CL, LR, REP, SERIE, DI, PESI, STRUMENTO, TR), input_performance(PR, PS, PTR, PP), disponibilita(DAY, _)}.


% vincoli

% Rimuovere l'answer set che contiene più di una variante dello stesso esercizio nello stesso giorno.

% test(DOPPI, NOME_ESERCIZIO, DAY) :- DOPPI = #count{ STRUMENTO : esegui(NOME_ESERCIZIO, _, _, _, _, _, _, _, _, _, STRUMENTO, _, DAY)}, esercizio(NOME_ESERCIZIO,_,_,_), disponibilita(DAY,_).
:- #count{STRUMENTO : esegui(NOME_ESERCIZIO, _, _, _, _, _, _, _, _, _, STRUMENTO, _, DAY)} > 1, esercizio(NOME_ESERCIZIO,_,_,_), disponibilita(DAY, _).

% Filtro giorni
% Rimuove solo gli answer che non contengono tutti i giorni disponibili
:- NUM_GIORNI_IN_ANSWER = #count{DAY : esegui(_,_,_,_,_,_,_,_,_,_,_,_,DAY)}, NUM_DAYS = #count{DAY : disponibilita(DAY,_)}, NUM_GIORNI_IN_ANSWER != NUM_DAYS.


% Filtro livello
% Usato per stabilire una continuità tra il numero dei livelli
next_level(X,Z) :- esegui(_,_,_,_,_,X,_,_,_,_,_,_,_), X = Z-1, #false : X < Y, esegui(_,_,_,_,_,Y,_,_,_,_,_,_,_), Y < Z; esegui(_,_,_,_,_,Z,_,_,_,_,_,_,_), X < Z.

% % se livello attuale è diverso a 1 (b1), cancella gli answer che hanno 0 next_level.
:- livello_attuale(_,Livello_ramo), Livello_ramo != 1, #count{X,Y : next_level(X,Y)} = 0.

% % cancella gli answers che hanno 2 o più next_level(_,_).
:- #count{X,Y: next_level(X,Y)} > 1.

% % controlla se il livello attuale è compreso nel next_level, se no cancello.
:- livello_attuale(_,Livello_ramo), next_level(_,X), Livello_ramo != X.

% % prendo tutti gli atoms di esercizio dove ci possono essere al massimo 2 livelli. (Esempio se sono livello B3, avrò esercizi solo di B3 e B2).
:- #count{Ramo, Livello_ramo : esegui(_,_,_,_,Ramo,Livello_ramo,_,_,_,_,_,_,_)} > 2.
% nel caso S5, abbiamo answer set con esercizi B1 S4 S5 che non va bene, il vincolo serve per cancellare questo answer.

% tolgo tutti gli esercizi che hanno il livello_ramo > di livello attuale.
:- livello_attuale(_, Livello_attuale), esegui(_,_,_,_,_,Livello_ramo,_,_,_,_,_,_,_), Livello_ramo > Livello_attuale.

% filtro per lettera categoria esercizio.
:- #count{Categoria: esegui(_,_,_,_,Categoria,_,_,_,_,_,_,_,_), livello_attuale(Categoria, _)} = 0.

% se sono livello s w e, non devo vedere esercizi di altri rami. Per esempio se sono s5, non devo vedere e4 e w4.
:- livello_attuale(Categoria,_), esegui(_,_,_,_,Cat_es,_,_,_,_,_,_,_,_), Categoria != Cat_es, Cat_es != b.

% filtro per strumenti a disposizione.
% conta quante volte un esercizio utilizza quel determinato strumento.
countStrumenti(CS, Strumento) :- CS = #count{NOME_ESERCIZIO, DAY : esegui(NOME_ESERCIZIO,_,_,_,_,_,_,_,_,_,Strumento,_,DAY)}, strumento(Strumento).
% contiamo gli esercizi con chiave NOME_ESERCIZIO, Strumento e DAY.
countEs(CE) :- CE = #count{NOME_ESERCIZIO, Strumento, DAY : esegui(NOME_ESERCIZIO,_,_,_,_,_,_,_,_,_,Strumento,_,DAY)}.

:- TOTAL = #sum{CS : countStrumenti(CS, _)}, countEs(CE), TOTAL != CE.


%% TODOS

% FATTO dopo aver fatto la lista degli esercizi e database

% DA FARE conto dei minuti per ogni giorno che sono occupati da tutti gli esercizi che abbiamo scelto giorno per giorno.

% DA VALUTARE calcolare l'intensità se la sessione giornaliera è pesante.

% DA VALUTARE vedere se abbiamo dei giorni vicini con next_level, se sì, verificare che non abbiano un carico alto.


#show esegui/13.
% #show next_level/2.
#show countStrumenti/2.
#show countEs/1.
#show total/1.
% #show test/3.