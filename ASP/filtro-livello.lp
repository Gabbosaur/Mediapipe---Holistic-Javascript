% {esercizio(w, 1..3); esercizio(s, 1..3); esercizio(e, 1..3)} = 1.

livello_attuale(s, 5). % con 1 livello di profondità, quindi proporremo esercizi del livello attuale e del livello precedente.


{esercizio(pushup,1, b, 1)}.
% {esercizio(pushup,2, b, 1)}.
% {esercizio(pushup,3, b, 1)}.
{esercizio(pushup,4, b, 2)}.
% {esercizio(pushup,5, b, 2)}.
% {esercizio(pushup,6, b, 2)}.
% {esercizio(pushup,7, b, 2)}.
% {esercizio(pushup,8, b, 3)}.
{esercizio(pushup,9, b, 3)}.

{esercizio(tuck_frontlever, 0, s, 4)}.
{esercizio(skinthecat,1, s, 4)}.
{esercizio(advtuck_frontlever,0, s, 5)}.
{esercizio(pushup,12, e, 4)}.
% {esercizio(pushup,5, w, 4)}.
{esercizio(pushup,6, w, 4)}.

% Usato per stabilire una continuità tra il numero dei livelli
next(X,Z) :- esercizio(_,_, _, X), X = Z-1, #false : X < Y, esercizio(_,_, _, Y), Y < Z; esercizio(_,_, _ ,Z), X < Z.

% :- not next(_,_).
% :- #count{Ramo, Livello_ramo : esercizio(Ramo, Livello_ramo)} > 2.



% Ottengo 2 livelli inferiore a quello attuale + livello attuale

% se livello attuale è diverso a 1 (b1), cancella gli answer che hanno 0 next.
:- livello_attuale(_,Livello_ramo), Livello_ramo != 1, #count{X,Y : next(X,Y)} = 0.

% prendo tutti gli atoms di esercizio dove ci possono essere al massimo 2 livelli. (Esempio se sono livello B3, avrò esercizi solo di B3 e B2).
:- #count{Ramo, Livello_ramo : esercizio(_,_, Ramo, Livello_ramo)} > 2. % solo per caso B1

% cancella gli answers che hanno 2 o più next(_,_).
:- #count{X,Y: next(X,Y)} > 1.

% controlla se il livello attuale è compreso nel next, se no cancello.
:- livello_attuale(_,Livello_ramo), next(_,X), Livello_ramo != X.


% tolgo tutti gli esercizi che hanno il livello_ramo > di livello attuale.
:- livello_attuale(_, Livello_attuale), esercizio(_,_, _, Livello_ramo), Livello_ramo > Livello_attuale.


% % filtro per lettera categoria esercizio.
:- #count{Categoria: esercizio(_,_,Categoria, _), livello_attuale(Categoria, _)} = 0.

% se sono livello s w e, non devo vedere esercizi di altri rami. Per esempio se sono s5, non devo vedere e4 e w4.
:- livello_attuale(Categoria, Livello_attuale), Livello_attuale = 4, esercizio(_,_,Cat_es, _), Categoria != Cat_es, % DA FINIRE



