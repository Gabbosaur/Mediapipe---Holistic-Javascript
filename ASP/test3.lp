% {esercizio(pushup,1..3, b, 1); esercizio(pushup,4, b, 2)}.
% next(0,1).
next(1,2).
next(2,3).
livello_attuale(b,3).

check :- livello_attuale(_, Livello_ramo), next(Livello_ramo-1, Livello_ramo).
:- not check.
