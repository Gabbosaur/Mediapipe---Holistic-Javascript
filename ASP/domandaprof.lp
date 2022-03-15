esercizio(1..3, 1..2).
aves(1..3).

:- aves(X), #count{X: esercizio(X,_)} = 3.

