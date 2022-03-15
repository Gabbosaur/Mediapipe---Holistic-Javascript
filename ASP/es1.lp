% isLevel(1).
% goal(parte_superiore).
% availableTime(120).

%

esercizio(squat, 6, parte_inferiore, 30).
esercizio(squat, 8, parte_inferiore, 30).
esercizio(jump_squat, 5, parte_inferiore, 30).
esercizio(crunches, 20, core, 30).
esercizio(pullup, 5, parte_superiore, 30).
esercizio(pushup, 8, parte_superiore, 30).

% serie(nome_esercizio, categoria_esercizio, numero_serie).
% serie(pushup, endurance, X).


{esercizio(squat, 5+num_liv, parte_inferiore, 30*num_cat);
 esercizio(jump_squat, 10+num_liv, parte_inferiore, 30*num_cat)}.



% do(LEVEL, NOME_ESERCIZIO, NUM_REP) :- isLevel(LEVEL), esercizio(NOME_ESERCIZIO, NUM_REP, GRUPPO_MUSCOLARE).

