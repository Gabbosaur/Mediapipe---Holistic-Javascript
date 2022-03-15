availablePackages('milano', 10).
availablePackages('milano', 5).
availablePackages('milano', 4).
availablePackages('genova', 5).
%availablePackages('firenze',5).


buy('milano', 10).
buy('milano', 5).
buy('milano', 4).
buy('genova', 5).
buy('firenze', 5).



:- #count{Q,P: buy(P,Q)} >1, availablePackages(T,_).

