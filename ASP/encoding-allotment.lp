%input
% pacchettiDisponibili( iDPacchetto, fornitore, meta, tipoStruttura, prezzo, durata, data_partenza, pax, prezzoAcquisto, valutazione, disponibilità).
% richiestaPrevista(ClassePacchetto, Quantita)
% classePacchetto(CP, Meta, Tipo, Pmin, Pmax, DurataMin, DurataMax)
% availablePackages(PackageId, Stockist, Destination, Type, Price, From, To, Pax, Income, Rating, Quantity).
% requiredPackages(ClassId, Quantity, Destination, Type, MinPrice, MaxPrice, MinDuration, MaxDuration ).

%sistemiamo richiesta prevista in un'unica tabella
richiestaPrevista(Quantita, Meta, Tipo, Pmin, Pmax, I1, I2) :- richiestaPrevista(Gruppo, CP, Quantita), classePacchetto(CP, Meta, Tipo, Pmin, Pmax, I1, I2).

buy(P,F,M,T,Pr,PrA,V,Q1) | notBuy(P,F,M,T,Pr,PrA,V,Q1) 
            :- richiestaPrevista(Q,M,T,Pmin,Pmax,_,_), pacchettiDisponibili(P,F,M,T,Pr,_,_,Pax,PrA,V,Di),Pr<=Pmax,Pr>=Pmin, maxNumero(Q1),Q1>0,Q1<=Di,Q1<=Q.

:- #count{Q:buy(Pk,_,_,_,_,_,_,Q) } > 1, pacchettiDisponibili(Pk,_,_,_,_,_,_,_,_,_,_).

fornQuant(F,Q) :- buy(_,F,_,_,_,_,_,Q).
fornQuant(F,Q) :- notBuy(_,F,_,_,_,_,_,Q).

% Applica sconto
correttivo(F,R) :- fornQuant(F,Q), Q >= B, sconto(F,B,R).
correttivo(F,0) :- pacchettiDisponibili(_,F,_,_,_,_,_,_,_,_,_).
maxCorrettivo(F,Rmax) :- #max{R : correttivo(F,R)} = Rmax, correttivo(F,_).

finalBuy(P,F,M,T,Pr,(PrA-(PrA*Rmax)/100),V,Q) :- buy(P,F,M,T,Pr,PrA,V,Q), maxCorrettivo(F,Rmax).
finalNotBuy(P,F,M,T,Pr,(PrA-(PrA*Rmax)/100),V,Q) :- notBuy(P,F,M,T,Pr,PrA,V,Q), maxCorrettivo(F,Rmax).

% Rispettiamo il limite di budget
:- budget(B), #sum{ C,Pk: finalBuy(Pk,_,_,_,_,PrA,_,Q), C = PrA*Q } > B.

% Massimizza la rendita
:~ finalNotBuy(P,F,M,T,Pr,PrA,V,Q), R=(Pr-PrA)*Q. [R@10,P]

% Massimizza i pacchetti da comprare
:~ notBuy(Pk,_,_,_,_,_,_,Q). [Q@6,Pk,Q]

% Preferisce certi fornitori a seconda della meta
:~ preferenzaFornitorePerMeta(F,M,V), notBuy(_,F,M,_,_,_,_,Q). [Q*V@5,F,M]

% Preferisce certi fornitori a seconda della struttura
:~ preferenzaFornitorePerStruttura(F,T,V), notBuy(_,F,_,T,_,_,_,Q). [Q*V@5,F,T]

% Preferisci i pacchetti con valutazione utente piu alta
:~ notBuy(Pk,_,_,_,_,_,V,Q). [Q*V@4,Pk]

% Minimizza i soldi da sborsare
:~ finalBuy(Pk,_,_,_,Pr,_,_,Q). [Pr*Q@2,Pk]

#show buy/8.
