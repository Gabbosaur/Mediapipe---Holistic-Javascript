% Input dell'utente
% esercizio(nome_esercizio, fattore_utilita_superiore, fattore_utilita_core, fattore_utilita_inferiore).
% variante_esercizio(id, nome_esercizio, categoria_esercizio, livello_ramo, n_rep, n_serie, durata_rep, durata_isometria, pesi_extra, strumento, tempo_riposo).

% input utente
livello_attuale(s,5). % livello attuale dell'utente (categoria livello, livello_ramo)
disponibilita(1, 1800). % giorno disponibile espresso in numero, secondi dell'allenamento disponibili.
disponibilita(3, 1500). % giorno disponibile espresso in numero, secondi dell'allenamento disponibili.
% disponibilita(5, 1000). % giorno disponibile espresso in numero, secondi dell'allenamento disponibili.
% disponibilita(6, 1600).
% il form compilato dall'utente produrrà questi input_performance, in base agli attrezzi di supporto (pesi extra e bande elastiche)
% i valori in input_performance verranno bilanciati di conseguenza.
input_performance_base(2,5,30, 0). % n_rep, n_serie, tempo riposo, pesi extra. (Ricalibrare poi in Python il file di output)
input_performance_skills(2,1,-10, 0). % n_rep, n_serie, tempo riposo, pesi extra. (Ricalibrare poi in Python il file di output)
input_performance_endurance(2,1,-10, 0). % n_rep, n_serie, tempo riposo, pesi extra. (Ricalibrare poi in Python il file di output)
input_performance_weighted(2,1,-10, 0). % n_rep, n_serie, tempo riposo, pesi extra. (Ricalibrare poi in Python il file di output)
%input performance... +-k dove k è rep o serie in più o meno

% ragionare su questo atom per quando avremo rep negativi
% mettere il tempo di esecuzione per i vari esercizi? per quando si ottimizza i tempi?

warmup_time(600). % secondi di tempo di riscaldamento

% % check strumenti
strumento("corpo libero").
% strumento("portable parallel bars").
strumento("portable parallel bars").


% BOZZA 1 (easy)
% Preferenza gruppo muscolare ("parte_superiore", "core", "parte_inferiore")
pref_gm("core").


% BOZZA 2 (hard)
% Preferenza gruppo muscolare ("parte_superiore", "core", "parte_inferiore")
% pref_gm("parte_superiore", 10, "core", 8, "parte_inferiore", 1).


% haveElasticBand.
% haveWeights.


% Livello B1
esercizio("pushup rialzato prona",7,3,1).
% AGGIUNGERE DURATA_REP PER OGNI VARIANTE
variante_esercizio(1, "pushup rialzato prona", b, 1, 3, 1, 3, 0, 0, "corpo libero", 30).
variante_esercizio(2, "pushup rialzato prona", b, 1, 3, 1, 3, 0, 0, "portable parallel bars", 30).
variante_esercizio(3, "pushup rialzato prona", b, 1, 3, 1, 3, 0, 0, "fixed parallel bars", 30).

esercizio("pushup rialzato neutro",7,3,1).
variante_esercizio(5, "pushup rialzato neutro", b, 1, 3, 1, 3, 0, 0, "corpo libero", 30).
variante_esercizio(6, "pushup rialzato neutro", b, 1, 3, 1, 3, 0, 0, "portable parallel bars", 30).
variante_esercizio(7, "pushup rialzato neutro", b, 1, 3, 1, 3, 0, 0, "fixed parallel bars", 30).

esercizio("alzate laterali",6,1,1). % valutare un attimino i fattori utilità per i vari esercizi
variante_esercizio(9,"alzate laterali",b,1,3,1,3,0,0,"corpo libero",30).

esercizio("biceps curl",7,1,1).
variante_esercizio(10, "biceps curl",b,1,3,1,2,0,2,"dumbbells",30).
variante_esercizio(11, "biceps curl",b,1,3,1,2,0,5,"elastic band",30). % 5 10 15 20 25

esercizio("australian pullups 45",7,3,1).
variante_esercizio(12,"australian pullups 45",b,1,3,1,3,0,0,"medium height bar",30).
variante_esercizio(13,"australian pullups 45",b,1,3,1,3,0,0,"rings",30).

esercizio("australian scapula shrugs",7,3,1).
variante_esercizio(14,"australian scapula shrugs",b,1,3,1,2,0,0,"medium height bar",30).
variante_esercizio(15,"australian scapula shrugs",b,1,3,1,2,0,0,"rings",30).

esercizio("bar hang",5,3,0).
variante_esercizio(16,"bar hang",b,1,1,1,0,10,0,"pullup bar",30).
variante_esercizio(17,"bar hang",b,1,1,1,0,10,0,"rings",30).

esercizio("knee raises",3,6,1).
variante_esercizio(18,"knee raises",b,1,3,1,2,0,0,"pullup bar",30).
variante_esercizio(19,"knee raises",b,1,3,1,2,0,0,"portable parallel bars",30).

esercizio("crunches",0,6,0).
variante_esercizio(20, "crunches", b,1,3,1,2,0,0,"corpo libero",30).

esercizio("barchetta tuck",1,5,2).
variante_esercizio(21,"barchetta tuck",b,1,0,1,0,10,0,"corpo libero",30).
% % CAMBIARE NEL CASO TUTTI GLI ES ISOMETRIE LE REP A 0.


esercizio("half squat",0,2,6).
variante_esercizio(22, "half squat", b, 1, 3, 1, 2, 0, 0, "corpo libero", 30).
variante_esercizio(23, "half squat", b, 1, 3, 1, 2, 0, 0, "medium height bar", 30).
variante_esercizio(24, "half squat", b, 1, 3, 1, 2, 0, 0, "portable parallel bars", 30).
variante_esercizio(25, "half squat", b, 1, 3, 1, 2, 0, 0, "fixed parallel bars", 30).

esercizio("calf raises",0,1,6).
variante_esercizio(1026, "calf raises", b, 1, 3, 1, 1, 0, 0, "corpo libero", 30).



% % Range? Livelli base no, ma livelli alti vanno bene mettere un range di rep/serie.


% % Livello B2

esercizio("pushup ginocchia a terra prona",7,3,1).
variante_esercizio(26, "pushup ginocchia a terra prona", b, 2, 3, 1, 2, 0, 0, "corpo libero", 30).
variante_esercizio(27, "pushup ginocchia a terra prona", b, 2, 3, 1, 2, 0, 0, "portable parallel bars", 30).
variante_esercizio(28, "pushup ginocchia a terra prona", b, 2, 3, 1, 2, 0, 0, "parallettes", 30).

esercizio("pushup ginocchia a terra neutro",7,3,1).
variante_esercizio(29, "pushup ginocchia a terra neutro", b, 2, 3, 1, 2, 0, 0, "corpo libero", 30).
variante_esercizio(30, "pushup ginocchia a terra neutro", b, 2, 3, 1, 2, 0, 0, "portable parallel bars", 30).
variante_esercizio(31, "pushup ginocchia a terra neutro", b, 2, 3, 1, 2, 0, 0, "parallettes", 30).

esercizio("barchetta adv tuck",1,6,2).
variante_esercizio(32,"barchetta adv tuck",b,2,1,1,0,10,0,"corpo libero",30).

esercizio("scapula shrugs",7,3,1).
variante_esercizio(33,"scapula shrugs",b,2,3,1,2,0,0,"pullup bar",30).
variante_esercizio(34,"scapula shrugs",b,2,3,1,2,0,0,"rings",30).

esercizio("normal squat",0,2,7).
variante_esercizio(35,"normal squat",b,2,3,1,2,0,0,"corpo libero",30).

esercizio("l sit one leg switch",3,6,3).
variante_esercizio(36,"l sit one leg switch",b,2,6,1,1,0,0,"parallettes",30).
variante_esercizio(37,"l sit one leg switch",b,2,6,1,1,0,0,"portable parallel bars",30).
variante_esercizio(38,"l sit one leg switch",b,2,6,1,1,0,0,"fixed parallel bars",30).

esercizio("bench dips",7,2,0).
variante_esercizio(39,"parallel dips",b,2,3,1,2,0,0,"portable parallel bars",30).

esercizio("band assisted dips",7,3,0).
variante_esercizio(40,"band assisted dips",b,2,3,1,2,0,-15,"portable parallel bars",30). % uso bande elastiche -15
variante_esercizio(41,"band assisted dips",b,2,3,1,2,0,-15,"fixed parallel bars",30). % uso bande elastiche -15

esercizio("l sit tuck",3,6,2).
variante_esercizio(42,"l sit tuck",b,2,1,1,0,5,0,"parallettes",30).
variante_esercizio(43,"l sit tuck",b,2,1,1,0,5,0,"portable parallel bars",30).
variante_esercizio(44,"l sit tuck",b,2,1,1,0,5,0,"fixed parallel bars",30).

esercizio("knee scapular pushups",3,2,1).
variante_esercizio(45,"knee scapular pushups",b,2,5,1,2,0,0,"corpo libero",30).

esercizio("calf raises one leg",0,1,7).
variante_esercizio(46,"calf raises one leg",b,2,6,1,1,0,0,"corpo libero",30).

esercizio("pullup isometric hold", 7, 3, 1).
variante_esercizio(47, "pullup isometric hold", b, 2, 3, 1, 5, 0, 0, "pullup bar", 30).
esercizio("chinup isometric hold", 7, 3, 1).
variante_esercizio(48, "chinup isometric hold", b, 2, 3, 1, 5, 0, 0, "pullup bar", 30).
esercizio("pike hold", 7, 3, 1).
variante_esercizio(49, "pike hold", b, 2, 1, 3, 0, 5, 0, "corpo libero", 30).
variante_esercizio(1049, "pike hold", b, 2, 1, 3, 0, 5, 0, "parallettes", 30).


esercizio("band assisted pullups",7,3,1).
variante_esercizio(50,"band assisted pullups",b,2,3,1,3,0,-15,"pullup bar",30). % uso bande elastiche -15
variante_esercizio(51,"band assisted pullups",b,2,3,1,3,0,-15,"rings",30).

esercizio("band assisted chinup",7,3,1).
variante_esercizio(52,"band assisted chinup",b,2,3,1,3,0,-15,"pullup bar",30). % uso bande elastiche -15
variante_esercizio(53,"band assisted chinup",b,2,3,1,3,0,-15,"rings",30).


esercizio("jumping pullups",7,3,1).
variante_esercizio(54,"jumping pullups",b,2,3,1,3,0,0,"pullup bar",30).
variante_esercizio(55,"jumping pullups",b,2,3,1,3,0,0,"rings",30).

esercizio("jumping chinup",7,3,1).
variante_esercizio(56,"jumping chinup",b,2,3,1,3,0,0,"pullup bar",30).
variante_esercizio(57,"jumping chinup",b,2,3,1,3,0,0,"rings",30).


esercizio("pike negative pushup", 7, 3, 1).
variante_esercizio(58, "pike negative pushup", b, 2, 3, 1, 5, 0, 0, "corpo libero", 30).
variante_esercizio(1058, "pike negative pushup", b, 2, 3, 1, 5, 0, 0, "parallettes", 30).


esercizio("bycicle crunches",1,6,3).
variante_esercizio(59,"bycicle crunches",b,2,10,1,1,0,0,"corpo libero",30).






% % Livello B3


esercizio("parallel dips",8,2,0).
variante_esercizio(60,"parallel dips",b,3,3,1,3,0,0,"portable parallel bars",30).

esercizio("bar dips",8,2,0).
variante_esercizio(62,"bar dips",b,3,3,1,3,0,0,"medium height bar",30).
variante_esercizio(63,"bar dips",b,3,3,1,3,0,0,"pullup bar",30).
variante_esercizio(64,"bar dips",b,3,3,1,3,0,0,"portable parallel bars",30).
variante_esercizio(65,"bar dips",b,3,3,1,3,0,0,"fixed parallel bars",30).
esercizio("ring dips",8,3,0).
variante_esercizio(66,"ring dips",b,3,3,1,3,0,0,"rings",30).

esercizio("barchetta full",1,6,2).
variante_esercizio(67,"barchetta full",b,3,1,1,0,10,0,"corpo libero",30).

esercizio("pushup prona",7,3,1).
variante_esercizio(68, "pushup prona", b, 3, 3, 1, 3, 0, 0, "corpo libero", 30).
variante_esercizio(69, "pushup prona", b, 3, 3, 1, 3, 0, 0, "portable parallel bars", 30).
variante_esercizio(70, "pushup prona", b, 3, 3, 1, 3, 0, 0, "parallettes", 30).

esercizio("pushup neutro",7,3,1).
variante_esercizio(71, "pushup neutro", b, 3, 3, 1, 3, 0, 0, "corpo libero", 30).
variante_esercizio(72, "pushup neutro", b, 3, 3, 1, 3, 0, 0, "portable parallel bars", 30).
variante_esercizio(73, "pushup neutro", b, 3, 3, 1, 3, 0, 0, "parallettes", 30).

esercizio("pushup supina",7,3,1).
variante_esercizio(74, "pushup supina", b, 3, 3, 1, 3, 0, 0, "corpo libero", 30).
variante_esercizio(75, "pushup supina", b, 3, 3, 1, 3, 0, 0, "parallettes", 30).

esercizio("pullups",7,3,1).
variante_esercizio(80,"pullups",b,3,3,1,3,0,0,"pullup bar",30).
variante_esercizio(81,"pullups",b,3,3,1,3,0,0,"rings",30).

esercizio("chinup",7,3,1).
variante_esercizio(82,"chinup",b,3,3,1,3,0,0,"pullup bar",30).
variante_esercizio(83,"chinup",b,3,3,1,3,0,0,"rings",30).

esercizio("l sit",4,6,2).
variante_esercizio(84,"l sit",b,3,1,1,0,5,0,"fixed parallel bars",30).
variante_esercizio(85,"l sit",b,3,1,1,0,5,0,"portable parallel bars",30).
variante_esercizio(86,"l sit",b,3,1,1,0,5,0,"parallettes",30).
variante_esercizio(87,"l sit",b,3,1,1,0,5,0,"rings",30).

esercizio("leg raises",0,2,8).
variante_esercizio(88,"leg raises",b,3,3,1,2,0,0,"pullup bar",30).

esercizio("plank hold",1,7,1).
variante_esercizio(88,"plank hold",b,3,1,1,0,20,0,"corpo libero",30).

esercizio("commando pullups",7,3,1).
variante_esercizio(90,"commando pullups",b,3,3,1,3,0,0,"pullup bar",30).

esercizio("vertical crunches",1,6,2).
variante_esercizio(99,"vertical crunches",b,3,3,1,2,0,0,"corpo libero",30).

esercizio("jump squat",0,2,8).
variante_esercizio(100,"jump squat",b,3,3,1,3,0,0,"corpo libero",30).

esercizio("side squat",0,2,7).
variante_esercizio(101,"side squat",b,3,3,1,2,0,0,"corpo libero",30).

esercizio("lungees",0,2,7).
variante_esercizio(102,"lungees",b,3,3,1,2,0,0,"corpo libero",30).

esercizio("sumo squat",0,2,7).
variante_esercizio(103,"sumo squat",b,3,3,1,2,0,0,"corpo libero",30).

esercizio("wall sit",0,1,7).
variante_esercizio(104,"wall sit",b,3,1,1,0,10,0,"corpo libero",30).

esercizio("flutter kicks",0,6,3).
variante_esercizio(105,"flutter kicks",b,3,6,1,1,0,0,"corpo libero",30).

esercizio("lying leg raises",0,7,3).
variante_esercizio(106,"lying leg raises",b,3,3,1,2,0,0,"corpo libero",30).

esercizio("v ups",0,8,3).
variante_esercizio(107,"v ups",b,3,3,1,2,0,0,"corpo libero",30).

esercizio("toes to bar",3,8,3).
variante_esercizio(108,"toes to bar",b,3,3,1,2,0,0,"pullup bar",30).

esercizio("side plank",2,6,2).
variante_esercizio(109,"side plank",b,3,1,1,0,15,0,"corpo libero",30).

esercizio("reverse plank",3,6,2).
variante_esercizio(110,"reverse plank",b,3,1,1,0,15,0,"corpo libero",30).

esercizio("plank wipers",2,6,2).
variante_esercizio(111,"plank wipers",b,3,6,1,2,0,0,"corpo libero",30).

esercizio("russian twist",2,6,2).
variante_esercizio(112,"russian twist",b,3,10,1,1,0,0,"corpo libero",30).

esercizio("scapular pushup",4,3,1).
variante_esercizio(113,"scapular pushup",b,3,5,1,1,0,0,"corpo libero",30).

esercizio("pike pushup",7,3,1).
variante_esercizio(114,"pike pushup",b,3,3,1,2,0,0,"corpo libero",30).
variante_esercizio(115,"pike pushup",b,3,3,1,2,0,0,"parallettes",30).







% Livello S4
esercizio("explosive pullups",8,3,1).
variante_esercizio(200,"explosive pullups",s,4,3,1,3,0,0,"pullup bar",30).
variante_esercizio(201,"explosive pullups",s,4,3,1,3,0,0,"rings",30).

esercizio("skin the cat",6,6,1).
variante_esercizio(202,"skin the cat",s,4,1,1,5,0,0,"pullup bar",30).
variante_esercizio(203,"skin the cat",s,4,1,1,5,0,0,"rings",30).

% Planche
esercizio("planche lean",7,3,1).
variante_esercizio(204,"planche lean",s,4,1,3,0,5,0,"corpo libero",30).

esercizio("frog stand",6,4,1).
variante_esercizio(205,"frog stand",s,4,1,3,0,5,0,"corpo libero",30).
variante_esercizio(206,"frog stand",s,4,1,3,0,5,0,"parallettes",30).

esercizio("pike pushup rialzato",7,3,1).
variante_esercizio(207,"pike pushup rialzato",s,4,3,1,2,0,0,"corpo libero",30).
variante_esercizio(208,"pike pushup rialzato",s,4,3,1,2,0,0,"parallettes",30).

% Back lever
esercizio("tuck back lever",6,4,1).
variante_esercizio(210,"tuck back lever",s,4,1,3,0,5,0,"pullup bar",30).
variante_esercizio(211,"tuck back lever",s,4,1,3,0,5,0,"fixed parallel bars",30).
variante_esercizio(212,"tuck back lever",s,4,1,3,0,5,0,"portable parallel bars",30).

esercizio("tuck back lever pulse",6,4,1).
variante_esercizio(215,"tuck back lever pulse",s,4,3,2,3,0,0,"pullup bar",30).
variante_esercizio(216,"tuck back lever pulse",s,4,3,2,3,0,0,"fixed parallel bars",30).
variante_esercizio(217,"tuck back lever pulse",s,4,3,2,3,0,0,"portable parallel bars",30).

% Front lever
esercizio("tuck front lever",6,6,1).
variante_esercizio(220,"tuck front lever",s,4,1,3,0,5,0,"pullup bar",30).
variante_esercizio(221,"tuck front lever",s,4,1,3,0,5,0,"fixed parallel bars",30).
variante_esercizio(222,"tuck front lever",s,4,1,3,0,5,0,"portable parallel bars",30).

esercizio("tuck front lever pulse",6,6,1).
variante_esercizio(225,"tuck front lever pulse",s,4,3,2,3,0,0,"pullup bar",30).
variante_esercizio(226,"tuck front lever pulse",s,4,3,2,3,0,0,"fixed parallel bars",30).
variante_esercizio(227,"tuck front lever pulse",s,4,3,2,3,0,0,"portable parallel bars",30).

% Handstand
esercizio("back to wall handstand",7,5,1).
variante_esercizio(230,"back to wall handstand",s,4,1,3,0,5,0,"corpo libero",30).

esercizio("elbow lever",4,3,1).
variante_esercizio(235,"elbow lever",s,4,1,1,0,5,0,"corpo libero",30).
variante_esercizio(236,"elbow lever",s,4,1,1,0,5,0,"parallettes",30).

% Muscle up
esercizio("explosive pullup with swing",8,4,1).
variante_esercizio(240,"explosive pullup with swing",s,4,2,3,2,0,0,"pullup bar",30).

esercizio("jumping muscle up",7,4,3).
variante_esercizio(241,"jumping muscle up",s,4,3,3,3,0,0,"pullup bar",30).

esercizio("negative muscle up",7,4,3).
variante_esercizio(242,"negative muscle up",s,4,3,3,5,0,0,"pullup bar",30).















% Livello S5

% Planche
esercizio("tuck planche",7,3,1).
variante_esercizio(260,"tuck planche",s,5,1,1,0,5,0,"corpo libero",30).
variante_esercizio(261,"tuck planche",s,5,1,1,0,5,0,"parallettes",30).
variante_esercizio(262,"tuck planche",s,5,1,1,0,5,0,"fixed parallel bars",30).
variante_esercizio(263,"tuck planche",s,5,1,1,0,5,0,"portable parallel bars",30).

esercizio("planche lean pushup",8,3,1).
variante_esercizio(265,"planche lean pushup",s,5,3,1,2,0,0,"corpo libero",30).
variante_esercizio(266,"planche lean pushup",s,5,3,1,2,0,0,"parallettes",30).

esercizio("l sit to tuck planche",8,5,1).
variante_esercizio(270,"l sit to tuck planche",s,5,3,1,3,0,0,"corpo libero",30).
variante_esercizio(271,"l sit to tuck planche",s,5,3,1,3,0,0,"parallettes",30).

esercizio("tuck planche raise",8,4,1).
variante_esercizio(275,"tuck planche raise",s,5,3,1,3,0,0,"portable parallel bars",30).
variante_esercizio(276,"tuck planche raise",s,5,3,1,3,0,0,"fixed parallel bars",30).

esercizio("negative planche pushup",7,4,1).
variante_esercizio(280,"negative planche pushup",s,5,3,1,5,0,0,"portable parallel bars",30).
variante_esercizio(281,"negative planche pushup",s,5,3,1,5,0,0,"fixed parallel bars",30).

% Back lever
esercizio("adv tuck back lever",7,4,1).
variante_esercizio(284,"adv tuck back lever",s,5,1,3,0,5,0,"rings",30).
variante_esercizio(285,"adv tuck back lever",s,5,1,3,0,5,0,"portable parallel bars",30).
variante_esercizio(286,"adv tuck back lever",s,5,1,3,0,5,0,"fixed parallel bars",30).
variante_esercizio(287,"adv tuck back lever",s,5,1,3,0,5,0,"pullup bar",30).


esercizio("adv tuck back lever pulse",7,4,1).
variante_esercizio(288,"adv tuck back lever pulse",s,5,3,3,3,0,0,"portable parallel bars",30).
variante_esercizio(289,"adv tuck back lever pulse",s,5,3,3,3,0,0,"fixed parallel bars",30).
variante_esercizio(290,"adv tuck back lever pulse",s,5,3,3,3,0,0,"pullup bar",30).
variante_esercizio(291,"adv tuck back lever pulse",s,5,3,3,3,0,0,"rings",30).

% Front lever
esercizio("advanced tuck front lever",7,4,1).
variante_esercizio(295,"advanced tuck front lever",s,5,1,1,0,5,0,"pullup bar",30).
variante_esercizio(296,"advanced tuck front lever",s,5,1,1,0,5,0,"rings",30).
variante_esercizio(297,"advanced tuck front lever",s,5,1,1,0,5,0,"portable parallel bars",30).
variante_esercizio(298,"advanced tuck front lever",s,5,1,1,0,5,0,"fixed parallel bars",30).

esercizio("advanced tuck front lever raises",7,4,1).
variante_esercizio(295,"advanced tuck front lever raises",s,5,3,1,3,0,0,"pullup bar",30).
variante_esercizio(296,"advanced tuck front lever raises",s,5,3,1,3,0,0,"rings",30).
variante_esercizio(297,"advanced tuck front lever raises",s,5,3,1,3,0,0,"portable parallel bars",30).
variante_esercizio(298,"advanced tuck front lever raises",s,5,3,1,3,0,0,"fixed parallel bars",30).

esercizio("tuck front lever pulse",7,5,1).
variante_esercizio(300,"tuck front lever pulse",s,5,3,1,3,0,0,"pullup bar",30).
variante_esercizio(301,"tuck front lever pulse",s,5,3,1,3,0,0,"rings",30).
variante_esercizio(302,"tuck front lever pulse",s,5,3,1,3,0,0,"portable parallel bars",30).
variante_esercizio(303,"tuck front lever pulse",s,5,3,1,3,0,0,"fixed parallel bars",30).

esercizio("tuck front lever pullup",7,5,1).
variante_esercizio(305,"tuck front lever pullup",s,5,3,1,3,0,0,"pullup bar",30).
variante_esercizio(306,"tuck front lever pullup",s,5,3,1,3,0,0,"rings",30).
variante_esercizio(307,"tuck front lever pullup",s,5,3,1,3,0,0,"portable parallel bars",30).
variante_esercizio(308,"tuck front lever pullup",s,5,3,1,3,0,0,"fixed parallel bars",30).

% Handstand
esercizio("chest to wall handstand",7,4,1).
variante_esercizio(310,"chest to wall handstand",s,5,1,3,0,5,0,"corpo libero",30).

esercizio("handstand pushup back to wall",8,3,1).
variante_esercizio(315,"handstand pushup back to wall",s,5,3,1,3,0,0,"corpo libero",30).

% Muscle up
esercizio("band assisted muscle up",8,4,1).
variante_esercizio(320,"band assisted muscle up",s,5,1,1,5,0,0,"pullup bar",30).


% % % Siccome sono lvl S5, negli answer set mi dà gli esercizi di S5 e B3. Probabile per un certo vincolo che obbliga solo B3 (?)



% % Livello S6


% Livello E4
esercizio("AAApullups",7,3,1).
variante_esercizio(400,"AAApullups",e,4,5,1,2,0,0,"pullup bar",20).
variante_esercizio(401,"AAApullups",e,4,5,1,2,0,0,"rings",20).

% Livello E5

% Livello E6


% Livello W4

% Livello W5

% Livello W6
