% Input dell'utente
% esercizio(nome_esercizio, fattore_utilita_superiore, fattore_utilita_core, fattore_utilita_inferiore).
% variante_esercizio(id, nome_esercizio, categoria_esercizio, livello_esercizio, n_rep, n_serie, durata_isometria, pesi_extra, strumento, tempo_riposo).

% input utente
livello_attuale(s,5). % livello attuale dell'utente (categoria livello, livello_ramo)
disponibilita(1, 3600). % giorno disponibile espresso in numero, secondi dell'allenamento disponibili.
disponibilita(99, 3600). % giorno disponibile espresso in numero, secondi dell'allenamento disponibili.
input_performance(2,1,-10, 0). % n_rep, n_serie, tempo riposo, pesi.
%input performance... +-k dove k è rep o serie in più o meno

% ragionare su questo atom per quando avremo rep negativi
% mettere il tempo di esecuzione per i vari esercizi? per quando si ottimizza i tempi?

% % check strumenti
strumento("pullup bar").
% haveElasticBand.
% haveWeights.


% Livello B1
esercizio("pushup rialzato prona",7,3,1).
variante_esercizio(1, "pushup rialzato prona", b, 1, 3, 1, 0, 0, "corpo libero", 30).
variante_esercizio(2, "pushup rialzato prona", b, 1, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(3, "pushup rialzato prona", b, 1, 3, 1, 0, 0, "fixed parallel bars", 30).

esercizio("pushup rialzato neutro",7,3,1).
variante_esercizio(5, "pushup rialzato neutro", b, 1, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(6, "pushup rialzato neutro", b, 1, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(7, "pushup rialzato neutro", b, 1, 3, 1, 0, 0, "fixed parallel bars", 30).

% esercizio("alzate laterali",6,1,1). % valutare un attimino i fattori utilità per i vari esercizi
% variante_esercizio(9,"alzate laterali",b,1,3,1,0,0,"corpo libero",30).

% esercizio("biceps curl",7,1,1).
% variante_esercizio(10, "biceps curl",b,1,3,1,0,2,"dumbbells",30).
% variante_esercizio(11, "biceps curl",b,1,3,1,0,5,"elastic band",30). % 5 10 15 20 25

% esercizio("australian pullups 45",7,3,1).
% variante_esercizio(12,"australian pullups 45",b,1,3,1,0,0,"medium height bar",30).
% variante_esercizio(13,"australian pullups 45",b,1,3,1,0,0,"rings",30).

% esercizio("australian scapula shrugs",7,3,1).
% variante_esercizio(14,"australian scapula shrugs",b,1,3,1,0,0,"medium height bar",30).
% variante_esercizio(15,"australian scapula shrugs",b,1,3,1,0,0,"rings",30).

% esercizio("bar hang",5,3,0).
% variante_esercizio(16,"bar hang",b,1,1,1,10,0,"pullup bar",30).
% variante_esercizio(17,"bar hang",b,1,1,1,10,0,"rings",30).

% esercizio("knee raises",3,6,1).
% variante_esercizio(18,"knee raises",b,1,3,1,0,0,"pullup bar",30).
% variante_esercizio(19,"knee raises",b,1,3,1,0,0,"portable parallel bars",30).

% esercizio("crunches",0,6,0).
% variante_esercizio(20, "crunches", b,1,3,1,0,0,"corpo libero",30).

% esercizio("barchetta tuck",1,5,2).
% variante_esercizio(21,"barchetta tuck",b,1,1,1,10,0,"corpo libero",30).

% esercizio("squat assistito",0,2,6).
% variante_esercizio(22, "squat assistito", b, 1, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(23, "squat assistito", b, 1, 3, 1, 0, 0, "medium height bar", 30).
% variante_esercizio(24, "squat assistito", b, 1, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(25, "squat assistito", b, 1, 3, 1, 0, 0, "fixed parallel bars", 30).



% % Range? Livelli base no, ma livelli alti vanno bene mettere un range di rep/serie.


% % Livello B2

% esercizio("pushup ginocchia a terra prona",7,3,1).
% variante_esercizio(26, "pushup ginocchia a terra prona", b, 2, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(27, "pushup ginocchia a terra prona", b, 2, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(28, "pushup ginocchia a terra prona", b, 2, 3, 1, 0, 0, "parallettes", 30).

% esercizio("pushup ginocchia a terra neutro",7,3,1).
% variante_esercizio(29, "pushup ginocchia a terra neutro", b, 2, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(30, "pushup ginocchia a terra neutro", b, 2, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(31, "pushup ginocchia a terra neutro", b, 2, 3, 1, 0, 0, "parallettes", 30).

esercizio("barchetta adv tuck",1,6,2).
variante_esercizio(32,"barchetta adv tuck",b,2,1,1,10,0,"corpo libero",30).

% esercizio("scapula shrugs",7,3,1).
% variante_esercizio(33,"scapula shrugs",b,2,3,1,0,0,"pullup bar",30).
% variante_esercizio(34,"scapula shrugs",b,2,3,1,0,0,"rings",30).

esercizio("normal squat",0,2,7).
variante_esercizio(35,"normal squat",b,2,3,1,0,0,"corpo libero",30).

% esercizio("l sit one leg switch",3,6,3).
% variante_esercizio(36,"l sit one leg switch",b,3,6,1,0,0,"parallettes",30).
% variante_esercizio(37,"l sit one leg switch",b,3,6,1,0,0,"portable parallel bars",30).
% variante_esercizio(38,"l sit one leg switch",b,3,6,1,0,0,"fixed parallel bars",30).

% esercizio("bench dips",7,2,0).
% variante_esercizio(39,"parallel dips",b,3,3,1,0,0,"portable parallel bars",30).

% esercizio("band assisted dips",7,3,0).
% variante_esercizio(40,"band assisted dips",b,2,3,1,0,-15,"portable parallel bars",30). % uso bande elastiche -15
% variante_esercizio(41,"band assisted dips",b,2,3,1,0,-15,"fixed parallel bars",30). % uso bande elastiche -15

% esercizio("l sit tuck",3,6,2).
% variante_esercizio(42,"l sit one leg switch",b,3,1,1,5,0,"parallettes",30).
% variante_esercizio(43,"l sit one leg switch",b,3,1,1,5,0,"portable parallel bars",30).
% variante_esercizio(44,"l sit one leg switch",b,3,1,1,5,0,"fixed parallel bars",30).

% esercizio("knee scapular pushups",3,2,1).
% variante_esercizio(45,"knee scapular pushups",b,3,5,1,0,0,"corpo libero",30).


% esercizio("band assisted pullups",7,3,1).
% variante_esercizio(50,"band assisted pullups",b,2,3,1,0,-15,"pullup bar",30). % uso bande elastiche -15
% variante_esercizio(51,"band assisted pullups",b,2,3,1,0,-15,"rings",30).

% esercizio("band assisted chinup",7,3,1).
% variante_esercizio(52,"band assisted chinup",b,2,3,1,0,-15,"pullup bar",30). % uso bande elastiche -15
% variante_esercizio(53,"band assisted chinup",b,2,3,1,0,-15,"rings",30).


% esercizio("jumping pullups",7,3,1).
% variante_esercizio(54,"jumping pullups",b,2,3,1,0,0,"pullup bar",30).
% variante_esercizio(55,"jumping pullups",b,2,3,1,0,0,"rings",30).

% esercizio("jumping chinup",7,3,1).
% variante_esercizio(56,"jumping chinup",b,2,3,1,0,0,"pullup bar",30).
% variante_esercizio(57,"jumping chinup",b,2,3,1,0,0,"rings",30).

% esercizio("calf raises",0,1,7).
% variante_esercizio(58,"calf raises",b,3,3,1,0,0,"corpo libero",30).

% esercizio("bycicle crunches",1,6,3).
% variante_esercizio(59,"bycicle crunches",b,3,3,1,0,0,"corpo libero",30).


% % Livello B3


esercizio("parallel dips",8,2,0).
variante_esercizio(60,"parallel dips",b,3,3,1,0,0,"portable parallel bars",30).

% esercizio("bar dips",8,2,0).
% variante_esercizio(62,"bar dips",b,3,3,1,0,0,"medium height bar",30).
% variante_esercizio(63,"bar dips",b,3,3,1,0,0,"pullup bar",30).
% variante_esercizio(64,"bar dips",b,3,3,1,0,0,"portable parallel bars",30).
% variante_esercizio(65,"bar dips",b,3,3,1,0,0,"fixed parallel bars",30).
% esercizio("ring dips",8,3,0).
% variante_esercizio(66,"ring dips",b,3,3,1,0,0,"rings",30).

% esercizio("barchetta full",1,6,2).
% variante_esercizio(67,"barchetta full",b,3,1,1,10,0,"corpo libero",30).

% esercizio("pushup prona",7,3,1).
% variante_esercizio(68, "pushup prona", b, 3, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(69, "pushup prona", b, 3, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(70, "pushup prona", b, 3, 3, 1, 0, 0, "parallettes", 30).

% esercizio("pushup neutro",7,3,1).
% variante_esercizio(71, "pushup neutro", b, 3, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(72, "pushup neutro", b, 3, 3, 1, 0, 0, "portable parallel bars", 30).
% variante_esercizio(73, "pushup neutro", b, 3, 3, 1, 0, 0, "parallettes", 30).

% esercizio("pushup supina",7,3,1).
% variante_esercizio(74, "pushup supina", b, 3, 3, 1, 0, 0, "corpo libero", 30).
% variante_esercizio(75, "pushup supina", b, 3, 3, 1, 0, 0, "parallettes", 30).

% esercizio("pullups",7,3,1).
% variante_esercizio(80,"pullups",b,3,3,1,0,0,"pullup bar",30).
% variante_esercizio(81,"pullups",b,3,3,1,0,0,"rings",30).

% esercizio("chinup",7,3,1).
% variante_esercizio(82,"chinup",b,3,3,1,0,0,"pullup bar",30).
% variante_esercizio(83,"chinup",b,3,3,1,0,0,"rings",30).

% esercizio("l sit",4,6,2).
% variante_esercizio(84,"l sit",b,3,1,1,5,0,"fixed parallel bars",30).
% variante_esercizio(85,"l sit",b,3,1,1,5,0,"portable parallel bars",30).
% variante_esercizio(86,"l sit",b,3,1,1,5,0,"parallettes",30).
% variante_esercizio(87,"l sit",b,3,1,1,5,0,"rings",30).

% esercizio("leg raises",0,2,8).
% variante_esercizio(88,"leg raises",b,3,3,1,0,0,"pullup bar",30).

% esercizio("plank hold",1,7,1).
% variante_esercizio(88,"plank hold",b,3,1,1,20,0,"corpo libero",30).

% esercizio("commando pullups",7,3,1).
% variante_esercizio(90,"commando pullups",b,3,3,1,0,0,"pullup bar",30).

% esercizio("jump squat",0,2,8).
% variante_esercizio(100,"jump squat",b,3,3,1,0,0,"corpo libero",30).

% esercizio("side squat",0,2,7).
% variante_esercizio(101,"side squat",b,3,3,1,0,0,"corpo libero",30).

% esercizio("lungees",0,2,7).
% variante_esercizio(102,"lungees",b,3,3,1,0,0,"corpo libero",30).

% esercizio("sumo squat",0,2,7).
% variante_esercizio(103,"sumo squat",b,3,3,1,0,0,"corpo libero",30).

% esercizio("wall sit",0,1,7).
% variante_esercizio(104,"wall sit",b,3,1,1,10,0,"corpo libero",30).

% esercizio("flutter kicks",0,6,3).
% variante_esercizio(105,"flutter kicks",b,3,6,1,0,0,"corpo libero",30).

% esercizio("vertical crunches",1,6,2).
% variante_esercizio(105,"vertical crunches",b,3,3,1,0,0,"corpo libero",30).

% esercizio("lying leg raises",0,7,3).
% variante_esercizio(106,"lying leg raises",b,3,3,1,0,0,"corpo libero",30).

% esercizio("v ups",0,8,3).
% variante_esercizio(107,"v ups",b,3,3,1,0,0,"corpo libero",30).

% esercizio("toes to bar",3,8,3).
% variante_esercizio(108,"toes to bar",b,3,3,1,0,0,"pullup bar",30).

% esercizio("side plank",2,6,2).
% variante_esercizio(109,"side plank",b,3,1,1,15,0,"corpo libero",30).

% esercizio("reverse plank",3,6,2).
% variante_esercizio(110,"reverse plank",b,3,1,1,15,0,"corpo libero",30).

% esercizio("plank wipers",2,6,2).
% variante_esercizio(111,"plank wipers",b,3,6,1,0,0,"corpo libero",30).

% esercizio("russian twist",2,6,2).
% variante_esercizio(112,"russian twist",b,3,10,1,0,0,"corpo libero",30).

% esercizio("scapular pushup",4,3,1).
% variante_esercizio(113,"scapular pushup",b,3,5,1,0,0,"corpo libero",30).












% Livello S4
esercizio("explosive pullups",8,3,1).
variante_esercizio(200,"explosive pullups",s,4,3,1,0,0,"pullup bar",30).
variante_esercizio(201,"explosive pullups",s,4,3,1,0,0,"rings",30).

% Livello S5
esercizio("advanced tuck front lever",7,3,1).
variante_esercizio(251,"advanced tuck front lever",s,5,1,1,5,0,"pullup bar",30).
variante_esercizio(252,"advanced tuck front lever",s,5,1,1,5,0,"rings",30).
variante_esercizio(253,"advanced tuck front lever",s,5,1,1,5,0,"portable parallel bars",30).
variante_esercizio(254,"advanced tuck front lever",s,5,1,1,5,0,"fixed parallel bars",30).

% Livello S6


% Livello E4
esercizio("AAApullups",7,3,1).
variante_esercizio(400,"AAApullups",e,4,5,1,0,0,"pullup bar",20).
variante_esercizio(401,"AAApullups",e,4,5,1,0,0,"rings",20).

% Livello E5

% Livello E6


% Livello W4

% Livello W5

% Livello W6
