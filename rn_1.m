% Entrada de dados 
p = [0 1 2 3 4 5 6 7 8];

% Saída real dos dados 
t = [0 0.84 0.91 0.14 -0.77 -0.96 -0.28 0.66 0.99];

% Plot do comportamento real do sistema (bola) 
plot(p,t,'o')
hold
% Criação da e simulação da rede neural 
net = newff(p,t,14)
a=sim(net,p)

% Treinando e plotando o comportamento da rede
net=train(net, p,t)
a=sim(net,p)
plot(p,a,'m+')