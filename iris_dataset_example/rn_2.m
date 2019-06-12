% Vetor de entrada com 2 padrões de entrada
p = [0 1 2 3 4 5 6 7 8; 0 1 2 3 4 5 6 7 8];
% Saída do sistema
t = [0 0.84 0.91 0.14 -0.77 -0.96 -0.28 0.66 0.99]
net = newff(p,t,14)
net.divideFcn='divideind';
net.divideParam.trainInd=1:2:9;
net.divideParam.valInd=2:2:9;
net.divideParam.testInd=2:2:9;
[net,tr] = train(net,p,t)
a = sim(net,p)
plot(t,'o');
hold
plot(a,'m+')