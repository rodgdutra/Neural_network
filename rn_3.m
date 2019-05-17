% importando tabela de informações
data = readtable('C:\Users\Rodrigo\Desktop\inteligencia_comp\scripts_rn\dados.csv');
% Vetores com as entradas
A = transpose(data.Var1);
B = transpose(data.Var2);
C = transpose(data.Var3);
D = transpose(data.Var4);
p = [A;B;C;D];
% Vetor com a saída
t = transpose(data.Var5);
% Criando a rede
net = newff(p,t,30);
net.divideFcn='divideind';
% Dividindo parametros de treinamento validação e teste
% pegando 90 entradas das primeiras 150 entradas
train_ind = [floor(1:1.666:50),floor(51:1.666:100),floor(101:1.666:150)];
net.divideParam.trainInd=train_ind;
net.divideParam.valInd=1:5:150;
net.divideParam.testInd=2:5:150;
[net,tr] = trainlm(net,p,t);
x = 1:1:150;
a = sim(net,p);
subplot(2,1,1);
plot(x,t,'o',x,a,'m+');
grid on 
xlabel("Indice das entradas")
ylabel("Padrão de saída")
legend('Padrão real','RN com 1 neurônio','Location','southeast')
subplot(2,1,2);
error = abs(t-a)
plot(error,'r');
