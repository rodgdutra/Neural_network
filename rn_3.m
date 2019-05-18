% importando tabela de informações
data = readtable('C:\Users\Rodrigo\Desktop\inteligencia_comp\scripts_rn\dados.csv');
% Vetores com as entradas
A = transpose(data.Var1);
B = transpose(data.Var2);
C = transpose(data.Var3);
D = transpose(data.Var4);

% Matriz das entradas
input = [A;B;C;D];

% Vetor com a saída
output = transpose(data.Var5);

% Criando a rede
net = newff(input,output,1);
net.divideFcn='divideind';

% Dividindo parametros de treinamento validação e teste
% pegando 90 entradas das primeiras 150 entradas
net.divideParam.trainInd=floor(1:1.666:150);

% 30 dados de validação
net.divideParam.valInd=1:5:150;

% 30 dados de teste
test_id                 = 2:5:150;
net.divideParam.testInd = test_id;

% Treinando a rede
[net,tr]     = trainlm(net,input,output);
test_in      = [A(test_id);B(test_id);C(test_id);D(test_id)];
test_out     = round(net(test_in));

% Dados para plot
id       = 1:1:150;
net_out  = round(sim(net,input));
sim_err  = round(abs(output-net_out));
err_id   = find(sim_err);
net_err  = net_out(err_id);

subplot(2,1,1);
plot(id,output,'o',id,net_out,'m+',test_id,test_out,'y*',err_id,net_err,'rd');
grid on 
xlabel("Indice das entradas")
ylabel("Padrão de saída")
legend('Padrão real','Rede Neural','RN teste','Erro da RN','Location','southeast')
subplot(2,1,2);
plot(sim_err,'rd');
xlabel("Indice das entradas")
ylabel("Erro da rede saída")
legend('Erro','Location','southeast')