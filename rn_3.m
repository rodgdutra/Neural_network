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
output  = transpose(data.Var5);

% Valores para repetições 
total_rep = 100;

% Gerando valores de index de treino, validação e teste
train_id = round(1:1.666:150);
ids_possible = (1:1:150);

% Alocando os vetores de teste e validação 
val_id  = 1:2:60;
test_id = 2:2:60;
for i = 1:1:90
    ids_possible(train_id(i)) = 0;
end
ids_possible = find(ids_possible);

for i = 1:1:30
    val_id(i) = ids_possible(val_id(i));
    test_id(i)= ids_possible(test_id(i));
end

if total_rep~=0
    N_hidden = [1 5 10 15 30 60];
    results = zeros(1,6); 
    error_table = zeros(total_rep,6);
    for rep = 1:total_rep

    for i = 1:6

    % Criando a rede
    net = newff(input,output,N_hidden(i));
    net.divideFcn='divideind';

    % Dividindo parametros de treinamento validação e teste
    % pegando 90 entradas das primeiras 150 entradas
    net.divideParam.trainInd=train_id;

    % 30 dados de validação
    net.divideParam.valInd=val_id;

    % 30 dados de teste
    
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

    results(i) = length(net_err);
    end
    true_error = results*100/150;
    error_table(rep,1:6) = true_error;
    end

    avarege_error = zeros(1,6);
    error_sum = zeros(1,6);
    for i = 1:6
    avarege_error(i) = sum(error_table(:,i))/total_rep;
    error_sum(i) = sum(error_table(:,i));
    end
else
    % Criando a rede
    net = newff(input,output,1);
    net.divideFcn='divideind';

    % Dividindo parametros de treinamento validação e teste
    % pegando 90 entradas das primeiras 150 entradas
    net.divideParam.trainInd=train_id;

    % 30 dados de validação
    net.divideParam.valInd=val_id;

    % 30 dados de teste
    net.divideParam.testInd = test_id;

    % Treinando a rede
    [net,tr]     = trainlm(net,input,output);
    
    % Saída de teste
    test_in      = [A(test_id);B(test_id);C(test_id);D(test_id)];
    test_out     = round(net(test_in));
     
    % Dados para plot
    id       = 1:1:150;
    net_out  = round(sim(net,input));
    sim_err  = round(abs(output-net_out));
    err_id   = find(sim_err);
    net_err  = net_out(err_id);
end

subplot(2,1,1);
plot(id,output,'o',id,net_out,'m+',test_id,test_out,'g*',err_id,net_err,'rd');
grid on 
xlabel("Indice das entradas")
ylabel("Padrão de saída")
legend('Padrão real','Rede Neural','RN teste','Erro da RN','Location','southeast')

subplot(2,1,2);
semilogy(tr.epoch,tr.perf,'b',tr.epoch,tr.vperf,'g',tr.epoch,tr.tperf,'r','LineWidth',2);
xlabel("Epoch")
ylabel("MSE")
legend('Train performance',' Validation performance',' Test performance','Location','northeast')
