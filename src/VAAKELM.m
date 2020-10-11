
function [labels labeltr layerinfo] = AAKELM_MSCV_ML_allmu(train_data,test_data,max_layer,C,Cl,gamma,noofclusters,L,lambda)

for layer =1:max_layer
    if layer==1
        [K, Ktest, ~] = kernel_rbf(train_data,test_data,gamma);
        II = eye(size(K,1));

        % Start of Embedding Minimum Sub-class Variance ELM
        S = mscv_calc(K, ones(size(K,2),1), noofclusters);

        %%% Simply add Graphical information to calculated minimum variance based feature
        %%% space values, So, code of GOCKELM_ML would be intact after these lines
        if ~isempty(L)
            S = S+L; %%% Just modify this line from previous version
            % L=S;  %%% For the case when S and L both minimized at each layer otherwise comment it
        end
        % End of Minimum Sub-class Variance ELM
        
        if (layer~=max_layer)
            if nargin < 8
                B = K + II/C;                       %1st layer of MLAAKELM
            elseif nargin == 8
                B = K + (S*K)/C;                    %1st layer of Unregularized MLAAKELM_MSCV
            elseif nargin ==9       
                B = K + (lambda/C)*(S*K) + II/C;    %1st layer of Regularized MLAAKELM_MSCV (Used)
            end
        
            a = pinv(B) * train_data';
            Ot = a' * Ktest;
            Otr = a' * K;
        else
            Ot = test_data;
            Otr = train_data;
            Cl = C;
        end
    end
    
    if layer==max_layer
        clear B a K Ktest;
        [K, Ktest, ~] = kernel_rbf(Otr,Ot,gamma);
        II = eye(size(K,1));
        
        if (max_layer==1)
            if nargin < 8           
                B = K + II/Cl;                      %AAKELM
            elseif nargin == 8      
                B = K + (S*K)/Cl;                   %Unregularized AAKELM_MSCV
            elseif nargin ==9       
                B = K + (lambda/Cl)*(S*K) + II/Cl;  %Regularized AAKELM_MSCV (Used)
            end
        else
            if nargin < 8           
                B = K + II/Cl;      %Final layer of MLAAKELM
            elseif nargin == 8      
                B = K + (L*K)/Cl;   %Final layer of Unregularized MLAAKELM_MSCV
            elseif nargin ==9       
                if ~isempty(L)
                    B = K + (lambda/Cl)*(L*K) + II/Cl;
                else
                    B = K + II/Cl;  %Final layer of Regularized MLAAKELM_MSCV (Used)
                end
            end
        end
        
        try
        a = pinv(B) * Otr';
        catch
            warning('Problem with pinv, SVD did not converge');
        a = inv(B) * Otr';
        end
        clear Ot Otr;
        Ot = a' * Ktest;
        Otr = a' * K;
        
    elseif ((layer~=max_layer) & (layer~=1))
        clear B a K Ktest;
        [K, Ktest, ~] = kernel_rbf(Otr,Ot,gamma);
        II = eye(size(K,1));
        
        if nargin < 8           %Intermediate layer of MLAAKELM
            B = K + II/C;
        elseif nargin == 8
            B = K + (L*K)/C;    %Intermediate layer of Unregularized MLAAKELM_MSCV
        elseif nargin ==9
            if ~isempty(L)
            B = K + (lambda/C)*(L*K) + II/C;
            else
                B = K + II/C;   %Intermediate layer of Regularized MLAAKELM_MSCV (Used)
            end
        end
        try
            a = pinv(B) * Otr';
        catch
            warning('Problem with pinv, SVD did not converge');
            a = inv(B) * Otr';
        end
        clear Ot Otr;        
        Ot = a' * Ktest;
        Otr = a' * K;
    end
    
    %%%% Just gathering the information regarding outputweight of all layers and output
    %%%% of autoencoder at each layer except last layer as last is not an
    %%%% autoencoder but calculate final output.
    if layer==max_layer
        OutputWeight_max_layers= a';
    else
        OutputWeights(:,:,layer)= a';
        layer_autoenc_trains(:,:,layer)= Otr;
        layer_autoenc_tests(:,:,layer)= Ot;
    end
end

%%% Just storing all information into a single variable as a structre
layerinfo.OutputWeight_max_layer=OutputWeight_max_layers;
if (layer~=1)
layerinfo.OutputWeight=OutputWeights;
layerinfo.OutputWeight_max_layer=OutputWeight_max_layers;
layerinfo.layer_autoenc_train=layer_autoenc_trains;
layerinfo.layer_autoenc_test=layer_autoenc_tests;
end

%%% One-class Classification as per single node output %%%
%%%%% Try 5% rejection
%%% For training
mu=[0.01 0.05 0.1];

difftr = sum((train_data'-Otr').^2,2);
[sout,~] = sort(difftr);
m = size(train_data,2);
for frej=1:3
    labeltr_temp = ones(m,1)*2; 
    thresh = sout(ceil(m*(1-mu(frej))),1);
    labeltr_temp(difftr<thresh)=1;
    labeltr(:,frej) = labeltr_temp;

    %%% For Testing
    labels_temp = ones(size(test_data,2),1)*2;
    diffs = sum((test_data'-Ot').^2,2);
    labels_temp(diffs<thresh)=1;
    labels(:,frej) = labels_temp;
end

%%%%% End of 10% rejection

%%% For training
% meanOtr = mean(Otr);
% thresh =meanOtr*mu;%epsilon
% labeltr = ones(size(K,2),1)*2;
% difftr = abs(Otr-meanOtr);
% labeltr(difftr<thresh)=1;

%%% For Testing
% labels = ones(size(Ktest,2),1)*2;
% diffs = abs(Ot-meanOtr);
% labels(diffs<thresh)=1;
%%% End of One-class Classification as per single node output %%%
