clc;
clear all;
addpath('src');

dataSet = ["Biomed","DiabeticRetinopathy","Ecoli","Imports","Vowel","Concordia_0","Concordia_1"];
% dataSet = ["Biomed"];
kf = 5; trainFlag=1;

for data_num=1:length(dataSet)
    dataset_name = char(dataSet(data_num));
        load(['data/' dataset_name])
        train_pos_data_all = train_pos_data_all';

        %%%Train-test data Normalization %%%
        [ntrain_data_all norm_valt] = norm_denorm(train_pos_data_all', 2, 1);
        ntest_data = norm_denorm(test_data', 2, 0, norm_valt);

        train_pos_data_all = ntrain_data_all';
        test_data = ntest_data'; 
        clear ntest_data;
        %%% End of Normalization %%%
        
    if trainFlag==1
        %%%% Final Prepared Data    
        for fld = 1:kf
            train_data = folds_train(fld).train_pos';
            val_data = folds_train(fld).val';
            val_labels = folds_train(fld).val_labels;

            %%%Train-validation data Normalization %%%
            [ntrain_data norm_val] = norm_denorm(train_data', 2, 1);
            nval_data = norm_denorm(val_data', 2, 0, norm_val);

            clear train_data val_data;
            train_data = ntrain_data'; % Just for keeping in the required format
            val_data = nval_data'; 
            clear ntrain_data nval_data;
            %%% End of Normalization %%%

            % Range of all parameter
            Range_gamma = 0;
            Range_rr = 1;
            Range_C = power(2,-5:5);
            Range_Cl = power(2,-5:5);
            temp_ind = 0;
            for i=1:length(Range_gamma)
                gama = Range_gamma(i);
                for j=1:length(Range_C)
                    C = Range_C(j);
                    for k=1:length(Range_Cl)
                        Cl = Range_Cl(k);
                        for gl=1:length(Range_rr) % Graph Laplacian parameter
                            rr = Range_rr(gl); 
                            for noofcluster=1:10

                                Laplacian = [];
                                [labelval_AAKELMMSCV_tmp labeltr_AAKELMMSCV_tmp] = VAAKELM(train_data,val_data,1,C,Cl,gama,noofcluster,Laplacian,rr);                                                               
                                if fld==1
                                    [labels_AAKELMMSCV_tmp, ~] = VAAKELM(train_pos_data_all,test_data,1,C,Cl,gama,noofcluster,Laplacian,rr);
                                end

                                for frej=1:3 
                                    labelval_AAKELMMSCV = labelval_AAKELMMSCV_tmp(:,frej);
                                    [accuAAMLMSCVval(fld,i,j,k,gl,noofcluster,frej) sensAAMLMSCVval(fld,i,j,k,gl,noofcluster,frej)....
                                        specAAMLMSCVval(fld,i,j,k,gl,noofcluster,frej) precAAMLMSCVval(fld,i,j,k,gl,noofcluster,frej)...
                                        recAAMLMSCVval(fld,i,j,k,gl,noofcluster,frej) f11AAMLMSCVval(fld,i,j,k,gl,noofcluster,frej)...
                                        gmAAMLMSCVval(fld,i,j,k,gl,noofcluster,frej)] = Evaluate(val_labels,labelval_AAKELMMSCV,1);
                                end
                                clear Laplacian;
                            end
                        end
                    end
                end
            end  
        end

        %%%% Save validation results
        save(['Results/' dataset_name '_Results_val'], 'accuAAMLMSCVval','sensAAMLMSCVval','specAAMLMSCVval','precAAMLMSCVval','recAAMLMSCVval','f11AAMLMSCVval','gmAAMLMSCVval');     
    end
    
	load(['Results/' dataset_name '_Results_val']);
	mu = [0.01 0.05 0.1];
	
	for frej=1:3
		Range_gamma = 0;
		Range_rr = 1;
		Range_C = power(2,-5:5);
		Range_Cl = power(2,1);
		for i=1:length(Range_gamma)
			for j=1:length(Range_C)
				for k=1:length(Range_Cl)
					for gl=1:length(Range_rr)
						for noofcluster=1:10
							avg_gmAAMLMSCVval(i,j,k,gl,noofcluster,frej) = mean(gmAAMLMSCVval(:,i,j,k,gl,noofcluster,frej));
						end
					end
				end
			end
		end
	end
	
	[max_gmAAMLMSCVval indmax] =max(avg_gmAAMLMSCVval(:));
	%%% Order of name of parameter in these indexes Range_gamma,Range_C,Range_Cl,Range_rr,noofcluster
	[AACVgind3 AACVgind4 AACVgind5 AACVgind6 AACVgind7 AACVgind8]=ind2sub(size(avg_gmAAMLMSCVval),indmax);
	noofcluster=1:10;
	param = ["Range_gamma" "Range_C" "Range_Cl" "Range_rr" "noofcluster" "frej"; ...
		Range_gamma Range_C(AACVgind4) Range_Cl(AACVgind5) Range_rr noofcluster(AACVgind7) mu(AACVgind8)];
		
	%Begin Testing phase
	Laplacian = [];
	%Setting parameters
	gama = str2double(param(2,1));
	C = str2double(param(2,2));
	Cl = str2double(param(2,3));
	rr = str2double(param(2,4));
	noofcluster = str2double(param(2,5));
	frej = AACVgind8;
	%Calling model
	[labels_AAKELMMSCV_tmp, ~] = VAAKELM(train_pos_data_all,test_data,1,C,Cl,gama,noofcluster,Laplacian,rr);
	labels_AAKELMMSCV = labels_AAKELMMSCV_tmp(:,frej);
	[accu sens spec prec rec f11 gm] = Evaluate(test_labels,labels_AAKELMMSCV,1);
	clear Laplacian;
	
	f11 = round(f11*100,2); gm = round(gm*100,2); accu = round(accu*100,2); prec = round(prec*100,2); rec = round(rec*100,2);
	save(['Results/' dataset_name '_Results.mat'],'dataset_name','f11','gm','accu','prec','rec');

	disp(dataset_name);
    fprintf('  F1 Score for VAAKELM: %s\n', num2str(f11));
    fprintf('  Parameters selected for VAAKELM:\n');
	fprintf('     Regularization parameter (C): %s\n', param(2,2));
	fprintf('     Graph Regularization parameter (lambda): %s\n', param(2,4));
	fprintf('     Number of clusters (k): %s\n', param(2,5));
	fprintf('     Percentage of dismissal (delta): %s\n', param(2,6));
	
end