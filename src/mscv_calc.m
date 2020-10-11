% Minimum Sub-class Variance ELM
function [S] = mscvelm(H, train_lbls, clustersPerClass)
% H --> LxN matrix output of hidden layer (N training samples) or NxN kernel matrix output of hidden layer (N training samples)
% H_test --> LxM matrix output of hidden layer (N test samples)
% train_lbls --> Nx1 vector of classes for multi-class problem
% clustersPerClass --> Cx1 vector containing the number of clusters per class (C classes)

% calculate number of classes in targets
noOfClasses = length(unique(train_lbls));

% regularization ELM
uniqueLbls = unique(train_lbls);
noOfClasses = length(uniqueLbls);      % number of classes
[dataDim,noOfData] = size(H);
totalNoOfClusters = sum(clustersPerClass);
clusterCenters = zeros(dataDim,totalNoOfClusters);
clusterClassLabels = zeros(totalNoOfClusters,1);

% cluster each class 
clusterIdx = 0;
for cl=1:noOfClasses
    
    currH = H(:,find(train_lbls==cl));
    
    if clustersPerClass(cl) > 1
    
        ok = 0;
        while ok == 0
            rng(2,'twister'); %%%% Just to reproduce %added now
            [Idx clustersCentersT] = kmeans(currH',clustersPerClass(cl), 'EmptyAction', 'drop'); 
            %Idx contains cluster no. for each sample and clustersCentersT has centroid values

            if sum(sum(isnan(clustersCentersT))) == 0
                ok = 1;
            end
        end

        for i=1:clustersPerClass(cl)
            clusterIdx = clusterIdx +1;
            currCluster = currH(:,find(Idx==i));    %Contains samples belonging to cluster i
            clusterCenters(:,clusterIdx) = mean(currCluster,2); %Contains mean of cluster samples
            clusterClassLabels(clusterIdx) = cl;
        end
    else
        %All the data is taken in a single cluster (noofclusters=1)
        clusterIdx = clusterIdx +1;
        clusterCenters(:,clusterIdx) = mean(currH,2);   %Contains mean of total data (for noofclusters=1)
        clusterClassLabels(clusterIdx) = cl;
    end
end

clear currCluster clustersCentersT currH Idx ok totalNoOfClusters noOfData

% classify each element
clusterLabels = zeros(size(train_lbls));
for i=1:size(size(H,2))
    currClass = train_lbls(i);
    currCenters = clusterCenters(:,find(clusterClassLabels==currClass));    %same as clusterCenters
    diffMat = H(:,i)*ones(1,size(currCenters,2));   %H(:,1)
    diffMat = diffMat - currCenters;    %H(:,1) - mean of cluster samples
    distanceVec = diffMat' * diffMat;
    distanceVector = diag(distanceVec); 
    minDistance = min(distanceVector);
    clusterLabels(i) = find(distanceVector==minDistance);    
end

clear clusterClassLabels currClass currCenters diffMat distanceVec distanceVector minDistance

% compute within scatter matrix
S = zeros(dataDim,dataDim);
clusterIdx = 0;
for cl=1:noOfClasses
    for clst=1:clustersPerClass(cl)
        clusterIdx = clusterIdx +1;
        currElements = H(:,find(train_lbls==uniqueLbls(cl) & clusterLabels==clst)); %stores samples belonging to cluster clst
    
        % update within scatter matrix
        for k=1:size(currElements,2)
            diffMatrix = currElements(:,k) - clusterCenters(:,clusterIdx);  %Samples - cluster mean
            S = S + ((1.0/size(currElements,2)) * (diffMatrix*diffMatrix'));
        end
    end
end

len = length(S);
for r=10^(-8):10^(-8):1
    if rank(S) < len % if it is ill-posed
        S = S + r*eye(len,len); % reguralization
    else
        break;        
    end
end % reguralization

