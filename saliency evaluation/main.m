%{
Code structure is taken from https://github.com/jiwei0921/Saliency-Evaluation-Toolbox
but the code for all metrics, besides FmeasureF, is implemented from scratch.
%}
clear all; close all; clc;

%Set your dataset path and saliency map result path.
gtPath = ['../datasets/testingnew/mask/'];
salPath = ['../CRF/crf_results/10/'];

%Obtain the total number of images (ground-truth)
imgFiles = dir(gtPath);
imgNUM = length(imgFiles)-2;

MAE=zeros(1,imgNUM);

%Initilization of metric variables.
Precision=zeros(1,imgNUM);
Recall=zeros(1,imgNUM);
Accuracy=zeros(1,imgNUM);
MeanAccuracy=zeros(1,imgNUM);
F1_score=zeros(1,imgNUM);
FmeasureF=zeros(1,imgNUM);
MeanIou=zeros(1,imgNUM);
WeightedIou=zeros(1,imgNUM);

tic;
for i = 1:imgNUM
    
    fprintf('Evaluating: %d/%d\n',i,imgNUM);
    
    name =  imgFiles(i+2).name;
    
    %Load ground truth map
    gt = imread([gtPath name]);
    
    if numel(size(gt))>2
        gt = rgb2gray(gt);
    end
    if ~islogical(gt)
        gt = gt(:,:,1) > 128;
    end
    
    %Load saliency map
    sal  = imread([salPath name]);
    
    %Check size
    if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
        sal = imresize(sal,size(gt));
        imwrite(sal,[salPath name]);
        fprintf('Error occurs in the path: %s!!!\n', [salPath name]);
       
    end
    
    sal = im2double(sal(:,:,1));
    
    %Normalize saliency map to [0, 1]
    sal = reshape(mapminmax(sal(:)',0,1),size(sal));
    
    temp = calc_metrics(sal,double(gt),size(gt));
    Accuracy(i) = temp(1);
    MeanAccuracy(i) = temp(2);
    Precision(i) = temp(3);
    Recall(i) = temp(4);
    F1_score(i) = temp(5);
    FmeasureF(i) = temp(6);
    MeanIou(i) = temp(7);
    WeightedIou(i) = temp(8);
    MAE(i) = mean2(abs(double(logical(gt)) - sal));

end

toc;

mae = mean2(MAE);
Acc = mean2(Accuracy)*100;
MeanAcc = mean2(MeanAccuracy)*100;
Prec = mean2(Precision)*100;
Rec = mean2(Recall)*100;
F1 = mean2(F1_score)*100;
Fm = mean2(FmeasureF)*100;
MeanIoU = mean2(MeanIou)*100;
WeightedIoU = mean2(WeightedIou)*100;

%Print metrics for the whole dataset.
fprintf('\nGlobal Accuracy: %.2f; Mean Accuracy: %.2f; MeanIoU: %.2f; WeightedIoU: %.2f; Precision: %.2f; Recall: %.2f; F1: %.2f; Fmeasure: %.2f; MAE: %.4f.\n',  Acc, MeanAcc, MeanIoU, WeightedIoU, Prec, Rec, F1, Fm, mae);
fprintf(newline)

%Print per image metrics.
for i = 1:imgNUM
    fprintf('image %d; Global Accuracy: %.2f; Mean Accuracy: %.2f; MeanIoU: %.2f; WeightedIoU: %.2f; Precision: %.2f; Recall: %.2f; F1: %.2f; Fmeasure: %.2f.\n',i, Accuracy(i)*100, MeanAccuracy(i)*100, MeanIou(i)*100, WeightedIou(i)*100, Precision(i)*100, Recall(i)*100, F1_score(i)*100, FmeasureF(i)*100);
end