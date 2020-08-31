clear all; close all; clc;

%set your dataset path and saliency map result path.
dataset = 'Mine';
gtPath = ['../datasets/testingnew/mask/'];
salPath = ['/home/stamatis/Desktop/Imperial Thesis/Brain_Tumor_Segmentation_US_Images/CRF/crf_results/10/'];

%obtain the total number of image (ground-truth)
imgFiles = dir(gtPath);
imgNUM = length(imgFiles)-2;

%evaluation score initilization.
Smeasure=zeros(1,imgNUM);
Emeasure=zeros(1,imgNUM);
MAE=zeros(1,imgNUM);
F_wm=zeros(1,imgNUM);


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
    %name = name(:,3:10);
    
    %load gt
    gt = imread([gtPath name]);
    
    if numel(size(gt))>2
        gt = rgb2gray(gt);
    end
    if ~islogical(gt)
        gt = gt(:,:,1) > 128;
    end
    
    %load salency
    sal  = imread([salPath name]);
    
    %check size
    if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
        sal = imresize(sal,size(gt));
        imwrite(sal,[salPath name]);
        fprintf('Error occurs in the path: %s!!!\n', [salPath name]);
       
    end
    
    sal = im2double(sal(:,:,1));
    
    %normalize sal to [0, 1]
    sal = reshape(mapminmax(sal(:)',0,1),size(sal));
    
    Smeasure(i) = StructureMeasure(sal,logical(gt));
    
    temp = calc_metrics(sal,double(gt),size(gt)); % Using the 2 times of average of sal map as the threshold.
    Accuracy(i) = temp(1);
    MeanAccuracy(i) = temp(2);
    Precision(i) = temp(3);
    Recall(i) = temp(4);
    F1_score(i) = temp(5);
    FmeasureF(i) = temp(6);
    MeanIou(i) = temp(7);
    WeightedIou(i) = temp(8);
   
    MAE(i) = mean2(abs(double(logical(gt)) - sal));
    F_wm(i) = WFb(sal, logical(gt));
    
    %You can change the method of binarization method. As an example, here just use adaptive threshold.
    threshold =  2* mean(sal(:)) ;
    if ( threshold > 1 )
        threshold = 1;
    end
    Bi_sal = zeros(size(sal));
    Bi_sal(sal>threshold)=1;
    Emeasure(i) = Enhancedmeasure(Bi_sal,gt);
    
end

toc;

Sm = mean2(Smeasure);
Em = mean2(Emeasure);
mae = mean2(MAE);
F_wm = mean2(F_wm);

%%%MyCode
Acc = mean2(Accuracy)*100;
MeanAcc = mean2(MeanAccuracy)*100;
Prec = mean2(Precision)*100;
Rec = mean2(Recall)*100;
F1 = mean2(F1_score)*100;
Fm = mean2(FmeasureF)*100;
MeanIoU = mean2(MeanIou)*100;
WeightedIoU = mean2(WeightedIou)*100;
%%%

fprintf('(%s Dataset)Emeasure: %.3f; Smeasure: %.3f; weighted_F: %.3f; Fmeasure %.3f; MAE: %.3f.\n',dataset,Em, Sm, F_wm, Fm, mae);
fprintf('Global Accuracy: %.2f; Mean Accuracy: %.2f; MeanIoU: %.2f; WeightedIoU: %.2f; Precision: %.2f; Recall: %.2f; F1: %.2f; Fmeasure: %.2f.\n', Acc, MeanAcc, MeanIoU, WeightedIoU, Prec, Rec, F1, Fm);
fprintf(newline)

for i = 1:imgNUM
    fprintf('image %d; Global Accuracy: %.2f; Mean Accuracy: %.2f; MeanIoU: %.2f; WeightedIoU: %.2f; Precision: %.2f; Recall: %.2f; F1: %.2f; Fmeasure: %.2f.\n',i, Accuracy(i)*100, MeanAccuracy(i)*100, MeanIou(i)*100, WeightedIou(i)*100, Precision(i)*100, Recall(i)*100, F1_score(i)*100, FmeasureF(i)*100);
end