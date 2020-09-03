%This function calculates the metrics used for evaluating the segmentation
%of one image.
function Metrics = calc_metrics(sMap,gtMap,gtsize)

Label3 = zeros( gtsize );
Label3( sMap>=0.5 ) = 1;           %Saliency map with ones and zeros.

P = length( find( Label3==1 ) );        %Number of predicted pixels as positives in saliency map.
LabelAnd = Label3 & gtMap;              %True positives with 1.
TP = length( find ( LabelAnd==1 ) );    %Number of true positives.
FP = P - TP;                            %False Positives.
num_obj = sum(sum(gtMap));

N = length( find( Label3==0 ) );        %Number of predicted pixels as negatives in saliency map.
LabelOr = Label3 | gtMap;               %True negatives with 0.
TN = length( find ( LabelOr==0 ) );     %Number of true negatives.
FN = N - TN;                            %False Negatives.

T = length( find ( gtMap==1 ) );        %Number of tumor pixels
B = length( find ( gtMap==0 ) );        %Number of background pixels

if TP == 0
    Precision = 0;
    Recall = 0;
    F1_score = 0;
    FmeasureF = 0;

else
    Precision = TP/P;
    Recall = TP/num_obj;
    F1_score = ((2 * Precision * Recall)/(Precision + Recall));
    %Fmeasure general formula with b weight
    FmeasureF = ((1.3 * Precision * Recall) / ( .3 * Precision + Recall ) );
end

Accuracy = 0;
MeanAccuracy = 0;
MeanIou = 0;
WeightedIou = 0;

Accuracy = (TP+TN)/(P+N);
MeanAccuracy = 0.5 * (TP/P + TN/N);
MeanIou = 0.5 * (TP/(TP+FP+FN) + TN/(TN+FP+FN));
WeightedIou = (T/(T+B))*(TP/(TP+FP+FN)) + (B/(T+B))*(TN/(TN+FP+FN));

Metrics = [Accuracy, MeanAccuracy, Precision, Recall, F1_score, FmeasureF, MeanIou, WeightedIou];
