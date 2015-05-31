clc;
clear all;
%%TRAINING IMAGES
%Choose the path to the training data

TrainDatabasePath = uigetdir('/Users/gayathribalasubramaniam/Documents/MATLAB/', 'Select training database path' );
TrainFiles = dir(TrainDatabasePath);
ImageCount = 0;

%count the number of images
for i = 1:size(TrainFiles,1)
        ImageCount = ImageCount + 1;
end
T = [];
disp(ImageCount)


for i = 1 : ImageCount-2
   
    FileName = int2str(i);
    FileName = strcat('/',FileName,'.jpg');
    disp(FileName);
    FileName = strcat(TrainDatabasePath,FileName);
    
    %reads the image from the path
    
    img = imread(FileName);
    
    % convert to a grey scaled image
    
    img = rgb2gray(img);
    [rows columns numberOfColorChannels] = size(img);
    disp(rows);
    disp(columns);
    temp = reshape(img',rows*columns,1); 
    [rows columns numberOfColorChannels] = size(temp);
    disp(rows);
    disp(columns);
    T = [T temp];
end
disp('Output Matrix: features * images');
[rows columns numberOfColorChannels] = size(T)
%disp(rows);
%disp(columns);

%%Columns denote the images and rows the features

%%Perform PCA on this data

%calculating mean subtracted data
 
meanValue =  mean(T,2);  % 36000 x 1 %for each feature
disp('Mean Value');
disp(size(meanValue));
%subtract from original data


MeanSubData = []

for i = 1 : ImageCount - 2
    temp = double(T(:,i)) - meanValue ;  %mean subtracted data for ith image
    MeanSubData = [MeanSubData temp]; %merging into new data set
end

disp('Mean Subtracted Data');
 disp(size(MeanSubData));   
 
 %covariance matrix is equal to MeanSubData' * MeanSubData
 
 % eigen values are the diagonal elements
 
 SCovMatrix = MeanSubData' * MeanSubData;  % cov = MeeanSubData * MeanSubData' SCovMatrix is a substitue for cov inorder to resuce dimensions.
 
 %[V,D] = eig(A) produces matrices of eigenvalues (D) and eigenvectors (V) of matrix A, so that A*V = V*D. 
 
 [V,D] = eig(SCovMatrix);
 
 
 %eigen values D are in sorted order
 
 %playing a threshold on these values to remove few eigen vectors.
 %Resultant  eigen vectors will be less that number of images in the matrix

 EigenVectors = []
 
 for i = 1 : size(V,2)
     if(D(i,i) > 1)
         EigenVectors = [ EigenVectors V(:,i)];
     end
 end
 
 
 %EigenVectors for the Cov matrix is also called EigenFaces
 
 EigenFaces = MeanSubData * EigenVectors;
 disp('EigenFaces');
 disp(size(EigenFaces));
 
 
ReducedImg = [];

for i = 1 : ImageCount - 2 
    temp = EigenFaces' * MeanSubData(:,i);
    ReducedImg = [ReducedImg temp];
end
disp('Reduced Data');
disp(size(ReducedImg));
%disp(ReducedImg); % 19 features * 20 images

%%TEST IMAGE 

%reading test image

TestDatabasePath = uigetdir('/Users/gayathribalasubramaniam/Documents/MATLAB/', 'Select test database path' );
TestFiles = dir(TestDatabasePath);
prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title ='Test Image';
default = {'1'};

TestImage  = inputdlg(prompt,dlg_title,1,default);
TestImage = strcat(TestDatabasePath,'/',char(TestImage),'.jpg');
img = imread(TestImage);

Test = []

%converting to gray scaled image

img = rgb2gray(img);
disp('Test Image');
[rows columns numberOfColorChannels] = size(img)
%disp(rows);
%disp(columns);

%converting 2D image to 1D image

temp = reshape(img',rows*columns,1); 

[rows columns numberOfColorChannels] = size(temp);
%disp(rows);
%disp(columns);

Test = [Test temp];

disp('test img after reshape');
%disp(Test);
disp(size(Test));

%Subtracting data and mean

MeanSubTest = []
temp1 = double(Test) - meanValue ;  %mean subtracted data for test image
MeanSubTest = [MeanSubTest temp1]; 

%applying PCA on test data
disp('Feature Reduced Test Img');
ReducedTestImg = EigenFaces' * MeanSubTest;
%disp(ReducedTestImg);
disp(size(ReducedTestImg));   % 19 features * 1 image

%KNN 

EuclideanDistance = [];
for i = 1 : ImageCount - 2
    q = ReducedImg(:,i);
    temp = sqrt(( norm( ReducedTestImg - q ) )^2);
    EuclideanDistance = [EuclideanDistance temp];
end

%[EuclideanDistance_min , Recognized_index] = min(EuclideanDistance);
%A = [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10]

A = zeros(1,ImageCount-2);
imgCnt = 1
for i = 1 : (ImageCount-2)/2 
   A(1,imgCnt) = i;
   A(1,imgCnt+1) = i;
   imgCnt = imgCnt + 2;
end
disp(A);
EuclideanDistance = [A;EuclideanDistance]
x=sortrows(EuclideanDistance',2)

 
% min1 = x(1,2);
% min2 = x(2,2);
% min3 = x(3,2);
% min4 = x(4,2);
% min5 = x(5,2);
% in1 = x(1,1)
% in2 = x(2,1)
% in3 = x(3,1)
% in4 = x(4,1)
% in5 = x(5,1)

%Weignted KNN
k= 5
cntNN = zeros(1,10);
for i = 1 : k 
    dis = x(i,2);
    ind = x(i,1);
    cntNN(1,ind) = cntNN(1,ind)+(1/dis);

end
[maxVal , pos ] = max(cntNN) ;
disp(pos);
Recognized_index = pos*2;
OutputName = strcat(int2str(Recognized_index),'.jpg')
EquivImage = strcat(TrainDatabasePath,'/',OutputName);
EquivImg = imread(EquivImage);
img = imread(TestImage);
imshow(img)
title('Test Image');
figure,imshow(EquivImg);
title('Equivalent Image');
disp('Test image');
disp(TestImage);
disp('Matched image');
disp(EquivImage);
