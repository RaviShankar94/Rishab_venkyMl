TrainDatabasePath = uigetdir('/Users/gayathribalasubramaniam/Documents/MATLAB/', 'Select training database path' );
TrainnFiles = dir(TrainDatabasePath);
for i = 1:size(TrainFiles,1)
        Train_Number = Train_Number + 1; % Number of all images in the training database
end
T = [];
for i = 1 : Train_Number
    str = int2str(i);
    str = strcat('/',str,'.jpg');
    str = strcat(TrainDatabasePath,str);
    img = imread(str);
    disp(img);
end