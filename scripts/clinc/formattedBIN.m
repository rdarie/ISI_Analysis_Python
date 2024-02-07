% Updated by: Ekta Tiwari 6-8-2022
%This is an initial script that will read and convert formatted file
%created by CLINC and will save into .mat file


function [outputStruct, outputFileName] = formattedBIN(folder_path, saving, keep_in_memory)
    if nargin < 1
        folder_path = uigetdir(pwd, "Select Capture Folder");
    end
    if nargin < 2
        saving = true;
    end
    if nargin < 3
        keep_in_memory = false;
    end
    
    if ~exist(folder_path, "dir")
        error("Could not find requested folder!")
    end

    file_list = struct2table(dir(folder_path));
    %formattedFiles = find(contains(file_list.name, "formatted.bin"));
    formattedFiles = contains(file_list.name, "f.bin");
    file_list = file_list(formattedFiles, :);
    % Read data from formatted files
    
    for this_file = 1:height(file_list)
        file_name = char(strcat(file_list.folder{this_file}, filesep, file_list.name{this_file}));
    
        fprintf("Reading %s...\n", file_name);
        fid = fopen(file_name);
        contents = fread(fid, [70, Inf], 'int16', 0, 'l')';
  

        data_this_file.Meta.filename = file_name;
        data_this_file.Meta.conversionTime = datetime('now');
        data_this_file.ChannelData = contents(:, 1:64);
        data_this_file.SyncWave = contents(:, 65);
        data_this_file.StimStart = contents(:, 66);
        data_this_file.MuxConfig = contents(:, 67);
       
        sample_count_MSB = string(dec2bin(contents(:, 68)));
        sample_count_LSB = string(dec2bin(contents(:, 69)));
        sample_count_bin = strcat(sample_count_MSB, sample_count_LSB);
        sample_count = bin2dec(sample_count_bin);

        data_this_file.SampleCount = sample_count;
        data_this_file.usbPacketCount = contents(:, 70);
        
        if saving
            [path, name, ~] = fileparts(file_name);
            outputFileName{this_file} = strcat(path, filesep, name, ".mat");
            fprintf("Saving output file to %s...\n", outputFileName{this_file});
            save(outputFileName{this_file}, "data_this_file", '-v7.3')
        else
            outputFileName = [];
        end
        if keep_in_memory
            outputStruct{[this_file]} = data_this_file;
        end
    end
    disp("Done!")
end
