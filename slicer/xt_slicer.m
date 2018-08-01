function xt_slicer(inputFile, out_dir)
    vid = VideoReader(inputFile);
    mkdir(out_dir);
    numFrames = vid.NumberOfFrames;
    n = numFrames;
    [y,x,z] = size(read(vid,1));
    slice = zeros(n,x,z);

    for i = 1:n
        frame = read(vid,i);
        col = frame(y/2,:,:);
        slice(i,:,:) = col;
    end

    imwrite(uint8(slice), strcat(out_dir, 'xt_slice', '.jpg'));