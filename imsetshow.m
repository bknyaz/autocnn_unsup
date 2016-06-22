function im = imsetshow(images)
% Returns and shows a set of images: 4D array (rows x cols x colors x n_images)

if (iscell(images))
    images = cat(4, images{:});
end
sz = size(images);
if (length(sz) < 3)
   sz(3:4) = 1; 
elseif (length(sz) < 4)
    if (sz(3) == 3)
        sz(4) = 1;
    else
        images = permute(images, [1,2,4,3]);
        sz = size(images);
    end
end
s = sqrt(sz(4));
m = round(s);
n = ceil(s);
im = zeros(m*sz(2),n*sz(1),sz(3));
for i=1:sz(4)
    [k1,k2] = ind2sub([m,n],i);
    im(sz(2)*(k1-1)+1:k1*sz(2),sz(1)*(k2-1)+1:k2*sz(1),:) = mat2gray(images(:,:,:,i));
end
im = mat2gray(im);
imshow(im);

end