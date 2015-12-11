function [new_img] = ensure_image_is_color_image(img)
    if length(size(img)) == 2
        new_img = ones(size(img,1), size(img, 2), 3, class(img));
        new_img(:,:,1) = img;
        new_img(:,:,2) = img;
        new_img(:,:,3) = img;
    else
        new_img = img;
	end
end
