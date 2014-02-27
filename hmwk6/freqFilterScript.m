
%% Setup/preprocessing
% Read the raw image
FILENAME = 'input.png';
f_t = double( imread(FILENAME) );

% Convert image to a gray floating-point (double) image in range 0.0 to 1.0
if ndims(f_t) == 3
    f_t = double( rgb2gray(f_t) )/255.0;
else
    f_t = double(f_t)/255.0;
end

%% Forward FFT
% Do the forward FFT and shift to be zero-centered
F_u = fft2(f_t);
F_u = fftshift(F_u);


%% Do frequency space filtering here...
G_u = zeros( size(F_u) );
% G_u = H_u .* F_u % for example given a transfer function H_u


%% Inverse FFT
% Convert back to spatial domain
g_t = ifft2( ifftshift(G_u) );


%% Post-processing in spatial domain
imwrite( uint8(g_t*255), 'output.png' );



