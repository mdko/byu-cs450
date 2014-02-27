pkg load image
IMG = imread('mystery.png')
GRAY = rgb2gray(IMG)

GRAYS = zeros(256,'uint8')
for r = 1:size(GRAY,1)			       # for each row
	for c = 1:size(GRAY,2)		       # for each column
		for colchan = 1:size(GRAY,3)# for each channel (here, just gray)
			colval = GRAY(r,c,colchan)
			GRAYS(colval+1)++
		end
	end
end

bar(GRAYS(1:256))
print("histogram.png")
