# Assignment 1

## Part 1

Pick up a monochrome image.

### (a)
Obtain the bit-plane mapped images for all the eight bit planes.

### (b)
Represent the image by an 8-bit gray code defined as follows:

$$
g_7 = b_7
$$

$$
g_i = b_i \oplus b_{i+1}, \quad 0 \le i \le 6
$$

where  
$b_7 b_6 \ldots b_1 b_0$ represents the binary values.

Obtain the gray-coded bit plane images for all the eight planes.

### (c)
Compare the two sets of bit-plane images in (a) and (b).  
In what sense should gray-coded bit-planes be better? Justify your answer.

---

## Part 2

### (a)
On the above bit-plane images, perform 1-D run-length coding, as described in Section-5.2.  

If each run is to be represented by a 6-bit value, calculate the compression ratio:

**(compressed bits : uncompressed bits)**

for:

- (i) binary-mapped bit-planes  
- (ii) gray-coded bit-planes  

### (b)
From the statistics of gray-coded bit-planes, obtain the probabilities of the run-lengths.

Assign Huffman codes to the run-lengths and calculate the compression ratio for the resulting encoded bit stream.

Compare this result with that of (a)-(ii).

---

## Part 3

### (a)
Using the same monochrome image, obtain the predicted image and the error image using:

$$
a_1 = a_2 = a_3 = a_4 = 0.25
$$

### (b)
Compute the histograms and the entropies of the original image and the error image.