from huffman import HuffmanCoding
import sys
import time

path = "sample.txt"

h = HuffmanCoding(path)

start = time.time()
h.create_coding()
print("Coding created in ", time.time() - start, " seconds")
start = time.time()
output_path = h.compress()
print("Compressed in ", time.time() - start, " seconds")
print("Compressed file path: " + output_path)

decom_path = h.decompress(output_path)
print("Decompressed file path: " + decom_path)