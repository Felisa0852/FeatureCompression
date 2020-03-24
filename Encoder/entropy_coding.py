#-*-coding:utf-8-*-

#
# Compression application using adaptive arithmetic coding
#
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
#
# Copyright (c) Project Nayuki
#
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#
import contextlib, sys
from Encoder.arith import arithmeticcoding
python3 = sys.version_info.major >= 3


def entropy_encode(inputfile,outputfile):
    # Perform file compression
    with open(inputfile, "rb") as inp, \
            contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
        compress(inp, bitout)


def compress(inp, bitout):
    initfreqs = arithmeticcoding.FlatFrequencyTable(257)
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    while True:
        # Read and encode one byte
        symbol = inp.read(1)
        # print(inp)
        # print('simbol:', symbol)
        # symbol = inp
        if len(symbol) == 0:
            break
        symbol = symbol[0] if python3 else ord(symbol)
        enc.write(freqs, symbol)
        freqs.increment(symbol)
    enc.write(freqs, 256)  # EOF
    enc.finish()  # Flush remaining code bits


def entropy_decode(inputfile,outputfile):
    # Handle command line arguments
    # if len(args) != 2:
    # 	sys.exit("Usage: python adaptive-arithmetic-decompress.py InputFile OutputFile")
    # inputfile, outputfile = args

    # Perform file decompression
    with open(inputfile, "rb") as inp, open(outputfile, "wb") as out:
        bitin = arithmeticcoding.BitInputStream(inp)
        decompress(bitin, out)


def decompress(bitin, out):
    initfreqs = arithmeticcoding.FlatFrequencyTable(257)
    freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    while True:
        # Decode and write one byte
        symbol = dec.read(freqs)
        if symbol == 256:  # EOF symbol
            break
        out.write(bytes((symbol,)) if python3 else chr(symbol))
        freqs.increment(symbol)



