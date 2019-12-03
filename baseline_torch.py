"""

Simple non-learned baseline.

Input: Frame 1, Frame 2 (F1, F2)

Algorithm:
    Encode
      1. Calculate Residual_Normalized = (F2 - F1) // 2 + 127
         (this is no in range {0, ..., 255}. The idea of the //2 is to have 256 possible values, because
          otherwise we would have 511 values.)
      2. Compress Residual_Normalized with JPG
      -> Bitstream
    Decode, given F1 and Bitstream
      1. get Residual_Normalized from JPG in Bistream
      2. F2' = F1 + ( Residual_normalized - 127 ) * 2
"""
